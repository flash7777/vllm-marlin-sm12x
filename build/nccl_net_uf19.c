/*
 * NCCL Net Plugin UF19 — Minimal ibverbs for 2-node TP=2
 *
 * Replaces NCCL's built-in IB transport with bare-minimum ibverbs:
 * - 1 QP per connection (no multi-QP, no multi-rail)
 * - Direct ibv_post_send/recv (no batching, no FIFO)
 * - Inline send for small messages
 * - TCP bootstrap for QP info exchange
 *
 * Build:
 *   gcc -O2 -fPIC -shared -o libnccl-net-uf19.so nccl_net_uf19.c -libverbs
 *
 * Use:
 *   NCCL_NET_PLUGIN=uf19 LD_LIBRARY_PATH=/path/to:$LD_LIBRARY_PATH ...
 *
 * Env vars:
 *   UF19_GID_INDEX  - GID index for RoCE (default: 3, or NCCL_IB_GID_INDEX)
 *   UF19_IB_DEV     - IB device name (default: auto-detect, or NCCL_IB_HCA)
 */

#include <infiniband/verbs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>

/* ================================================================
 * NCCL Net Plugin v9 API types (embedded to avoid NCCL source dep)
 * ================================================================ */

typedef enum { ncclSuccess = 0, ncclSystemError = 2, ncclInternalError = 3 } ncclResult_t;
typedef void (*ncclDebugLogger_t)(int, unsigned long, const char*, int, const char*, ...);
typedef enum { NCCL_NET_DEVICE_HOST = 0 } ncclNetDeviceType;

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_NET_MAX_REQUESTS 32
#define NCCL_NET_MAX_DEVS_PER_NIC 4

typedef struct {
    ncclNetDeviceType netDeviceType;
    int netDeviceVersion;
    void* handle;
    size_t size;
    int needsProxyProgress;
} ncclNetDeviceHandle_v9_t;

typedef struct {
    int ndevs;
    int devs[NCCL_NET_MAX_DEVS_PER_NIC];
} ncclNetVDeviceProps_v9_t;

typedef struct {
    char* name;
    char* pciPath;
    uint64_t guid;
    int ptrSupport;
    int regIsGlobal;
    int forceFlush;
    int speed;
    int port;
    float latency;
    int maxComms;
    int maxRecvs;
    ncclNetDeviceType netDeviceType;
    int netDeviceVersion;
    ncclNetVDeviceProps_v9_t vProps;
    size_t maxP2pBytes;
    size_t maxCollBytes;
} ncclNetProperties_v9_t;

typedef struct {
    const char* name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int* ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v9_t* props);
    ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
    ncclResult_t (*connect)(int dev, void* handle, void** sendComm,
                            ncclNetDeviceHandle_v9_t** sendDevComm);
    ncclResult_t (*accept)(void* listenComm, void** recvComm,
                           ncclNetDeviceHandle_v9_t** recvDevComm);
    ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
    ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                                uint64_t offset, int fd, void** mhandle);
    ncclResult_t (*deregMr)(void* comm, void* mhandle);
    ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                          void* mhandle, void** request);
    ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes,
                          int* tags, void** mhandles, void** request);
    ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                           void** mhandles, void** request);
    ncclResult_t (*test)(void* request, int* done, int* sizes);
    ncclResult_t (*closeSend)(void* sendComm);
    ncclResult_t (*closeRecv)(void* recvComm);
    ncclResult_t (*closeListen)(void* listenComm);
    ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
    ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
    ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v9_t* props);
} ncclNet_v9_t;

/* ================================================================
 * Plugin constants and globals
 * ================================================================ */

#define UF19_MAX_INLINE 256
#define UF19_CQ_DEPTH   256

static ncclDebugLogger_t g_log = NULL;
static int g_gid_idx = 3;

static struct ibv_context* g_ib_ctx = NULL;
static struct ibv_pd*      g_pd     = NULL;
static char     g_dev_name[64]  = {0};
static char     g_pci_path[256] = {0};
static uint64_t g_guid          = 0;
static int      g_port_speed    = 200000;
static int      g_active_port   = 1;

#define WARN(fmt, ...) do { \
    if (g_log) g_log(2, 0, __FILE__, __LINE__, "UF19 " fmt, ##__VA_ARGS__); \
    else fprintf(stderr, "[UF19] " fmt "\n", ##__VA_ARGS__); \
} while(0)

#define INFO(fmt, ...) do { \
    if (g_log) g_log(3, 0, __FILE__, __LINE__, "UF19 " fmt, ##__VA_ARGS__); \
    else fprintf(stderr, "[UF19] " fmt "\n", ##__VA_ARGS__); \
} while(0)

/* ================================================================
 * Data structures
 * ================================================================ */

struct uf19_comm;  /* forward */

struct uf19_request {
    int used;
    int done;
    int size;       /* actual bytes (recv only) */
    int is_send;
    struct uf19_comm* comm;  /* back-pointer for CQ poll in test() */
};

struct uf19_comm {
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    int is_send;
    struct uf19_request reqs[NCCL_NET_MAX_REQUESTS];
    int next_req;
    /* Pending connect/accept state:
     * tcp_fd >= 0  → QP exchange in progress, waiting for peer
     * tcp_fd == -1 → fully connected */
    int tcp_fd;
    int initiator;  /* 1=connect (initiator), 0=accept (responder) */
};

struct uf19_listen {
    int sock;
};

struct uf19_mr {
    struct ibv_mr* mr;
};

/* Handle exchanged via NCCL bootstrap (must fit NCCL_NET_HANDLE_MAXSIZE=128) */
struct uf19_handle {
    struct sockaddr_in addr;   /* 16 bytes: TCP listen address for QP exchange */
};

/* QP info exchanged over TCP during connect/accept */
struct uf19_qp_info {
    uint32_t     qpn;
    uint32_t     psn;
    union ibv_gid gid;
};

/* ================================================================
 * Helpers
 * ================================================================ */

static int full_write(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) return -1;
        p += n; len -= n;
    }
    return 0;
}

static int full_read(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n <= 0) return -1;
        p += n; len -= n;
    }
    return 0;
}

/* ---- QP state transitions ---- */

static int qp_to_init(struct ibv_qp* qp) {
    struct ibv_qp_attr a = {0};
    a.qp_state        = IBV_QPS_INIT;
    a.port_num         = g_active_port;
    a.pkey_index       = 0;
    a.qp_access_flags  = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ;
    return ibv_modify_qp(qp, &a,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int qp_to_rtr(struct ibv_qp* qp, uint32_t rqpn, uint32_t rpsn,
                      union ibv_gid* rgid) {
    struct ibv_qp_attr a = {0};
    a.qp_state              = IBV_QPS_RTR;
    a.path_mtu              = IBV_MTU_4096;
    a.dest_qp_num           = rqpn;
    a.rq_psn                = rpsn;
    a.max_dest_rd_atomic     = 1;
    a.min_rnr_timer          = 12;
    a.ah_attr.port_num       = g_active_port;
    a.ah_attr.is_global      = 1;
    a.ah_attr.grh.dgid       = *rgid;
    a.ah_attr.grh.sgid_index = g_gid_idx;
    a.ah_attr.grh.hop_limit  = 64;
    return ibv_modify_qp(qp, &a,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int qp_to_rts(struct ibv_qp* qp, uint32_t psn) {
    struct ibv_qp_attr a = {0};
    a.qp_state       = IBV_QPS_RTS;
    a.timeout         = 14;
    a.retry_cnt       = 7;
    a.rnr_retry       = 7;
    a.sq_psn          = psn;
    a.max_rd_atomic   = 1;
    return ibv_modify_qp(qp, &a,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

/* ---- Create QP ---- */

static struct ibv_qp* create_qp(struct ibv_cq* cq) {
    struct ibv_qp_init_attr ia = {0};
    ia.send_cq = cq;
    ia.recv_cq = cq;
    ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr    = UF19_CQ_DEPTH;
    ia.cap.max_recv_wr    = UF19_CQ_DEPTH;
    ia.cap.max_send_sge   = 1;
    ia.cap.max_recv_sge   = 1;
    ia.cap.max_inline_data = UF19_MAX_INLINE;
    ia.sq_sig_all = 0;
    return ibv_create_qp(g_pd, &ia);
}

/* ---- TCP-based QP exchange and transition to RTS ---- */

/* Returns 0=OK, -1=fatal error, -2=timeout (retry later) */
static int setup_qp_via_tcp(int fd, struct ibv_qp* qp, int initiator, int timeout_ms) {
    union ibv_gid lgid;
    if (ibv_query_gid(g_ib_ctx, g_active_port, g_gid_idx, &lgid)) {
        WARN("ibv_query_gid failed"); return -1;
    }
    struct uf19_qp_info local = {
        .qpn = qp->qp_num,
        .psn = (uint32_t)(lrand48() & 0xFFFFFF),
        .gid = lgid,
    };
    struct uf19_qp_info remote;

    /* Write our info (non-blocking safe — small payload, always fits) */
    if (full_write(fd, &local, sizeof(local))) return -1;

    /* Read peer info with timeout to avoid deadlock:
     * Both proxies call connect() first → both write → both read.
     * If peer hasn't accept()ed yet, nobody reads our data.
     * Short timeout → return -2 → NCCL retries after processing accept(). */
    struct pollfd pfd = { .fd = fd, .events = POLLIN };
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret <= 0) return -2;  /* timeout or error → retry */

    if (full_read(fd, &remote, sizeof(remote))) return -1;

    if (qp_to_init(qp))                                        return -1;
    if (qp_to_rtr(qp, remote.qpn, remote.psn, &remote.gid))   return -1;
    if (qp_to_rts(qp, local.psn))                              return -1;
    return 0;
}

/* ---- Request management ---- */

static struct uf19_request* alloc_req(struct uf19_comm* c) {
    for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
        int idx = (c->next_req + i) % NCCL_NET_MAX_REQUESTS;
        if (!c->reqs[idx].used) {
            c->next_req = (idx + 1) % NCCL_NET_MAX_REQUESTS;
            struct uf19_request* r = &c->reqs[idx];
            r->used    = 1;
            r->done    = 0;
            r->size    = 0;
            r->is_send = 0;
            r->comm    = c;
            return r;
        }
    }
    return NULL;
}

/* ---- CQ polling ---- */

static void drain_cq(struct uf19_comm* c) {
    struct ibv_wc wc[16];
    int n;
    while ((n = ibv_poll_cq(c->cq, 16, wc)) > 0) {
        for (int i = 0; i < n; i++) {
            struct uf19_request* r = (struct uf19_request*)(uintptr_t)wc[i].wr_id;
            if (wc[i].status != IBV_WC_SUCCESS) {
                WARN("WC error: status=%d opcode=%d", wc[i].status, wc[i].opcode);
                if (r) { r->done = 1; r->size = 0; }
                continue;
            }
            if (r) {
                r->done = 1;
                if (!r->is_send)
                    r->size = (int)wc[i].byte_len;
            }
        }
    }
}

/* ================================================================
 * Plugin API implementation
 * ================================================================ */

static ncclResult_t uf19_init(ncclDebugLogger_t logFn) {
    g_log = logFn;

    /* Config */
    const char* e;
    e = getenv("UF19_GID_INDEX");
    if (!e) e = getenv("NCCL_IB_GID_INDEX");
    if (e) g_gid_idx = atoi(e);

    const char* want_dev = getenv("UF19_IB_DEV");
    if (!want_dev) want_dev = getenv("NCCL_IB_HCA");

    /* Open IB device */
    int ndev;
    struct ibv_device** list = ibv_get_device_list(&ndev);
    if (!list || ndev == 0) { WARN("no IB devices"); return ncclSystemError; }

    struct ibv_device* pick = NULL;
    for (int i = 0; i < ndev; i++) {
        const char* n = ibv_get_device_name(list[i]);
        if (want_dev) {
            /* Match prefix (NCCL_IB_HCA can be "mlx5_0:1" with port) */
            if (strncmp(n, want_dev, strlen(n)) == 0) { pick = list[i]; break; }
        } else if (!pick) {
            pick = list[i];
        }
    }
    if (!pick) {
        WARN("IB device not found (wanted: %s)", want_dev ? want_dev : "auto");
        ibv_free_device_list(list);
        return ncclSystemError;
    }

    g_ib_ctx = ibv_open_device(pick);
    if (!g_ib_ctx) { WARN("ibv_open_device failed"); ibv_free_device_list(list); return ncclSystemError; }
    snprintf(g_dev_name, sizeof(g_dev_name), "%s", ibv_get_device_name(pick));

    struct ibv_device_attr da;
    ibv_query_device(g_ib_ctx, &da);
    g_guid = da.node_guid;

    /* PCI path — construct from sysfs */
    snprintf(g_pci_path, sizeof(g_pci_path), "/sys/class/infiniband/%s/device", g_dev_name);

    /* Port speed */
    struct ibv_port_attr pa;
    if (ibv_query_port(g_ib_ctx, 1, &pa) == 0) {
        g_active_port = 1;
        int w = 1;
        switch (pa.active_width) {
            case 1: w=1; break; case 2: w=4; break;
            case 4: w=8; break; case 8: w=12; break; case 16: w=2; break;
        }
        int r = 2500;
        switch (pa.active_speed) {
            case 1: r=2500; break;  case 2: r=5000; break;
            case 4: case 8: r=10000; break;
            case 16: r=14000; break; case 32: r=25000; break;
            case 64: r=50000; break;
        }
        g_port_speed = w * r;
    }
    ibv_free_device_list(list);

    g_pd = ibv_alloc_pd(g_ib_ctx);
    if (!g_pd) { WARN("ibv_alloc_pd failed"); return ncclSystemError; }

    INFO("init: dev=%s gid_idx=%d speed=%d Mbps", g_dev_name, g_gid_idx, g_port_speed);
    return ncclSuccess;
}

static ncclResult_t uf19_devices(int* ndev) {
    *ndev = 1;
    return ncclSuccess;
}

static ncclResult_t uf19_getProperties(int dev, ncclNetProperties_v9_t* p) {
    memset(p, 0, sizeof(*p));
    p->name          = g_dev_name;
    p->pciPath       = g_pci_path;
    p->guid          = g_guid;
    p->ptrSupport    = NCCL_PTR_HOST;
    p->regIsGlobal   = 0;
    p->forceFlush    = 0;
    p->speed         = g_port_speed;
    p->port          = g_active_port;
    p->latency       = 3.0f;
    p->maxComms      = 65536;
    p->maxRecvs      = 1;
    p->netDeviceType = NCCL_NET_DEVICE_HOST;
    p->maxP2pBytes   = 1ULL << 30;
    p->maxCollBytes  = 1ULL << 30;
    return ncclSuccess;
}

/* ---- listen / connect / accept ---- */

static ncclResult_t uf19_listen(int dev, void* handle, void** listenComm) {
    struct uf19_listen* lc = calloc(1, sizeof(*lc));
    if (!lc) return ncclSystemError;

    lc->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (lc->sock < 0) { free(lc); return ncclSystemError; }

    int opt = 1;
    setsockopt(lc->sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in sa = {0};
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port        = 0;   /* OS picks free port */

    if (bind(lc->sock, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        WARN("bind: %s", strerror(errno));
        close(lc->sock); free(lc); return ncclSystemError;
    }
    if (listen(lc->sock, 128) < 0) {
        close(lc->sock); free(lc); return ncclSystemError;
    }

    socklen_t sl = sizeof(sa);
    getsockname(lc->sock, (struct sockaddr*)&sa, &sl);

    /* Resolve local IP from RoCEv2 GID (IPv4 mapped in bytes 12-15) */
    union ibv_gid gid;
    ibv_query_gid(g_ib_ctx, g_active_port, g_gid_idx, &gid);

    struct uf19_handle* h = (struct uf19_handle*)handle;
    memset(h, 0, NCCL_NET_HANDLE_MAXSIZE);
    h->addr          = sa;
    memcpy(&h->addr.sin_addr.s_addr, gid.raw + 12, 4);

    *listenComm = lc;

    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &h->addr.sin_addr, ip, sizeof(ip));
    INFO("listen: %s:%d", ip, ntohs(h->addr.sin_port));
    return ncclSuccess;
}

/* Pending connect state — keyed by handle port (unique per listen socket) */
struct uf19_pending {
    uint16_t port;       /* listen port from handle */
    int      tcp_fd;     /* connected socket, QP info already written */
    struct uf19_comm* c; /* partially initialized comm */
};
static struct uf19_pending g_pending[64];
static int g_npending = 0;

static struct uf19_pending* find_pending(uint16_t port) {
    for (int i = 0; i < g_npending; i++)
        if (g_pending[i].port == port && g_pending[i].tcp_fd >= 0)
            return &g_pending[i];
    return NULL;
}

static ncclResult_t uf19_connect(int dev, void* handle, void** sendComm,
                                  ncclNetDeviceHandle_v9_t** sendDevComm) {
    struct uf19_handle* h = (struct uf19_handle*)handle;
    uint16_t hport = h->addr.sin_port;

    /* --- Resume pending connect (we already wrote, just poll for read) --- */
    struct uf19_pending* p = find_pending(hport);
    if (p) {
        struct pollfd pfd = { .fd = p->tcp_fd, .events = POLLIN };
        if (poll(&pfd, 1, 100) <= 0) {
            /* Still waiting */
            *sendComm = NULL;
            return ncclSuccess;
        }
        struct uf19_qp_info remote;
        if (full_read(p->tcp_fd, &remote, sizeof(remote))) {
            WARN("connect retry: TCP read failed");
            close(p->tcp_fd); ibv_destroy_qp(p->c->qp);
            ibv_destroy_cq(p->c->cq); free(p->c);
            p->tcp_fd = -1;
            *sendComm = NULL; return ncclSuccess;
        }
        struct uf19_comm* c = p->c;
        if (qp_to_init(c->qp) || qp_to_rtr(c->qp, remote.qpn, remote.psn, &remote.gid)
            || qp_to_rts(c->qp, (uint32_t)(uintptr_t)c->qp->qp_context)) {
            WARN("connect retry: QP transition failed");
            close(p->tcp_fd); ibv_destroy_qp(c->qp);
            ibv_destroy_cq(c->cq); free(c);
            p->tcp_fd = -1;
            *sendComm = NULL; return ncclSuccess;
        }
        close(p->tcp_fd);
        c->tcp_fd = -1;
        p->tcp_fd = -1;
        *sendComm = c;
        if (sendDevComm) *sendDevComm = NULL;
        return ncclSuccess;
    }

    /* --- Fresh connect --- */
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { *sendComm = NULL; return ncclSuccess; }

    int fl = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &fl, sizeof(fl));

    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    int ret = connect(sock, (struct sockaddr*)&h->addr, sizeof(h->addr));
    if (ret < 0 && errno != EINPROGRESS) {
        close(sock); *sendComm = NULL; return ncclSuccess;
    }

    struct pollfd pfd = { .fd = sock, .events = POLLOUT };
    ret = poll(&pfd, 1, 200);
    if (ret <= 0) { close(sock); *sendComm = NULL; return ncclSuccess; }

    int err; socklen_t el = sizeof(err);
    getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &el);
    if (err) { close(sock); *sendComm = NULL; return ncclSuccess; }

    fcntl(sock, F_SETFL, flags);

    /* Create comm + QP */
    struct uf19_comm* c = calloc(1, sizeof(*c));
    c->is_send = 1;
    c->tcp_fd = -1;
    c->cq = ibv_create_cq(g_ib_ctx, UF19_CQ_DEPTH * 2, NULL, NULL, 0);
    if (!c->cq) { close(sock); free(c); return ncclSystemError; }
    c->qp = create_qp(c->cq);
    if (!c->qp) { ibv_destroy_cq(c->cq); close(sock); free(c); return ncclSystemError; }

    /* Prepare local QP info */
    union ibv_gid lgid;
    ibv_query_gid(g_ib_ctx, g_active_port, g_gid_idx, &lgid);
    uint32_t local_psn = (uint32_t)(lrand48() & 0xFFFFFF);
    struct uf19_qp_info local = { .qpn = c->qp->qp_num, .psn = local_psn, .gid = lgid };

    /* Write our QP info (small, always succeeds immediately) */
    if (full_write(sock, &local, sizeof(local))) {
        WARN("connect: TCP write failed");
        ibv_destroy_qp(c->qp); ibv_destroy_cq(c->cq);
        close(sock); free(c); *sendComm = NULL; return ncclSuccess;
    }

    /* Try to read peer's response (accept() writes back) */
    pfd.fd = sock; pfd.events = POLLIN;
    if (poll(&pfd, 1, 200) <= 0) {
        /* Timeout — peer hasn't called accept() yet. Save pending state. */
        c->tcp_fd = sock;
        c->qp->qp_context = (void*)(uintptr_t)local_psn;  /* save for RTS later */
        if (g_npending < 64) {
            g_pending[g_npending].port = hport;
            g_pending[g_npending].tcp_fd = sock;
            g_pending[g_npending].c = c;
            g_npending++;
        }
        *sendComm = NULL;
        return ncclSuccess;
    }

    /* Peer responded */
    struct uf19_qp_info remote;
    if (full_read(sock, &remote, sizeof(remote))) {
        ibv_destroy_qp(c->qp); ibv_destroy_cq(c->cq);
        close(sock); free(c); *sendComm = NULL; return ncclSuccess;
    }
    if (qp_to_init(c->qp) || qp_to_rtr(c->qp, remote.qpn, remote.psn, &remote.gid)
        || qp_to_rts(c->qp, local_psn)) {
        WARN("connect: QP transition failed");
        ibv_destroy_qp(c->qp); ibv_destroy_cq(c->cq);
        close(sock); free(c); *sendComm = NULL; return ncclSuccess;
    }

    close(sock);
    *sendComm = c;
    if (sendDevComm) *sendDevComm = NULL;
    return ncclSuccess;
}

static ncclResult_t uf19_accept(void* listenComm, void** recvComm,
                                 ncclNetDeviceHandle_v9_t** recvDevComm) {
    struct uf19_listen* lc = (struct uf19_listen*)listenComm;

    /* Non-blocking accept */
    struct pollfd pfd = { .fd = lc->sock, .events = POLLIN };
    if (poll(&pfd, 1, 0) <= 0) { *recvComm = NULL; return ncclSuccess; }

    int sock = accept(lc->sock, NULL, NULL);
    if (sock < 0) { *recvComm = NULL; return ncclSuccess; }

    int fl = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &fl, sizeof(fl));

    /* Read peer's QP info — should be available (connect() writes before reading) */
    struct uf19_qp_info remote;
    pfd.fd = sock; pfd.events = POLLIN;
    if (poll(&pfd, 1, 2000) <= 0 || full_read(sock, &remote, sizeof(remote))) {
        WARN("accept: failed to read peer QP info");
        close(sock); *recvComm = NULL; return ncclSuccess;
    }

    /* Create local QP */
    struct uf19_comm* c = calloc(1, sizeof(*c));
    c->is_send = 0;
    c->tcp_fd = -1;
    c->cq = ibv_create_cq(g_ib_ctx, UF19_CQ_DEPTH * 2, NULL, NULL, 0);
    if (!c->cq) { close(sock); free(c); return ncclSystemError; }
    c->qp = create_qp(c->cq);
    if (!c->qp) { ibv_destroy_cq(c->cq); close(sock); free(c); return ncclSystemError; }

    /* Prepare and send our QP info back */
    union ibv_gid lgid;
    ibv_query_gid(g_ib_ctx, g_active_port, g_gid_idx, &lgid);
    struct uf19_qp_info local = {
        .qpn = c->qp->qp_num,
        .psn = (uint32_t)(lrand48() & 0xFFFFFF),
        .gid = lgid,
    };
    if (full_write(sock, &local, sizeof(local))) {
        WARN("accept: TCP write failed");
        ibv_destroy_qp(c->qp); ibv_destroy_cq(c->cq);
        close(sock); free(c); *recvComm = NULL; return ncclSuccess;
    }

    /* QP transitions */
    if (qp_to_init(c->qp) || qp_to_rtr(c->qp, remote.qpn, remote.psn, &remote.gid)
        || qp_to_rts(c->qp, local.psn)) {
        WARN("accept: QP transition failed");
        ibv_destroy_qp(c->qp); ibv_destroy_cq(c->cq);
        close(sock); free(c); *recvComm = NULL; return ncclSuccess;
    }

    close(sock);
    *recvComm = c;
    if (recvDevComm) *recvDevComm = NULL;
    return ncclSuccess;
}

/* ---- Memory registration ---- */

static ncclResult_t uf19_regMr(void* comm, void* data, size_t size, int type,
                                void** mhandle) {
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ;
    struct ibv_mr* mr = ibv_reg_mr(g_pd, data, size, access);
    if (!mr) {
        WARN("ibv_reg_mr failed: %s (ptr=%p sz=%zu type=0x%x)",
             strerror(errno), data, size, type);
        return ncclSystemError;
    }
    struct uf19_mr* h = malloc(sizeof(*h));
    h->mr = mr;
    *mhandle = h;
    return ncclSuccess;
}

static ncclResult_t uf19_regMrDmaBuf(void* comm, void* data, size_t size,
                                      int type, uint64_t off, int fd,
                                      void** mhandle) {
    return ncclSystemError;
}

static ncclResult_t uf19_deregMr(void* comm, void* mhandle) {
    struct uf19_mr* h = (struct uf19_mr*)mhandle;
    if (h) { if (h->mr) ibv_dereg_mr(h->mr); free(h); }
    return ncclSuccess;
}

/* ---- Data transfer (HOT PATH) ---- */

static ncclResult_t uf19_isend(void* sendComm, void* data, size_t size,
                                int tag, void* mhandle, void** request) {
    struct uf19_comm* c = (struct uf19_comm*)sendComm;
    struct uf19_mr* mr  = (struct uf19_mr*)mhandle;

    drain_cq(c);

    struct uf19_request* r = alloc_req(c);
    if (!r) { *request = NULL; return ncclSuccess; }
    r->is_send = 1;

    struct ibv_sge sge = {
        .addr   = (uintptr_t)data,
        .length = (uint32_t)size,
        .lkey   = mr->mr->lkey,
    };
    struct ibv_send_wr wr = {0}, *bad;
    wr.wr_id      = (uintptr_t)r;
    wr.sg_list     = &sge;
    wr.num_sge     = 1;
    wr.opcode      = IBV_WR_SEND;
    wr.send_flags  = IBV_SEND_SIGNALED;
    if (size <= UF19_MAX_INLINE)
        wr.send_flags |= IBV_SEND_INLINE;

    if (ibv_post_send(c->qp, &wr, &bad)) {
        WARN("ibv_post_send failed: %s", strerror(errno));
        r->used = 0; return ncclSystemError;
    }
    *request = r;
    return ncclSuccess;
}

static ncclResult_t uf19_irecv(void* recvComm, int n, void** data,
                                size_t* sizes, int* tags, void** mhandles,
                                void** request) {
    struct uf19_comm* c = (struct uf19_comm*)recvComm;

    drain_cq(c);

    if (n != 1) { WARN("multi-recv n=%d unsupported", n); return ncclInternalError; }

    struct uf19_mr* mr = (struct uf19_mr*)mhandles[0];
    struct uf19_request* r = alloc_req(c);
    if (!r) { *request = NULL; return ncclSuccess; }
    r->is_send = 0;

    struct ibv_sge sge = {
        .addr   = (uintptr_t)data[0],
        .length = (uint32_t)sizes[0],
        .lkey   = mr->mr->lkey,
    };
    struct ibv_recv_wr wr = {0}, *bad;
    wr.wr_id   = (uintptr_t)r;
    wr.sg_list  = &sge;
    wr.num_sge  = 1;

    if (ibv_post_recv(c->qp, &wr, &bad)) {
        WARN("ibv_post_recv failed: %s", strerror(errno));
        r->used = 0; return ncclSystemError;
    }
    *request = r;
    return ncclSuccess;
}

static ncclResult_t uf19_iflush(void* recvComm, int n, void** data,
                                 int* sizes, void** mhandles, void** request) {
    *request = NULL;
    return ncclSuccess;
}

static ncclResult_t uf19_test(void* request, int* done, int* sizes) {
    struct uf19_request* r = (struct uf19_request*)request;
    if (!r) { *done = 1; return ncclSuccess; }

    if (!r->done) {
        drain_cq(r->comm);
    }
    if (r->done) {
        *done = 1;
        if (sizes && !r->is_send)
            sizes[0] = r->size;
        r->used = 0;
    } else {
        *done = 0;
    }
    return ncclSuccess;
}

/* ---- Cleanup ---- */

static ncclResult_t uf19_closeSend(void* sendComm) {
    struct uf19_comm* c = (struct uf19_comm*)sendComm;
    if (c) {
        if (c->qp) ibv_destroy_qp(c->qp);
        if (c->cq) ibv_destroy_cq(c->cq);
        free(c);
    }
    return ncclSuccess;
}

static ncclResult_t uf19_closeRecv(void* recvComm) {
    return uf19_closeSend(recvComm);
}

static ncclResult_t uf19_closeListen(void* listenComm) {
    struct uf19_listen* lc = (struct uf19_listen*)listenComm;
    if (lc) { close(lc->sock); free(lc); }
    return ncclSuccess;
}

static ncclResult_t uf19_getDeviceMr(void* comm, void* mhandle, void** dptr) {
    return ncclInternalError;
}

static ncclResult_t uf19_irecvConsumed(void* recvComm, int n, void* request) {
    return ncclSuccess;
}

static ncclResult_t uf19_makeVDevice(int* d, ncclNetVDeviceProps_v9_t* props) {
    return ncclInternalError;
}

/* ================================================================
 * Exported plugin symbol
 * ================================================================ */

__attribute__((visibility("default")))
const ncclNet_v9_t ncclNetPlugin_v9 = {
    .name            = "UF19",
    .init            = uf19_init,
    .devices         = uf19_devices,
    .getProperties   = uf19_getProperties,
    .listen          = uf19_listen,
    .connect         = uf19_connect,
    .accept          = uf19_accept,
    .regMr           = uf19_regMr,
    .regMrDmaBuf     = uf19_regMrDmaBuf,
    .deregMr         = uf19_deregMr,
    .isend           = uf19_isend,
    .irecv           = uf19_irecv,
    .iflush          = uf19_iflush,
    .test            = uf19_test,
    .closeSend       = uf19_closeSend,
    .closeRecv       = uf19_closeRecv,
    .closeListen     = uf19_closeListen,
    .getDeviceMr     = uf19_getDeviceMr,
    .irecvConsumed   = uf19_irecvConsumed,
    .makeVDevice     = uf19_makeVDevice,
};
