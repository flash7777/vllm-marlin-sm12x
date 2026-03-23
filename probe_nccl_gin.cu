// Probe NCCL 2.29 Device API + GIN support
// Compile: nvcc -o probe_nccl_gin probe_nccl_gin.cu -lnccl -lcudart
// Run via torchrun --nnodes=2 --nproc-per-node=1 ...

#include <nccl.h>
#include <nccl_device/core.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NCCL_CHECK(cmd) do {                          \
  ncclResult_t r = cmd;                                \
  if (r != ncclSuccess) {                              \
    printf("NCCL error %d: %s @ %s:%d\n",             \
           r, ncclGetErrorString(r), __FILE__,__LINE__); \
    exit(1);                                           \
  }                                                    \
} while(0)

int main(int argc, char** argv) {
    // Get rank/world from env (torchrun sets these)
    const char* rank_str = getenv("RANK");
    const char* world_str = getenv("WORLD_SIZE");
    const char* master_addr = getenv("MASTER_ADDR");
    const char* master_port = getenv("MASTER_PORT");

    if (!rank_str || !world_str) {
        printf("Usage: torchrun --nnodes=2 --nproc-per-node=1 ... probe_nccl_gin\n");
        return 1;
    }

    int rank = atoi(rank_str);
    int world = atoi(world_str);

    printf("[Rank %d] Starting NCCL 2.29 Device API probe (world=%d)\n", rank, world);
    printf("[Rank %d] NCCL version: %d.%d.%d\n", rank, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);

    cudaSetDevice(0);

    // Init NCCL
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);

    // Broadcast ID via TCP (use NCCL's bootstrap)
    // For simplicity, use ncclCommInitRank which handles this internally
    ncclComm_t comm;

    // Use ncclCommInitRankConfig with numRmaCtx
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 4;  // Enable RMA contexts for GIN

    // Need to broadcast unique ID - use store-based bootstrap
    // Actually ncclCommInitRank handles TCP bootstrap internally
    if (rank == 0) ncclGetUniqueId(&id);

    // For multi-node, we need to share the ID. Use a simple TCP approach or MPI.
    // Since we're running via torchrun, let's use torch distributed to share the ID.
    // Actually, let's just use ncclCommInitRank which uses NCCL's internal bootstrap.

    // Simpler: just probe properties on a single-GPU comm first
    printf("[Rank %d] Creating NCCL communicator...\n", rank);

    // For a real multi-node test we'd need to broadcast the ID.
    // For now, just create a single-rank comm to probe device API support.
    ncclGetUniqueId(&id);
    NCCL_CHECK(ncclCommInitRank(&comm, 1, id, 0));

    // Query properties
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));

    printf("[Rank %d] Properties:\n", rank);
    printf("  rank=%d, nRanks=%d\n", props.rank, props.nRanks);
    printf("  cudaDev=%d, nvmlDev=%d\n", props.cudaDev, props.nvmlDev);
    printf("  deviceApiSupport=%d\n", props.deviceApiSupport);
    printf("  multimemSupport=%d\n", props.multimemSupport);
    printf("  ginType=%d (0=NONE, 2=PROXY, 3=GDAKI)\n", (int)props.ginType);

    // Try creating a DevComm
    if (props.deviceApiSupport) {
        printf("[Rank %d] Device API supported! Creating DevComm...\n", rank);

        ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.ginForceEnable = true;
        reqs.ginContextCount = 1;
        reqs.ginSignalCount = 4;
        reqs.ginCounterCount = 4;

        ncclDevComm_t* devComm = nullptr;
        ncclResult_t r = ncclDevCommCreate(comm, &reqs, devComm);
        if (r == ncclSuccess) {
            printf("[Rank %d] DevComm created successfully!\n", rank);

            // Try window registration
            void* buf;
            cudaMalloc(&buf, 65536);
            ncclWindow_t win;
            r = ncclCommWindowRegister(comm, buf, 65536, &win, NCCL_WIN_DEFAULT);
            if (r == ncclSuccess) {
                printf("[Rank %d] Window registered: buf=%p, size=64KB\n", rank, buf);
                ncclCommWindowDeregister(comm, win);
            } else {
                printf("[Rank %d] Window registration failed: %s\n", rank, ncclGetErrorString(r));
            }
            cudaFree(buf);

            ncclDevCommDestroy(comm, devComm);
        } else {
            printf("[Rank %d] DevComm creation failed: %s\n", rank, ncclGetErrorString(r));
        }
    } else {
        printf("[Rank %d] Device API NOT supported on this platform.\n", rank);
    }

    ncclCommDestroy(comm);
    printf("[Rank %d] Done.\n", rank);
    return 0;
}
