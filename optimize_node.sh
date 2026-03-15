#!/bin/bash
# UMA Node Optimierung — maximiert freien RAM fuer GPU Inference
# Ausfuehren als root: sudo bash optimize_node.sh
#
# Idempotent: kann mehrfach aufgerufen werden ohne Schaden.
# Erkennt was bereits optimiert ist und ueberspringt es.
#
# Sicher fuer DGX Spark + PGX ThinkStation
# Behaelt: SSH, Podman, NVIDIA, RDMA/Mellanox, Netzwerk

set -euo pipefail

ZRAM_SIZE="2G"
ZRAM_ALGO="zstd"
TARGET_SWAPPINESS=10

echo "=== UMA Node Optimierung ==="
echo "Host: $(hostname)"
echo ""

FREED=0

# ============================================================
# 1. Disk-Swap deaktivieren
# ============================================================
echo "--- Disk-Swap ---"
DISK_SWAPS=$(swapon --show=NAME,TYPE --noheadings 2>/dev/null | awk '$2!="zram"{print $1}')
if [ -n "$DISK_SWAPS" ]; then
    for swap in $DISK_SWAPS; do
        swapoff "$swap" 2>/dev/null && echo "  $swap deaktiviert" || true
    done
else
    echo "  Kein Disk-Swap aktiv"
fi

# fstab: swap-Zeilen auskommentieren
if grep -qE '^[^#].*\bswap\b' /etc/fstab; then
    sed -i 's|^\([^#].*\bswap\b.*\)|#\1|' /etc/fstab
    echo "  fstab: swap auskommentiert"
fi

# ============================================================
# 2. zram (idempotent: prüft Größe + Algorithmus)
# ============================================================
echo ""
echo "--- zram ---"
ZRAM_OK=false
if [ -e /sys/block/zram0/disksize ]; then
    CURRENT_SIZE=$(cat /sys/block/zram0/disksize 2>/dev/null || echo 0)
    CURRENT_ALGO=$(cat /sys/block/zram0/comp_algorithm 2>/dev/null | grep -oP '\[\K[^\]]+' || echo "none")
    TARGET_BYTES=$(numfmt --from=iec "$ZRAM_SIZE" 2>/dev/null || echo 0)

    if [ "$CURRENT_SIZE" = "$TARGET_BYTES" ] && [ "$CURRENT_ALGO" = "$ZRAM_ALGO" ] && swapon --show 2>/dev/null | grep -q zram; then
        echo "  zram0: bereits aktiv ($ZRAM_SIZE $ZRAM_ALGO)"
        ZRAM_OK=true
    fi
fi

if ! $ZRAM_OK; then
    # zram neu aufsetzen
    swapoff /dev/zram0 2>/dev/null || true
    modprobe zram num_devices=1 2>/dev/null || true
    echo 1 > /sys/block/zram0/reset 2>/dev/null || true
    echo "$ZRAM_ALGO" > /sys/block/zram0/comp_algorithm
    echo "$ZRAM_SIZE" > /sys/block/zram0/disksize
    mkswap /dev/zram0 > /dev/null
    swapon -p 100 /dev/zram0
    echo "  zram0: $ZRAM_SIZE $ZRAM_ALGO aktiviert"
fi

# Systemd-Service fuer Persistenz (einmalig)
if [ ! -f /etc/systemd/system/zram-swap.service ]; then
    cat > /etc/systemd/system/zram-swap.service << SVCEOF
[Unit]
Description=zram swap (UMA optimiert)
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'modprobe zram num_devices=1; echo 1 > /sys/block/zram0/reset 2>/dev/null || true; echo $ZRAM_ALGO > /sys/block/zram0/comp_algorithm; echo $ZRAM_SIZE > /sys/block/zram0/disksize; mkswap /dev/zram0; swapon -p 100 /dev/zram0; echo $TARGET_SWAPPINESS > /proc/sys/vm/swappiness'
ExecStop=/bin/bash -c 'swapoff /dev/zram0 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
SVCEOF
    systemctl daemon-reload
    systemctl enable zram-swap.service 2>/dev/null
    echo "  systemd-service installiert (persistent)"
fi

# ============================================================
# 3. Swappiness
# ============================================================
CURRENT_SWAPPINESS=$(cat /proc/sys/vm/swappiness)
if [ "$CURRENT_SWAPPINESS" != "$TARGET_SWAPPINESS" ]; then
    echo "$TARGET_SWAPPINESS" > /proc/sys/vm/swappiness
    echo "  Swappiness: $CURRENT_SWAPPINESS -> $TARGET_SWAPPINESS"
else
    echo "  Swappiness: $TARGET_SWAPPINESS (bereits)"
fi

# ============================================================
# 4. Unnoetige Services stoppen + deaktivieren
# ============================================================
echo ""
echo "--- Services ---"
for svc in docker containerd snapd fwupd \
  cups cups-browsed \
  bluetooth ModemManager avahi-daemon wpa_supplicant \
  nfsd nfs-mountd nfs-blkmap nfs-idmapd nfsdcld rpc-statd rpcbind fsidd \
  multipathd rasdaemon smartmontools \
  dgx-dashboard dgx-dashboard-admin \
  udisks2 upower; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then
    MEM=$(systemctl show "$svc" --property=MemoryCurrent 2>/dev/null | cut -d= -f2)
    systemctl disable --now "$svc" 2>/dev/null
    MB=$((${MEM:-0} / 1048576))
    FREED=$((FREED + MB))
    echo "  $svc gestoppt (+${MB} MB)"
  fi
done
[ "$FREED" -eq 0 ] && echo "  Alle bereits gestoppt"

# ============================================================
# 5. Desktop-Prozesse killen
# ============================================================
echo ""
echo "--- Desktop-Prozesse ---"
KILLED_ANY=false
for proc in wireplumber pipewire pulseaudio snapd-desktop-integration \
  gsd- gnome- tracker- evolution-; do
  if pkill -f "$proc" 2>/dev/null; then
    echo "  $proc gekillt"
    KILLED_ANY=true
  fi
done
$KILLED_ANY || echo "  Keine Desktop-Prozesse aktiv"

# ============================================================
# 6. Kernel Module entladen (nur wenn refcount=0, skip wenn riskant)
# ============================================================
echo ""
echo "--- Kernel Module ---"
UNLOADED=false
# NUR Module mit refcount 0 entladen, Reihenfolge: abhaengige zuerst
for mod in btusb btrtl btmtk btintel btbcm bnep bluetooth \
  mt7925e mt7925_common mt792x_lib mt76_connac_lib mt76 mac80211 cfg80211 \
  snd_hda_codec_hdmi snd_hda_codec_tegrahdmi snd_hda_acpi snd_hda_codec snd_hda_core snd_pcm snd_timer snd soundcore \
  btrfs raid456; do
  # NICHT entladen: r8127 (Realtek LAN!), sunrpc/nfs (Kernel-Panic)
  REFCOUNT=$(lsmod 2>/dev/null | awk -v m="$mod" '$1==m{print $3}')
  if [ -n "$REFCOUNT" ] && [ "$REFCOUNT" = "0" ]; then
    if rmmod "$mod" 2>/dev/null; then
      echo "  $mod entladen"
      UNLOADED=true
    fi
  fi
done
# NFS/sunrpc: NICHT entladen (kann Kernel-Panic ausloesen wenn Sockets offen)
# Stattdessen nur Services stoppen (oben)
$UNLOADED || echo "  Keine Module mit refcount=0"

# ============================================================
# 7. journald RAM begrenzen
# ============================================================
echo ""
echo "--- journald ---"
JOURNAL_CONF="/etc/systemd/journald.conf.d/uma.conf"
if [ ! -f "$JOURNAL_CONF" ]; then
    mkdir -p /etc/systemd/journald.conf.d
    cat > "$JOURNAL_CONF" << 'EOF'
[Journal]
RuntimeMaxUse=32M
SystemMaxUse=100M
EOF
    systemctl restart systemd-journald
    echo "  Begrenzt auf 32M RAM / 100M Disk"
else
    echo "  Bereits konfiguriert"
fi

# ============================================================
# 8. Page Cache droppen
# ============================================================
echo ""
echo "--- Page Cache ---"
sync
echo 3 > /proc/sys/vm/drop_caches
echo "  Cache gedroppt"

# ============================================================
# Ergebnis
# ============================================================
echo ""
echo "========================================="
echo "  Host:       $(hostname)"
echo "  Services:   +${FREED} MB befreit"
free -h | awk '/Speicher/{printf "  RAM:        %s frei / %s\n", $4, $2}'
swapon --show --noheadings 2>/dev/null | awk '{printf "  Swap:       %s %s (%s)\n", $1, $3, $5}' || echo "  Swap: keiner"
echo "  Swappiness: $(cat /proc/sys/vm/swappiness)"
echo "========================================="
echo ""
echo "Top 5 Prozesse:"
ps aux --sort=-%mem | awk 'NR>1 && NR<=6{printf "  %5d MB  %s\n", int($6/1024), $11}'
