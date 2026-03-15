#!/bin/bash
# UMA-optimierter Swap: Disk-Swap weg, zram mit zstd
# Ausfuehren als root auf DGX Spark + PGX ThinkStation
#
# sudo bash setup_uma_swap.sh

set -euo pipefail

echo "=== UMA Swap Setup ==="

# 1. Aktive Disk-Swaps ermitteln und deaktivieren
echo "Deaktiviere Disk-Swap..."
DISK_SWAPS=$(swapon --show=NAME,TYPE --noheadings 2>/dev/null | awk '$2 != "zram" && $2 != "partition" {print $1}; $2 == "file" {print $1}; $2 == "partition" {print $1}')
if [ -z "$DISK_SWAPS" ]; then
    DISK_SWAPS=$(swapon --show=NAME --noheadings 2>/dev/null | grep -v zram)
fi
for swap in $DISK_SWAPS; do
    swapoff "$swap" 2>/dev/null && echo "  $swap deaktiviert" || echo "  $swap: bereits aus"
done
[ -z "$DISK_SWAPS" ] && echo "  Kein Disk-Swap aktiv"

# 2. Aus fstab entfernen (alle swap-Eintraege auskommentieren)
if grep -qE '^[^#].*\bswap\b' /etc/fstab; then
    sed -i 's|^\([^#].*\bswap\b.*\)|#\1|' /etc/fstab
    echo "  fstab: swap-Eintraege auskommentiert"
fi

# 3. Docker + containerd deaktivieren (nicht genutzt, spart ~150 MB)
for svc in docker containerd; do
    if systemctl is-active --quiet $svc 2>/dev/null; then
        systemctl disable --now $svc
        echo "  $svc deaktiviert"
    fi
done

# 4. zram mit zstd
echo "Aktiviere zram (16G, zstd)..."
modprobe zram num_devices=1

# Reset falls bereits konfiguriert
echo 1 > /sys/block/zram0/reset 2>/dev/null || true

echo zstd > /sys/block/zram0/comp_algorithm
echo 16G > /sys/block/zram0/disksize
mkswap /dev/zram0
swapon -p 100 /dev/zram0
echo "  zram0: 16G zstd aktiv"

# 5. Swappiness niedrig (nur CPU-Druck swappen, GPU-Allocs nie)
echo 10 > /proc/sys/vm/swappiness
echo "  swappiness: 10"

# 6. Dauerhaft machen
cat > /etc/systemd/system/zram-swap.service << 'EOF'
[Unit]
Description=zram swap (UMA optimiert)
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'modprobe zram num_devices=1; echo 1 > /sys/block/zram0/reset 2>/dev/null || true; echo zstd > /sys/block/zram0/comp_algorithm; echo 16G > /sys/block/zram0/disksize; mkswap /dev/zram0; swapon -p 100 /dev/zram0; echo 10 > /proc/sys/vm/swappiness'
ExecStop=/bin/bash -c 'swapoff /dev/zram0 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable zram-swap.service
echo "  systemd service installiert (persistent nach reboot)"

# 7. Verify
echo ""
echo "=== Status ==="
swapon --show
echo ""
free -h | head -3
echo ""
cat /sys/block/zram0/comp_algorithm | tr ' ' '\n' | grep '\[' || echo "  algo: $(cat /sys/block/zram0/comp_algorithm)"
echo "swappiness: $(cat /proc/sys/vm/swappiness)"
echo ""
echo "Done. Disk-Swap weg, zram zstd 16G aktiv."
