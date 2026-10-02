#!/usr/bin/env bash
set -e

echo "=== Setting up /scratch storage ==="

# Check if /scratch is already a mounted filesystem
if ! mountpoint -q /scratch; then
    # Check if the 8 local NVMe SSDs exist
    if ls /dev/nvme[1-8]n1 >/dev/null 2>&1; then
        echo "Found 8 local NVMe SSDs. Creating 3TB RAID-0 array at /dev/md0..."
        # Stop any existing md0 if partially assembled
        sudo mdadm --stop /dev/md0 2>/dev/null || true
        sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 /dev/nvme[1-8]n1 --batch --force
        echo "Formatting /dev/md0 with ext4..."
        sudo mkfs.ext4 -F /dev/md0
        sudo mkdir -p /scratch
        sudo mount -o discard,defaults /dev/md0 /scratch
    else
        echo "Local NVMe SSDs not found. Creating /scratch on local filesystem..."
        sudo mkdir -p /scratch
    fi
fi

# Ensure full permissions for non-root users
sudo chmod 777 /scratch
mkdir -p /scratch/data /scratch/out /scratch/.cache/huggingface

# Link repo directories directly to /scratch
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ln -sfn /scratch/data "$REPO_DIR/data"
ln -sfn /scratch/out "$REPO_DIR/out"

echo "=== Setup Complete ==="
df -h /scratch
echo ""
echo "Symlinks created in $REPO_DIR:"
ls -l "$REPO_DIR/data" "$REPO_DIR/out"
