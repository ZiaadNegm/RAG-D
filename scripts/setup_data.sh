#!/usr/bin/bash
# Setup script for WARP - run this when starting your VM
# This creates directories and downloads the BEIR dataset(s)

set -e  # Exit on error

echo "=== WARP Data Setup Script ==="

# Create directories
echo "Creating directories on /mnt..."
sudo mkdir -p /mnt/datasets/index
sudo mkdir -p /mnt/datasets/BEIR
sudo mkdir -p /mnt/datasets/LOTTE

# Set ownership to current user
echo "Setting ownership to $(whoami)..."
sudo chown -R $(whoami):$(whoami) /mnt/datasets

echo "Directories created:"
ls -la /mnt/datasets/

# Load environment variables
cd /home/azureuser/repos/xtr-warp
set -o allexport
source .env
set +o allexport

# Download BEIR datasets
echo ""
echo "=== Downloading BEIR Datasets ==="

BEIR=("quora")
for dataset in "${BEIR[@]}"; do
    echo "Downloading and extracting: $dataset"
    python utility/extract_collection.py -d "$dataset" -i "${BEIR_COLLECTION_PATH}" -s test
done

echo ""
echo "=== Setup Complete ==="
echo "Data location: ${BEIR_COLLECTION_PATH}"
echo "Index location: ${INDEX_ROOT}"
echo ""
echo "Next step: Run ./scripts/index_data.sh to build indexes"
