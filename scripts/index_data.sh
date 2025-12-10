#!/usr/bin/bash
# Indexing script for WARP
# Activates conda environment and builds indexes
# Note: GPU is strongly recommended for indexing

set -e  # Exit on error

echo "=== WARP Indexing Script ==="

# Navigate to repo root
cd /home/azureuser/repos/xtr-warp

# Load environment variables
set -o allexport
source .env
set +o allexport

# Activate conda environment
echo "Activating conda environment 'warp'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate warp

# Fix: Use conda's libstdc++ (has CXXABI_1.3.15 required by faiss-gpu)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Using Python: $(which python)"
echo "Index root: ${INDEX_ROOT}"
echo ""

# Build Indexes for BEIR/test (nbits=4)
echo "=== Building BEIR Indexes ==="

BEIR=("quora")
for dataset in "${BEIR[@]}"; do
    echo "Indexing: $dataset"
    python utils.py index -c beir -d "$dataset" -s test -n 4
done

echo ""
echo "=== Indexing Complete ==="
echo "Indexes saved to: ${INDEX_ROOT}"
ls -la "${INDEX_ROOT}"
