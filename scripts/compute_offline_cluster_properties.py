#!/usr/bin/env python3
"""
Compute offline cluster properties for a WARP index.

This script computes metrics A1, A2, A3, A5, B5 from WARP index files and
saves the results to {index_path}/cluster_properties_offline.parquet.

Usage:
    cd /home/azureuser/repos/RAG-D
    source /anaconda/etc/profile.d/conda.sh
    conda activate warp
    
    # Compute all metrics
    python scripts/compute_offline_cluster_properties.py \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4
    
    # Print summary of existing metrics
    python scripts/compute_offline_cluster_properties.py \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4 \
        --summary-only

See docs/CLUSTER_PROPERTIES_OFFLINE.md for metric definitions.
See docs/OFFLINE_CLUSTER_PROPERTIES_INTEGRATION_PLAN.md for implementation details.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from warp.utils.offline_cluster_properties import (
    OfflineClusterPropertiesComputer,
    OfflineMetricsConfig,
    main,
)


if __name__ == "__main__":
    main()
