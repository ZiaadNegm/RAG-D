#!/usr/bin/env python3
"""
Compute Golden Document Metrics (M4R/M6R)

This script computes oracle accessibility metrics restricted to golden documents -
documents known to be relevant according to qrels.

Usage:
    python scripts/compute_golden_metrics.py \
        --index-path /path/to/warp/index \
        --metrics-dir /path/to/metrics/run \
        --qrels-path /path/to/qrels.tsv \
        --output-dir /path/to/output

Example (BEIR-Quora):
    python scripts/compute_golden_metrics.py \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4 \
        --metrics-dir /mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425 \
        --qrels-path /mnt/datasets/beir/datasets/quora/qrels/test.tsv

Output files:
    - M4R.parquet: Per (query, token, golden_doc) oracle accessibility
    - routing_status.parquet: Per (query, golden_doc) three-way classification
    - M6R.parquet: Missed centroids for golden documents
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warp.utils.golden_metrics import (
    GoldenMetricsComputer,
    GoldenMetricsConfig,
    load_qrels
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute golden document metrics (M4R/M6R)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--index-path", 
        required=True, 
        help="Path to WARP index directory"
    )
    parser.add_argument(
        "--metrics-dir", 
        required=True, 
        help="Path to metrics directory containing R0.parquet, M1.parquet"
    )
    parser.add_argument(
        "--qrels-path", 
        required=True, 
        help="Path to qrels file (TSV or JSON)"
    )
    parser.add_argument(
        "--collection-map-path",
        help="Path to collection_map.json for doc ID translation"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory (default: metrics_dir)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--skip-m4r",
        action="store_true",
        help="Skip M4R computation (if already exists)"
    )
    parser.add_argument(
        "--skip-m6r",
        action="store_true",
        help="Skip M6R computation"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate paths
    index_path = Path(args.index_path)
    metrics_dir = Path(args.metrics_dir)
    qrels_path = Path(args.qrels_path)
    
    if not index_path.exists():
        print(f"Error: Index path not found: {index_path}")
        sys.exit(1)
    
    if not metrics_dir.exists():
        print(f"Error: Metrics directory not found: {metrics_dir}")
        sys.exit(1)
        
    if not qrels_path.exists():
        print(f"Error: Qrels file not found: {qrels_path}")
        sys.exit(1)
    
    # Print banner
    if not args.quiet:
        print("=" * 70)
        print("Golden Document Metrics Computation")
        print("=" * 70)
        print(f"Start time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Index:        {index_path}")
        print(f"Metrics:      {metrics_dir}")
        print(f"Qrels:        {qrels_path}")
        print(f"Output:       {output_dir}")
        print("=" * 70)
    
    start_time = time.time()
    
    # Initialize computer
    config = GoldenMetricsConfig(verbose=not args.quiet)
    computer = GoldenMetricsComputer(
        index_path=str(index_path),
        metrics_dir=str(metrics_dir),
        config=config
    )
    
    # Load qrels with optional collection map
    if not args.quiet:
        print(f"\nLoading qrels from {qrels_path}...")
    qrels = load_qrels(str(qrels_path), args.collection_map_path)
    if not args.quiet:
        print(f"Loaded {len(qrels):,} qrel entries")
        print(f"  Unique queries: {qrels['query_id'].nunique():,}")
        print(f"  Unique docs:    {qrels['doc_id'].nunique():,}")
        if args.collection_map_path:
            print(f"  (doc IDs translated using collection_map)")
    
    m4r_path = output_dir / "M4R.parquet"
    routing_status_path = output_dir / "routing_status.parquet"
    m6r_path = output_dir / "M6R.parquet"
    
    # Compute M4R
    if not args.skip_m4r:
        m4r_start = time.time()
        m4r_df = computer.compute_m4r(qrels, output_path=str(m4r_path))
        m4r_elapsed = time.time() - m4r_start
        if not args.quiet:
            print(f"M4R computation time: {m4r_elapsed:.1f}s")
        
        # Compute routing status from M4R
        rs_start = time.time()
        routing_status_df = computer.compute_routing_status(
            m4r_df, 
            output_path=str(routing_status_path)
        )
        rs_elapsed = time.time() - rs_start
        if not args.quiet:
            print(f"Routing status computation time: {rs_elapsed:.1f}s")
    else:
        if not args.quiet:
            print("Skipping M4R computation (--skip-m4r)")
    
    # Compute M6R
    if not args.skip_m6r:
        m6r_start = time.time()
        m6r_df = computer.compute_m6r(qrels, output_path=str(m6r_path))
        m6r_elapsed = time.time() - m6r_start
        if not args.quiet:
            print(f"M6R computation time: {m6r_elapsed:.1f}s")
    else:
        if not args.quiet:
            print("Skipping M6R computation (--skip-m6r)")
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    if not args.quiet:
        print(f"\n{'='*70}")
        print("Golden Metrics Computation Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"\nOutput files:")
        if not args.skip_m4r:
            print(f"  M4R:            {m4r_path}")
            print(f"  Routing Status: {routing_status_path}")
        if not args.skip_m6r:
            print(f"  M6R:            {m6r_path}")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
