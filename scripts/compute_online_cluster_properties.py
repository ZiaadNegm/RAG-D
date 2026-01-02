#!/usr/bin/env python3
"""
Compute online cluster properties from WARP measurement data.

Usage:
    python scripts/compute_online_cluster_properties.py \
        --run-dir /mnt/warp_measurements/runs/my_run \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4

    # Dry run (print summary only)
    python scripts/compute_online_cluster_properties.py \
        --run-dir /mnt/warp_measurements/runs/my_run \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4 \
        --summary-only

See docs/CLUSTER_PROPERTIES_ONLINE.md for metric definitions.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Compute online cluster properties from WARP measurement data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compute_online_cluster_properties.py --run-dir /path/to/run --index-path /path/to/index
  python scripts/compute_online_cluster_properties.py --run-dir /path/to/run --summary-only
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to measurement run directory (contains tier_a/, tier_b/)"
    )
    parser.add_argument(
        "--index-path", type=str, required=True,
        help="Path to WARP index directory"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: {run_dir}/cluster_properties_online)"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only print summary of existing metrics (no computation)"
    )
    
    # Hub classification thresholds
    parser.add_argument(
        "--hub-percentile", type=float, default=95.0,
        help="Percentile threshold for hub classification (default: 95.0)"
    )
    parser.add_argument(
        "--bad-yield-threshold", type=float, default=0.1,
        help="Yield below this → bad hub (default: 0.1)"
    )
    parser.add_argument(
        "--good-yield-threshold", type=float, default=0.3,
        help="Yield above this → good hub (default: 0.3)"
    )
    
    # Parallelization
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Number of workers for heavy metrics (default: 8)"
    )
    
    # Verbosity
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    
    index_path = Path(args.index_path)
    if not index_path.exists():
        print(f"Error: Index path not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    
    # Import here to avoid slow startup for --help
    from warp.utils.online_cluster_properties import (
        OnlineClusterPropertiesComputer,
        OnlineMetricsConfig,
    )
    
    # Create config
    config = OnlineMetricsConfig(
        hub_percentile=args.hub_percentile,
        bad_yield_threshold=args.bad_yield_threshold,
        good_yield_threshold=args.good_yield_threshold,
        num_workers=args.num_workers,
    )
    
    # Create computer
    output_dir = args.output_dir or str(run_dir / "cluster_properties_online")
    
    computer = OnlineClusterPropertiesComputer(
        run_dir=str(run_dir),
        index_path=str(index_path),
        output_dir=output_dir,
        config=config,
        verbose=not args.quiet,
    )
    
    if args.summary_only:
        computer.print_summary()
        return
    
    # Compute all metrics
    results = computer.compute_all(save=True)
    
    # Print final summary
    computer.print_summary()


if __name__ == "__main__":
    main()
