#!/usr/bin/env python3
"""
Compute derived metrics (M2, M5, M6) for a WARP measurement run.

These metrics are derived from the raw measurements (M1, M3, M4, R0) produced
during WARP search with measurement tracking enabled.

Usage:
    # Compute all derived metrics for a run
    python scripts/compute_derived_metrics.py --run-dir /mnt/warp_measurements/runs/my_run \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4
    
    # Dry run: check source files without computing
    python scripts/compute_derived_metrics.py --run-dir /mnt/warp_measurements/runs/my_run --dry-run
    
    # Compute specific metrics only
    python scripts/compute_derived_metrics.py --run-dir /mnt/warp_measurements/runs/my_run \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4 --metrics M2 M5
    
    # Compute M5 for top-k docs only (comparable to M4 E2E test results)
    python scripts/compute_derived_metrics.py --run-dir /mnt/warp_measurements/runs/my_run \
        --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4 --top-k-only

See docs/M2_M5_M6_INTEGRATION_PLAN.md for metric definitions.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warp.utils.derived_metrics import DerivedMetricsComputer


def find_latest_run(base_dir: str = "/mnt/warp_measurements/runs") -> Path:
    """Find the most recent measurement run directory."""
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def main():
    parser = argparse.ArgumentParser(
        description="Compute derived metrics (M2, M5, M6) for WARP measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  M2  Redundant computation: M1 (total sims) - M3 (influential pairs)
  M5  Routing misses: Oracle winner's centroid not in selected centroids
  M6  Missed centroid aggregation: Which centroids cause the most misses

Examples:
  # Compute all metrics
  python scripts/compute_derived_metrics.py --run-dir /path/to/run

  # Only check what files exist (dry run)
  python scripts/compute_derived_metrics.py --run-dir /path/to/run --dry-run
        """
    )
    parser.add_argument(
        "--run-dir",
        help="Path to measurement run directory. If not provided, uses most recent run."
    )
    parser.add_argument(
        "--index-path",
        help="Path to WARP index (required for M5/M6). Auto-detected from metadata if not provided."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["M2", "M5", "M6"],
        choices=["M2", "M5", "M6"],
        help="Which metrics to compute (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print what would be computed, don't write files"
    )
    parser.add_argument(
        "--top-k-only",
        action="store_true",
        help="For M5/M6, only compute for docs in top-k (comparable to M4 E2E test results)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing measurements, don't compute anything"
    )
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            print("Error: No run directory found. Specify with --run-dir or run a measurement collection first.")
            sys.exit(1)
        print(f"Using latest run: {run_dir}")
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Get index path from metadata if not provided
    index_path = args.index_path
    if not index_path:
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            index_path = metadata.get("index", {}).get("path")
            if index_path:
                print(f"Using index path from metadata: {index_path}")
    
    # Initialize computer
    try:
        computer = DerivedMetricsComputer(
            run_dir=str(run_dir),
            index_path=index_path
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Summary only mode
    if args.summary_only:
        computer.print_summary()
        return
    
    # Dry run mode
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - No files will be written")
        print("=" * 60)
        print(f"\nWould compute metrics: {args.metrics}")
        print(f"Run directory: {run_dir}")
        print(f"Index path: {index_path or '(not set - needed for M5/M6)'}")
        
        print("\nSource files:")
        for name, path in [
            ("M1", run_dir / "tier_a/M1_compute_per_centroid.parquet"),
            ("M3", run_dir / "tier_b/M3_observed_winners.parquet"),
            ("M4", run_dir / "tier_b/M4_oracle_winners.parquet"),
            ("R0", run_dir / "tier_b/R0_selected_centroids.parquet"),
        ]:
            status = "✓" if path.exists() else "✗"
            size = f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else ""
            print(f"  {status} {name}: {path.name} {size}")
        
        print("\nOutput files (would be created):")
        if "M2" in args.metrics:
            print(f"  tier_a/M2_redundant_computation.parquet")
        if "M5" in args.metrics:
            print(f"  tier_b/M5_routing_misses.parquet")
        if "M6" in args.metrics:
            print(f"  tier_a/M6_missed_centroids_global.parquet")
            print(f"  tier_b/M6_per_query.parquet")
        
        return
    
    # Compute requested metrics
    results = {}
    
    if "M2" in args.metrics:
        print("\n" + "=" * 60)
        print("Computing M2 (Redundant Computation)")
        print("=" * 60)
        try:
            m2 = computer.compute_m2()
            results["M2"] = m2
            print(f"\nResults:")
            print(f"  Queries: {len(m2):,}")
            print(f"  Mean redundancy rate: {m2['redundancy_rate'].mean():.1%}")
            print(f"  Min redundancy rate: {m2['redundancy_rate'].min():.1%}")
            print(f"  Max redundancy rate: {m2['redundancy_rate'].max():.1%}")
        except Exception as e:
            print(f"Error computing M2: {e}")
    
    if "M5" in args.metrics or "M6" in args.metrics:
        if not index_path:
            print("\nError: --index-path required for M5/M6 computation")
            print("Provide it explicitly or ensure metadata.json contains index path")
            sys.exit(1)
    
    if "M5" in args.metrics:
        print("\n" + "=" * 60)
        print("Computing M5 (Routing Misses)")
        if args.top_k_only:
            print("(top-k docs only)")
        print("=" * 60)
        try:
            m5 = computer.compute_m5(top_k_only=args.top_k_only)
            results["M5"] = m5
            print(f"\nResults:")
            print(f"  Total rows: {len(m5):,}")
            print(f"  Total misses: {m5['is_miss'].sum():,}")
            print(f"  Miss rate: {m5['is_miss'].mean():.2%}")
            if "score_delta" in m5.columns:
                misses = m5[m5["is_miss"]]
                if len(misses) > 0:
                    print(f"  Mean score_delta on miss: {misses['score_delta'].mean():.4f}")
                    print(f"  Max score_delta: {misses['score_delta'].max():.4f}")
        except Exception as e:
            print(f"Error computing M5: {e}")
            import traceback
            traceback.print_exc()
    
    if "M6" in args.metrics:
        print("\n" + "=" * 60)
        print("Computing M6 (Missed Centroid Aggregation)")
        print("=" * 60)
        try:
            m5_df = results.get("M5")
            m6_global, m6_per_query = computer.compute_m6(m5_df=m5_df)
            results["M6_global"] = m6_global
            results["M6_per_query"] = m6_per_query
            
            print(f"\nGlobal Results:")
            print(f"  Centroids with oracle wins: {len(m6_global):,}")
            print(f"  Centroids with any miss: {(m6_global['miss_count'] > 0).sum():,}")
            print(f"  Mean miss rate: {m6_global['miss_rate'].mean():.2%}")
            
            print(f"\nTop 5 problem centroids (by miss count):")
            top5 = m6_global.nlargest(5, "miss_count")
            for _, row in top5.iterrows():
                print(f"  Centroid {int(row['oracle_centroid_id'])}: "
                      f"{int(row['miss_count']):,} misses / {int(row['oracle_win_count']):,} wins "
                      f"({row['miss_rate']:.1%})")
        except Exception as e:
            print(f"Error computing M6: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    computer.print_summary()


if __name__ == "__main__":
    main()
