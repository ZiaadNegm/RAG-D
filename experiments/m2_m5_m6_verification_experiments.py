#!/usr/bin/env python3
"""
M2, M5, M6 Verification Experiments

These experiments validate the derived metrics computation:
- E1: Verify source data availability and schema
- E2: M2 computation validation (redundancy formula)
- E3: M5 centroid lookup performance (storage strategy decision)
- E4: Memory scalability (chunking decision)
- E5: M6 aggregation validation

Run with:
    conda activate warp
    python experiments/m2_m5_m6_verification_experiments.py [--run-dir PATH] [--index-path PATH] [--experiment E1|E2|E3|E4|E5|all]
    
Prerequisites:
    A completed measurement run with M1, M3, M4, R0 parquet files.
    Run a measurement collection first:
        python experiments/m4_e2e_test.py --experiment E2
"""

import os
import sys
import time
import argparse
import gc
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_run(base_dir: str = "/mnt/warp_measurements/runs") -> Path:
    """Find the most recent measurement run directory."""
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def load_offsets_compacted(index_path: str) -> torch.Tensor:
    """Load or compute offsets_compacted from index."""
    sizes = torch.load(os.path.join(index_path, "sizes.compacted.pt"))
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    torch.cumsum(sizes, dim=0, out=offsets[1:])
    return offsets


def experiment_e1_verify_source_data(run_dir: Path) -> bool:
    """
    E1: Verify source data availability and schema.
    
    Checks that all required Parquet files exist and have expected columns.
    """
    print("\n" + "=" * 60)
    print("E1: Verify Source Data Availability")
    print("=" * 60)
    
    files = {
        "M1": {
            "path": run_dir / "tier_a/M1_compute_per_centroid.parquet",
            "required_cols": ["query_id", "q_token_id", "centroid_id", "num_token_token_sims"]
        },
        "M3": {
            "path": run_dir / "tier_b/M3_observed_winners.parquet", 
            "required_cols": ["query_id", "q_token_id", "doc_id", "winner_embedding_pos", "winner_score"]
        },
        "M4": {
            "path": run_dir / "tier_b/M4_oracle_winners.parquet",
            "required_cols": ["query_id", "q_token_id", "doc_id", "oracle_embedding_pos", "oracle_score"]
        },
        "R0": {
            "path": run_dir / "tier_b/R0_selected_centroids.parquet",
            "required_cols": ["query_id", "q_token_id", "centroid_id", "rank"]
        },
    }
    
    all_passed = True
    
    for name, info in files.items():
        path = info["path"]
        required_cols = info["required_cols"]
        
        if not path.exists():
            print(f"✗ {name}: MISSING at {path}")
            all_passed = False
            continue
        
        try:
            df = pd.read_parquet(path)
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                print(f"✗ {name}: Missing columns: {missing_cols}")
                all_passed = False
            else:
                print(f"✓ {name}: {len(df):,} rows")
                print(f"    Columns: {list(df.columns)}")
                print(f"    Sample query_ids: {sorted(df['query_id'].unique()[:5])}")
                
        except Exception as e:
            print(f"✗ {name}: Error reading: {e}")
            all_passed = False
    
    print("\n" + "-" * 40)
    if all_passed:
        print("E1 PASSED: All source files available with correct schema")
    else:
        print("E1 FAILED: Some source files missing or malformed")
    
    return all_passed


def experiment_e2_m2_computation(run_dir: Path) -> bool:
    """
    E2: M2 computation validation.
    
    Validates that M2 = M1 - M3_count produces sensible results.
    """
    print("\n" + "=" * 60)
    print("E2: M2 Computation Validation")
    print("=" * 60)
    
    # Load M1
    m1_path = run_dir / "tier_a/M1_compute_per_centroid.parquet"
    m1 = pd.read_parquet(m1_path)
    print(f"M1 loaded: {len(m1):,} rows")
    
    # Aggregate M1 per query
    m1_per_query = m1.groupby("query_id")["num_token_token_sims"].sum().reset_index()
    m1_per_query.columns = ["query_id", "m1_total_sims"]
    print(f"M1 per query: {len(m1_per_query)} queries")
    print(f"  Mean M1: {m1_per_query['m1_total_sims'].mean():,.0f}")
    print(f"  Min M1: {m1_per_query['m1_total_sims'].min():,}")
    print(f"  Max M1: {m1_per_query['m1_total_sims'].max():,}")
    
    # Load M3
    m3_path = run_dir / "tier_b/M3_observed_winners.parquet"
    m3 = pd.read_parquet(m3_path)
    print(f"\nM3 loaded: {len(m3):,} rows")
    
    # Count unique (token, doc) pairs per query
    m3_per_query = m3.groupby("query_id").size().reset_index(name="m3_influential_pairs")
    print(f"M3 per query: {len(m3_per_query)} queries")
    print(f"  Mean M3: {m3_per_query['m3_influential_pairs'].mean():,.0f}")
    print(f"  Min M3: {m3_per_query['m3_influential_pairs'].min():,}")
    print(f"  Max M3: {m3_per_query['m3_influential_pairs'].max():,}")
    
    # Compute M2
    result = m1_per_query.merge(m3_per_query, on="query_id", how="outer").fillna(0)
    result["m2_redundant_sims"] = result["m1_total_sims"] - result["m3_influential_pairs"]
    result["redundancy_rate"] = result["m2_redundant_sims"] / result["m1_total_sims"]
    
    print("\n" + "-" * 40)
    print("M2 Results:")
    print(f"  Mean M2 (redundant sims): {result['m2_redundant_sims'].mean():,.0f}")
    print(f"  Mean redundancy rate: {result['redundancy_rate'].mean():.1%}")
    print(f"  Min redundancy rate: {result['redundancy_rate'].min():.1%}")
    print(f"  Max redundancy rate: {result['redundancy_rate'].max():.1%}")
    
    # Validation checks
    all_passed = True
    
    # Check 1: No negative M2 values
    negative_m2 = (result["m2_redundant_sims"] < 0).sum()
    if negative_m2 > 0:
        print(f"\n✗ VALIDATION FAILED: {negative_m2} queries have negative M2 (M3 > M1)")
        all_passed = False
    else:
        print(f"\n✓ All M2 values are non-negative")
    
    # Check 2: Redundancy rate in expected range (50-99%)
    mean_rate = result['redundancy_rate'].mean()
    if 0.5 <= mean_rate <= 0.99:
        print(f"✓ Mean redundancy rate ({mean_rate:.1%}) in expected range [50%, 99%]")
    else:
        print(f"⚠ Mean redundancy rate ({mean_rate:.1%}) outside expected range")
    
    # Check 3: M1 > M3 for all queries
    m1_gt_m3 = (result["m1_total_sims"] > result["m3_influential_pairs"]).all()
    if m1_gt_m3:
        print(f"✓ M1 > M3 for all queries (as expected)")
    else:
        print(f"✗ Some queries have M3 >= M1 (unexpected)")
        all_passed = False
    
    print("\n" + "-" * 40)
    if all_passed:
        print("E2 PASSED: M2 computation produces valid results")
    else:
        print("E2 FAILED: M2 validation errors detected")
    
    return all_passed


def experiment_e3_centroid_lookup(run_dir: Path, index_path: str) -> dict:
    """
    E3: M5 centroid lookup performance.
    
    Tests:
    - Single vs batch centroid lookup performance
    - Storage overhead of pre-computing oracle_centroid_id
    - JOIN performance for M3+M4
    """
    print("\n" + "=" * 60)
    print("E3: Centroid Lookup Performance")
    print("=" * 60)
    
    results = {}
    
    # Load offsets
    offsets = load_offsets_compacted(index_path)
    print(f"Loaded offsets_compacted: {len(offsets):,} centroids + 1")
    
    # Part A: Single vs Batch Lookup
    print("\n--- Part A: Lookup Performance ---")
    
    # Generate random positions
    num_positions = 100_000
    max_pos = offsets[-1].item()
    positions = torch.randint(0, max_pos, (num_positions,), dtype=torch.long)
    
    # Single lookups (sample of 1000)
    single_sample = 1000
    start = time.perf_counter()
    for pos in positions[:single_sample]:
        _ = torch.searchsorted(offsets, pos.unsqueeze(0), side="right").item() - 1
    single_time_total = time.perf_counter() - start
    single_time_per = single_time_total / single_sample
    print(f"Single lookup: {single_time_per*1e6:.1f} µs per position ({single_sample} samples)")
    
    # Batch lookup
    start = time.perf_counter()
    centroids = torch.searchsorted(offsets, positions, side="right") - 1
    batch_time_total = time.perf_counter() - start
    batch_time_per = batch_time_total / num_positions
    print(f"Batch lookup: {batch_time_per*1e6:.2f} µs per position ({num_positions} samples)")
    print(f"Speedup: {single_time_per / batch_time_per:.0f}x")
    
    results["single_lookup_us"] = single_time_per * 1e6
    results["batch_lookup_us"] = batch_time_per * 1e6
    results["lookup_speedup"] = single_time_per / batch_time_per
    
    # Part B: Storage Overhead
    print("\n--- Part B: Storage Overhead ---")
    
    m4_path = run_dir / "tier_b/M4_oracle_winners.parquet"
    m4 = pd.read_parquet(m4_path)
    
    current_size = m4.memory_usage(deep=True).sum() / 1e6
    with_centroid_id = current_size + (len(m4) * 4 / 1e6)  # int32 = 4 bytes
    
    print(f"M4 rows: {len(m4):,}")
    print(f"Current M4 memory: {current_size:.1f} MB")
    print(f"With oracle_centroid_id: {with_centroid_id:.1f} MB")
    print(f"Overhead: {with_centroid_id - current_size:.1f} MB ({(with_centroid_id/current_size - 1)*100:.0f}%)")
    
    results["m4_rows"] = len(m4)
    results["m4_current_mb"] = current_size
    results["m4_with_centroid_mb"] = with_centroid_id
    
    # Part C: JOIN Performance
    print("\n--- Part C: JOIN Performance ---")
    
    m3_path = run_dir / "tier_b/M3_observed_winners.parquet"
    m3 = pd.read_parquet(m3_path)
    
    print(f"M3 rows: {len(m3):,}")
    print(f"M4 rows: {len(m4):,}")
    
    start = time.perf_counter()
    merged = m3.merge(
        m4, 
        on=["query_id", "q_token_id", "doc_id"], 
        how="inner",
        suffixes=("_observed", "_oracle")
    )
    join_time = time.perf_counter() - start
    
    print(f"JOIN time: {join_time:.2f}s")
    print(f"Merged rows: {len(merged):,}")
    print(f"JOIN rate: {len(merged) / join_time:,.0f} rows/sec")
    
    results["m3_rows"] = len(m3)
    results["join_time_s"] = join_time
    results["merged_rows"] = len(merged)
    
    # Decision recommendation
    print("\n--- Recommendation ---")
    
    if batch_time_per * 1e6 < 1.0:  # < 1 µs per lookup
        print("✓ Batch lookup is fast (<1 µs/pos) → compute centroid on-the-fly (Option B)")
    else:
        print("⚠ Batch lookup is slow → consider pre-computing (Option A)")
    
    if join_time < 10.0:
        print("✓ JOIN is fast (<10s) → separate M3/M4 files are acceptable")
    else:
        print("⚠ JOIN is slow → consider combined table")
    
    return results


def experiment_e4_memory_scalability(run_dir: Path) -> dict:
    """
    E4: Memory scalability testing.
    
    Tests memory usage at different sample fractions to estimate
    requirements for larger datasets.
    """
    print("\n" + "=" * 60)
    print("E4: Memory Scalability")
    print("=" * 60)
    
    try:
        import psutil
    except ImportError:
        print("⚠ psutil not installed, skipping memory measurements")
        return {}
    
    results = []
    
    for sample_frac in [0.1, 0.25, 0.5, 1.0]:
        gc.collect()
        baseline_mem = psutil.Process().memory_info().rss / 1e6
        
        # Load M3 and M4 with sampling
        m3 = pd.read_parquet(run_dir / "tier_b/M3_observed_winners.parquet")
        m4 = pd.read_parquet(run_dir / "tier_b/M4_oracle_winners.parquet")
        
        if sample_frac < 1.0:
            # Sample by query to keep related rows together
            queries = m3["query_id"].unique()
            n_sample = max(1, int(len(queries) * sample_frac))
            sample_queries = np.random.choice(queries, n_sample, replace=False)
            m3 = m3[m3["query_id"].isin(sample_queries)]
            m4 = m4[m4["query_id"].isin(sample_queries)]
        
        after_load_mem = psutil.Process().memory_info().rss / 1e6
        
        # Perform JOIN
        merged = m3.merge(m4, on=["query_id", "q_token_id", "doc_id"], how="inner")
        
        after_join_mem = psutil.Process().memory_info().rss / 1e6
        
        result = {
            "sample_frac": sample_frac,
            "m3_rows": len(m3),
            "m4_rows": len(m4),
            "merged_rows": len(merged),
            "load_delta_mb": after_load_mem - baseline_mem,
            "join_delta_mb": after_join_mem - after_load_mem,
            "total_mb": after_join_mem - baseline_mem,
        }
        results.append(result)
        
        print(f"Sample {sample_frac:.0%}: M3={len(m3):,}, M4={len(m4):,}, "
              f"Merged={len(merged):,}, Memory={result['total_mb']:.0f}MB")
        
        # Clean up
        del m3, m4, merged
        gc.collect()
    
    # Extrapolate to larger scales
    if len(results) >= 2:
        print("\n--- Extrapolation ---")
        
        # Linear extrapolation based on M4 rows (largest factor)
        base = results[-1]  # 100% sample
        m4_per_mb = base["m4_rows"] / base["total_mb"]
        
        # Estimate for MS MARCO scale (10x queries, similar docs per query)
        msmarco_m4_rows = base["m4_rows"] * 50  # ~50x more queries
        msmarco_mem_est = msmarco_m4_rows / m4_per_mb
        
        print(f"Current scale: {base['m4_rows']:,} M4 rows → {base['total_mb']:.0f} MB")
        print(f"MS MARCO estimate: {msmarco_m4_rows:,} M4 rows → {msmarco_mem_est/1000:.1f} GB")
        
        if msmarco_mem_est > 32000:  # > 32 GB
            print("⚠ MS MARCO scale would exceed 32 GB → chunked processing recommended")
        else:
            print("✓ MS MARCO scale fits in memory")
    
    return results


def experiment_e5_m6_aggregation(run_dir: Path, index_path: str) -> bool:
    """
    E5: M6 aggregation validation.
    
    Computes M5 and M6 and validates the results make sense.
    """
    print("\n" + "=" * 60)
    print("E5: M6 Aggregation Validation")
    print("=" * 60)
    
    # Load data
    m4 = pd.read_parquet(run_dir / "tier_b/M4_oracle_winners.parquet")
    r0 = pd.read_parquet(run_dir / "tier_b/R0_selected_centroids.parquet")
    offsets = load_offsets_compacted(index_path)
    
    print(f"M4 loaded: {len(m4):,} rows")
    print(f"R0 loaded: {len(r0):,} rows")
    
    # Step 1: Compute oracle_centroid_id
    print("\nStep 1: Computing oracle_centroid_id...")
    start = time.perf_counter()
    
    oracle_positions = torch.tensor(m4["oracle_embedding_pos"].values, dtype=torch.long)
    oracle_centroids = torch.searchsorted(offsets, oracle_positions, side="right") - 1
    m4["oracle_centroid_id"] = oracle_centroids.numpy().astype("int32")
    
    centroid_time = time.perf_counter() - start
    print(f"  Time: {centroid_time:.2f}s")
    print(f"  Unique oracle centroids: {m4['oracle_centroid_id'].nunique():,}")
    
    # Step 2: Build selected centroid sets
    print("\nStep 2: Building selected centroid sets...")
    start = time.perf_counter()
    
    r0_sets = r0.groupby(["query_id", "q_token_id"])["centroid_id"].apply(set).to_dict()
    
    sets_time = time.perf_counter() - start
    print(f"  Time: {sets_time:.2f}s")
    print(f"  Unique (query, token) pairs: {len(r0_sets):,}")
    
    # Step 3: Compute is_miss
    print("\nStep 3: Computing is_miss...")
    start = time.perf_counter()
    
    # Vectorized approach using numpy
    keys = list(zip(m4["query_id"], m4["q_token_id"]))
    oracle_cents = m4["oracle_centroid_id"].values
    
    is_miss = []
    for (qid, tid), oc in zip(keys, oracle_cents):
        selected = r0_sets.get((qid, tid), set())
        is_miss.append(oc not in selected)
    
    m4["is_miss"] = is_miss
    
    miss_time = time.perf_counter() - start
    print(f"  Time: {miss_time:.2f}s")
    print(f"  Total misses: {sum(is_miss):,}")
    print(f"  Miss rate: {sum(is_miss) / len(is_miss):.2%}")
    
    # Step 4: M6 Global Aggregation
    print("\nStep 4: M6 Global Aggregation...")
    
    m6_global = m4.groupby("oracle_centroid_id").agg(
        miss_count=("is_miss", "sum"),
        oracle_win_count=("is_miss", "count")
    ).reset_index()
    m6_global["miss_rate"] = m6_global["miss_count"] / m6_global["oracle_win_count"]
    
    print(f"  Centroids with oracle wins: {len(m6_global):,}")
    print(f"  Centroids with any miss: {(m6_global['miss_count'] > 0).sum():,}")
    print(f"  Mean miss rate: {m6_global['miss_rate'].mean():.2%}")
    print(f"  Median miss rate: {m6_global['miss_rate'].median():.2%}")
    
    # Top problem centroids
    print("\n  Top 10 problem centroids (by miss count):")
    top10 = m6_global.nlargest(10, "miss_count")
    for _, row in top10.iterrows():
        print(f"    Centroid {row['oracle_centroid_id']}: "
              f"{row['miss_count']:,} misses / {row['oracle_win_count']:,} wins "
              f"({row['miss_rate']:.1%})")
    
    # Validation checks
    print("\n--- Validation ---")
    all_passed = True
    
    # Check 1: Miss count sums match
    total_misses_m5 = m4["is_miss"].sum()
    total_misses_m6 = m6_global["miss_count"].sum()
    if total_misses_m5 == total_misses_m6:
        print(f"✓ M5 total misses ({total_misses_m5:,}) == M6 sum ({total_misses_m6:,})")
    else:
        print(f"✗ Miss count mismatch: M5={total_misses_m5:,}, M6 sum={total_misses_m6:,}")
        all_passed = False
    
    # Check 2: Oracle win count matches M4 rows
    total_wins = m6_global["oracle_win_count"].sum()
    if total_wins == len(m4):
        print(f"✓ Total oracle wins ({total_wins:,}) == M4 rows ({len(m4):,})")
    else:
        print(f"✗ Oracle win count mismatch: M6={total_wins:,}, M4={len(m4):,}")
        all_passed = False
    
    # Check 3: Miss rate in expected range
    overall_miss_rate = total_misses_m5 / len(m4)
    if 0.01 <= overall_miss_rate <= 0.30:
        print(f"✓ Overall miss rate ({overall_miss_rate:.1%}) in expected range [1%, 30%]")
    else:
        print(f"⚠ Overall miss rate ({overall_miss_rate:.1%}) outside typical range")
    
    # Check 4: No miss_rate > 1.0
    if (m6_global["miss_rate"] > 1.0).any():
        print(f"✗ Some centroids have miss_rate > 100% (impossible)")
        all_passed = False
    else:
        print(f"✓ All miss rates in [0, 1] range")
    
    print("\n" + "-" * 40)
    if all_passed:
        print("E5 PASSED: M6 aggregation produces valid results")
    else:
        print("E5 FAILED: M6 validation errors detected")
    
    return all_passed


def run_all_experiments(run_dir: Path, index_path: str):
    """Run all verification experiments."""
    print("=" * 60)
    print("M2/M5/M6 VERIFICATION EXPERIMENTS")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Index path: {index_path}")
    
    results = {}
    
    # E1: Source data verification
    results["E1"] = experiment_e1_verify_source_data(run_dir)
    
    if not results["E1"]:
        print("\n⚠ E1 failed - cannot proceed with other experiments")
        return results
    
    # E2: M2 computation
    results["E2"] = experiment_e2_m2_computation(run_dir)
    
    # E3: Centroid lookup (needs index_path)
    if index_path:
        results["E3"] = experiment_e3_centroid_lookup(run_dir, index_path)
    else:
        print("\n⚠ Skipping E3 (no index path provided)")
    
    # E4: Memory scalability
    results["E4"] = experiment_e4_memory_scalability(run_dir)
    
    # E5: M6 aggregation (needs index_path)
    if index_path:
        results["E5"] = experiment_e5_m6_aggregation(run_dir, index_path)
    else:
        print("\n⚠ Skipping E5 (no index path provided)")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for exp, result in results.items():
        if isinstance(result, bool):
            status = "✓ PASSED" if result else "✗ FAILED"
        elif isinstance(result, dict):
            status = "✓ COMPLETED"
        else:
            status = "? UNKNOWN"
        print(f"  {exp}: {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="M2/M5/M6 Verification Experiments")
    parser.add_argument("--run-dir", type=str, help="Path to measurement run directory")
    parser.add_argument("--index-path", type=str, help="Path to WARP index")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["E1", "E2", "E3", "E4", "E5", "all"],
                       help="Which experiment to run")
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            print("Error: No run directory found. Run a measurement collection first:")
            print("  python experiments/m4_e2e_test.py --experiment E2")
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
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            index_path = metadata.get("index", {}).get("path")
            if index_path:
                print(f"Using index path from metadata: {index_path}")
    
    # Run experiments
    if args.experiment == "all":
        run_all_experiments(run_dir, index_path)
    elif args.experiment == "E1":
        experiment_e1_verify_source_data(run_dir)
    elif args.experiment == "E2":
        experiment_e2_m2_computation(run_dir)
    elif args.experiment == "E3":
        if not index_path:
            print("Error: --index-path required for E3")
            sys.exit(1)
        experiment_e3_centroid_lookup(run_dir, index_path)
    elif args.experiment == "E4":
        experiment_e4_memory_scalability(run_dir)
    elif args.experiment == "E5":
        if not index_path:
            print("Error: --index-path required for E5")
            sys.exit(1)
        experiment_e5_m6_aggregation(run_dir, index_path)


if __name__ == "__main__":
    main()
