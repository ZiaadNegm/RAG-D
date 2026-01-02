#!/usr/bin/env python3
"""
Full Metrics Run Script

Executes all SQ2 metrics collection and computation:
1. Collect raw measurements (M1, M3, M4, R0)
2. Compute offline cluster properties (A1-A3, A5, B5)
3. Compute derived metrics (M2, M5, M6)
4. Compute online cluster properties (A4, A6, B1-B4, C1-C6)

Usage:
    # Test run
    python experiments/run_full_metrics.py --config test
    
    # Production run  
    python experiments/run_full_metrics.py --config production
    
    # Skip specific steps
    python experiments/run_full_metrics.py --config test --skip-raw --skip-offline
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment(num_threads: int):
    """Set up environment variables for WARP."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)


def step_1_collect_raw_measurements(config, run_id: str) -> dict:
    """
    Step 1: Collect raw measurements (M1, M3, M4, R0).
    
    Uses the WARP searcher with measurement tracking enabled.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    import torch
    torch.set_num_threads(config.raw.num_threads)
    
    from warp.engine.config import WARPRunConfig
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    from warp.utils.tracker import ExecutionTracker
    
    print("\n" + "=" * 70)
    print("STEP 1: Collect Raw Measurements (M1, M3, M4, R0)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Configure WARP
    warp_config = WARPRunConfig(
        nbits=config.raw.nbits,
        collection=config.raw.collection,
        dataset=config.raw.dataset,
        datasplit=config.raw.datasplit,
        k=config.raw.k,
        nprobe=config.raw.nprobe,
        fused_ext=config.raw.fused_ext,
        centroid_only=False,
        nranks=config.raw.num_threads,
    )
    
    print(f"\nLoading searcher (num_threads={torch.get_num_threads()})...")
    searcher = WARPSearcher(warp_config)
    print(f"Index path: {warp_config.index_root}/{warp_config.index_name}")
    print(f"Searcher type: {type(searcher.searcher.ranker).__name__}")
    
    # Load queries
    queries = WARPQueries(warp_config)
    original_count = len(queries)
    queries.queries.data = dict(list(queries.queries.data.items())[:config.raw.num_queries])
    print(f"Queries: {len(queries)} / {original_count} (subset for this run)")
    
    # Setup tracker
    output_dir = config.get_run_output_dir()
    steps = ["Query Encoding", "Candidate Generation", "top-k Precompute", "Decompression", "Build Matrix"]
    
    tracker = ExecutionTracker(
        name="WARP_Metrics",
        steps=steps,
        measurement_run_id=run_id,
        measurement_output_dir=output_dir,
        index_path=config.index_path
    )
    
    # Enable M3 and M4 tracking
    tracker.enable_m3_tracking(True)
    tracker.enable_m4_tracking(True)
    print(f"\nM3 tracking enabled: {tracker._m3_tracking_enabled}")
    print(f"M4 tracking enabled: {tracker.m4_tracking_enabled}")
    
    # Run search
    print(f"\nRunning search on {len(queries)} queries...")
    rankings = searcher.search_all(
        queries,
        k=warp_config.k,
        batched=False,
        tracker=tracker,
        show_progress=True
    )
    print("Search complete.")
    
    # Finalize measurements
    tracker.finalize_measurements()
    
    elapsed = time.time() - start_time
    run_dir = f"{output_dir}/runs/{run_id}"
    
    # Collect file stats
    file_stats = {}
    for tier in ["tier_a", "tier_b"]:
        tier_dir = Path(run_dir) / tier
        if tier_dir.exists():
            for f in tier_dir.iterdir():
                size_kb = f.stat().st_size / 1024
                file_stats[f.name] = f"{size_kb:.1f} KB"
    
    print(f"\nStep 1 complete in {elapsed:.1f}s")
    print(f"Output directory: {run_dir}")
    print(f"Files created:")
    for name, size in file_stats.items():
        print(f"  {name}: {size}")
    
    return {
        "run_dir": run_dir,
        "elapsed_seconds": elapsed,
        "num_queries": len(queries),
        "file_stats": file_stats,
    }


def step_2_compute_offline_metrics(config) -> dict:
    """
    Step 2: Compute offline cluster properties (A1-A3, A5, B5).
    
    These are computed from the index alone, independent of queries.
    """
    from warp.utils.offline_cluster_properties import (
        OfflineClusterPropertiesComputer,
        OfflineMetricsConfig as OfflineConfig,
    )
    
    print("\n" + "=" * 70)
    print("STEP 2: Compute Offline Cluster Properties (A1-A3, A5, B5)")
    print("=" * 70)
    
    start_time = time.time()
    
    offline_config = OfflineConfig(
        a5_sample_fraction=config.offline.a5_sample_fraction,
        a5_min_samples=config.offline.a5_min_samples,
        a5_max_samples=config.offline.a5_max_samples,
        a5_seed=config.offline.a5_seed,
        b5_k_neighbors=config.offline.b5_k_neighbors,
    )
    
    computer = OfflineClusterPropertiesComputer(
        index_path=config.index_path,
        config=offline_config,
        verbose=True
    )
    
    df = computer.compute_all()
    computer.print_summary(df)
    
    elapsed = time.time() - start_time
    output_path = Path(config.index_path) / "cluster_properties_offline.parquet"
    
    print(f"\nStep 2 complete in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    
    return {
        "output_path": str(output_path),
        "elapsed_seconds": elapsed,
        "num_centroids": len(df),
    }


def step_3_compute_derived_metrics(config, run_dir: str) -> dict:
    """
    Step 3: Compute derived metrics (M2, M5, M6).
    """
    from warp.utils.derived_metrics import DerivedMetricsComputer
    
    print("\n" + "=" * 70)
    print("STEP 3: Compute Derived Metrics (M2, M5, M6)")
    print("=" * 70)
    
    start_time = time.time()
    
    computer = DerivedMetricsComputer(
        run_dir=run_dir,
        index_path=config.index_path
    )
    
    results = {}
    
    # M2: Redundant computation
    print("\nComputing M2 (Redundant Computation)...")
    try:
        m2 = computer.compute_m2()
        results["M2"] = {
            "rows": len(m2),
            "mean_redundancy_rate": float(m2['redundancy_rate'].mean()),
        }
        print(f"  Mean redundancy rate: {results['M2']['mean_redundancy_rate']:.1%}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["M2"] = {"error": str(e)}
    
    # M5: Routing misses
    print("\nComputing M5 (Routing Misses)...")
    try:
        m5 = computer.compute_m5(top_k_only=config.derived.top_k_only)
        results["M5"] = {
            "rows": len(m5),
            "miss_rate": float(m5['is_miss'].mean()) if 'is_miss' in m5.columns else None,
        }
        if results["M5"]["miss_rate"] is not None:
            print(f"  Miss rate: {results['M5']['miss_rate']:.1%}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["M5"] = {"error": str(e)}
    
    # M6: Missed centroids
    print("\nComputing M6 (Missed Centroids)...")
    try:
        m6_global, m6_per_query = computer.compute_m6()
        results["M6"] = {
            "global_rows": len(m6_global),
            "per_query_rows": len(m6_per_query),
        }
        print(f"  Global: {len(m6_global)} centroids with misses")
        print(f"  Per-query: {len(m6_per_query)} rows")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["M6"] = {"error": str(e)}
    
    elapsed = time.time() - start_time
    print(f"\nStep 3 complete in {elapsed:.1f}s")
    
    return {
        "elapsed_seconds": elapsed,
        "metrics": results,
    }


def step_4_compute_online_metrics(config, run_dir: str) -> dict:
    """
    Step 4: Compute online cluster properties (A4, A6, B1-B4, C1-C6).
    """
    from warp.utils.online_cluster_properties import (
        OnlineClusterPropertiesComputer,
        OnlineMetricsConfig as OnlineConfig,
    )
    
    print("\n" + "=" * 70)
    print("STEP 4: Compute Online Cluster Properties (A4, A6, B1-B4, C1-C6)")
    print("=" * 70)
    
    start_time = time.time()
    
    online_config = OnlineConfig(
        hub_percentile=config.online.hub_percentile,
        bad_yield_threshold=config.online.bad_yield_threshold,
        good_yield_threshold=config.online.good_yield_threshold,
        num_workers=config.online.num_workers,
        recall_k_values=config.online.recall_k_values,
    )
    
    output_dir = str(Path(run_dir) / "cluster_properties_online")
    
    computer = OnlineClusterPropertiesComputer(
        run_dir=run_dir,
        index_path=config.index_path,
        output_dir=output_dir,
        config=online_config,
        verbose=True,
    )
    
    results = computer.compute_all(save=True)
    computer.print_summary()
    
    elapsed = time.time() - start_time
    
    print(f"\nStep 4 complete in {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    
    return {
        "output_dir": output_dir,
        "elapsed_seconds": elapsed,
        "summary": results.get("global_summary", {}),
    }


def run_diagnostics(run_dir: str) -> dict:
    """
    Run diagnostic checks on collected data.
    """
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS: Checking collected data")
    print("=" * 70)
    
    diagnostics = {}
    run_path = Path(run_dir)
    
    # Check M1
    m1_path = run_path / "tier_a" / "M1_compute_per_centroid.parquet"
    if m1_path.exists():
        m1 = pd.read_parquet(m1_path)
        diagnostics["M1"] = {
            "rows": len(m1),
            "unique_queries": m1['query_id'].nunique(),
            "unique_centroids": m1['centroid_id'].nunique(),
            "total_sims": int(m1['num_token_token_sims'].sum()),
        }
        print(f"\n[M1] {len(m1):,} rows")
        print(f"  Unique queries: {diagnostics['M1']['unique_queries']}")
        print(f"  Unique centroids: {diagnostics['M1']['unique_centroids']}")
        print(f"  Total sims computed: {diagnostics['M1']['total_sims']:,}")
    else:
        print(f"\n[M1] NOT FOUND: {m1_path}")
        diagnostics["M1"] = {"error": "File not found"}
    
    # Check M3
    m3_path = run_path / "tier_b" / "M3_observed_winners.parquet"
    if m3_path.exists():
        m3 = pd.read_parquet(m3_path)
        diagnostics["M3"] = {
            "rows": len(m3),
            "unique_queries": m3['query_id'].nunique(),
            "unique_docs": m3['doc_id'].nunique(),
        }
        print(f"\n[M3] {len(m3):,} rows")
        print(f"  Unique queries: {diagnostics['M3']['unique_queries']}")
        print(f"  Unique docs: {diagnostics['M3']['unique_docs']}")
    else:
        print(f"\n[M3] NOT FOUND: {m3_path}")
        diagnostics["M3"] = {"error": "File not found"}
    
    # Check M4
    m4_path = run_path / "tier_b" / "M4_oracle_winners.parquet"
    if m4_path.exists():
        m4 = pd.read_parquet(m4_path)
        diagnostics["M4"] = {
            "rows": len(m4),
            "unique_queries": m4['query_id'].nunique(),
            "unique_docs": m4['doc_id'].nunique(),
            "size_mb": m4_path.stat().st_size / (1024 * 1024),
        }
        print(f"\n[M4] {len(m4):,} rows ({diagnostics['M4']['size_mb']:.1f} MB)")
        print(f"  Unique queries: {diagnostics['M4']['unique_queries']}")
        print(f"  Unique docs: {diagnostics['M4']['unique_docs']}")
        
        # M4 scope check
        if diagnostics.get("M3"):
            m3_docs = m3['doc_id'].nunique()
            m4_docs = diagnostics['M4']['unique_docs']
            print(f"  M4/M3 doc coverage: {m4_docs}/{m3_docs} = {m4_docs/m3_docs:.1%}")
    else:
        print(f"\n[M4] NOT FOUND: {m4_path}")
        diagnostics["M4"] = {"error": "File not found"}
    
    # Check R0
    r0_path = run_path / "tier_b" / "R0_selected_centroids.parquet"
    if r0_path.exists():
        r0 = pd.read_parquet(r0_path)
        diagnostics["R0"] = {
            "rows": len(r0),
            "unique_queries": r0['query_id'].nunique(),
            "unique_centroids": r0['centroid_id'].nunique(),
            "has_centroid_score": 'centroid_score' in r0.columns,
        }
        print(f"\n[R0] {len(r0):,} rows")
        print(f"  Unique queries: {diagnostics['R0']['unique_queries']}")
        print(f"  Unique centroids: {diagnostics['R0']['unique_centroids']}")
        print(f"  Has centroid_score: {diagnostics['R0']['has_centroid_score']}")
    else:
        print(f"\n[R0] NOT FOUND: {r0_path}")
        diagnostics["R0"] = {"error": "File not found"}
    
    # Summary
    print("\n" + "-" * 70)
    print("DIAGNOSTIC SUMMARY:")
    
    issues = []
    if diagnostics.get("M1", {}).get("error"):
        issues.append("M1 missing")
    if diagnostics.get("M3", {}).get("error"):
        issues.append("M3 missing")
    if diagnostics.get("M4", {}).get("error"):
        issues.append("M4 missing - cannot compute M5/M6/C3/C4/C6")
    if diagnostics.get("R0", {}).get("error"):
        issues.append("R0 missing - cannot compute M5/C4")
    if diagnostics.get("R0", {}).get("has_centroid_score") == False:
        issues.append("R0 missing centroid_score - C4 routing fidelity will fail")
    
    if issues:
        print("⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All required files present and valid")
    
    diagnostics["issues"] = issues
    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Run full SQ2 metrics collection and computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, choices=["test", "production"], required=True,
        help="Which configuration to use"
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Skip Step 1 (raw measurements collection)"
    )
    parser.add_argument(
        "--skip-offline", action="store_true",
        help="Skip Step 2 (offline cluster properties)"
    )
    parser.add_argument(
        "--skip-derived", action="store_true",
        help="Skip Step 3 (derived metrics)"
    )
    parser.add_argument(
        "--skip-online", action="store_true",
        help="Skip Step 4 (online cluster properties)"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Use existing run directory (for skipping Step 1)"
    )
    parser.add_argument(
        "--diagnostics-only", action="store_true",
        help="Only run diagnostics on existing run"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    from experiments.configs.metrics_run_config import (
        TEST_CONFIG, PRODUCTION_CONFIG, print_config
    )
    
    if args.config == "test":
        config = TEST_CONFIG
    else:
        config = PRODUCTION_CONFIG
    
    print("\n" + "=" * 70)
    print("FULL METRICS RUN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print_config(config)
    
    # Generate run ID
    run_id = f"metrics_{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup environment
    setup_environment(config.raw.num_threads)
    
    # Track results
    results = {
        "config": args.config,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "steps": {},
    }
    
    run_dir = args.run_dir
    
    # Diagnostics only mode
    if args.diagnostics_only:
        if not run_dir:
            print("ERROR: --run-dir required with --diagnostics-only")
            sys.exit(1)
        diagnostics = run_diagnostics(run_dir)
        return
    
    # Step 1: Raw measurements
    if not args.skip_raw:
        step1_result = step_1_collect_raw_measurements(config, run_id)
        results["steps"]["raw_measurements"] = step1_result
        run_dir = step1_result["run_dir"]
    elif run_dir:
        print(f"\nSkipping Step 1, using existing run_dir: {run_dir}")
    else:
        print("ERROR: --run-dir required when --skip-raw is set")
        sys.exit(1)
    
    # Run diagnostics on raw data
    diagnostics = run_diagnostics(run_dir)
    results["diagnostics"] = diagnostics
    
    if diagnostics.get("issues"):
        print("\n⚠️ WARNING: Issues found in raw data. Continuing anyway...")
    
    # Step 2: Offline metrics
    if not args.skip_offline:
        step2_result = step_2_compute_offline_metrics(config)
        results["steps"]["offline_metrics"] = step2_result
    else:
        print("\nSkipping Step 2 (offline metrics)")
    
    # Step 3: Derived metrics
    if not args.skip_derived:
        step3_result = step_3_compute_derived_metrics(config, run_dir)
        results["steps"]["derived_metrics"] = step3_result
    else:
        print("\nSkipping Step 3 (derived metrics)")
    
    # Step 4: Online metrics
    if not args.skip_online:
        step4_result = step_4_compute_online_metrics(config, run_dir)
        results["steps"]["online_metrics"] = step4_result
    else:
        print("\nSkipping Step 4 (online metrics)")
    
    # Final summary
    results["completed_at"] = datetime.now().isoformat()
    
    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    
    total_time = sum(
        step.get("elapsed_seconds", 0) 
        for step in results["steps"].values()
    )
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save results
    results_path = Path(run_dir) / "run_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
