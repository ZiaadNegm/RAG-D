#!/usr/bin/env python3
"""
Run WARP search on BEIR-Quora and save results for H3 Phase II analysis.

This script:
1. Runs WARP search on all 10,000 test queries
2. Saves ranking results (doc_id, score, rank) per query
3. Saves to parquet for easy analysis with golden metrics

Usage:
    nohup python scripts/run_warp_search_save_results.py > /tmp/warp_search.log 2>&1 &
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# WARP's search engine is CPU-only; hide GPUs to force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure threading environment BEFORE importing torch
NUM_THREADS = 8  # Use 8 threads for faster search

os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_AFFINITY"] = "disabled"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import torch
torch.set_num_threads(NUM_THREADS)

import pandas as pd
from tqdm import tqdm

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries
from warp.utils.tracker import ExecutionTracker


def main():
    start_time = time.time()
    
    # Output directory
    output_dir = Path('/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425/golden_metrics_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WARP Search - BEIR-Quora Full Test Set")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Configure the run - use production settings
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=32,          # Number of centroids to probe (increased for better recall)
        t_prime=10000,      # Token budget for candidate generation
        k=1000,             # Top-1000 for recall analysis
        bound=128,          # Max tokens per document
        runtime=None,
        centroid_only=False
    )
    
    print(f"Configuration:")
    print(f"  Index: {config.index_root}/{config.index_name}")
    print(f"  nprobe: {config.nprobe}")
    print(f"  t_prime: {config.t_prime}")
    print(f"  k: {config.k}")
    print(f"  NUM_THREADS: {NUM_THREADS}")
    print()
    
    # Initialize the searcher
    print("Loading searcher...")
    searcher = WARPSearcher(config)
    
    # Load queries
    print("Loading queries...")
    queries = WARPQueries(config)
    total_queries = len(queries.queries.data)
    print(f"Total queries: {total_queries:,}")
    
    # Setup execution tracking
    steps = ["Query Encoding", "Candidate Generation", "top-k Precompute", "Decompression", "Build Matrix"]
    tracker = ExecutionTracker(name="WARP-Search", steps=steps)
    
    # Run search on all queries
    print(f"\nRunning search on {total_queries:,} queries (k={config.k})...")
    print("This may take 10-30 minutes...")
    
    search_start = time.time()
    rankings = searcher.search_all(queries, k=config.k, batched=False, tracker=tracker, show_progress=True)
    search_elapsed = time.time() - search_start
    
    print(f"\nSearch completed in {search_elapsed:.1f}s ({search_elapsed/60:.1f} min)")
    print(f"Average: {search_elapsed/total_queries*1000:.1f}ms per query")
    
    # Evaluate against qrels
    print("\nEvaluating against qrels...")
    metrics = rankings.evaluate(queries.qrels, k=config.k)
    
    print("\n" + "=" * 50)
    print("Evaluation Metrics:")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Convert rankings to DataFrame and save
    print("\nConverting rankings to DataFrame...")
    
    results = []
    # WARPRanking wraps a Ranking object - access via .ranking.data
    # Each item is a tuple: (doc_id, rank, score)
    ranking_data = rankings.ranking.data
    for qid, doc_scores in tqdm(ranking_data.items(), desc="Processing rankings"):
        for pid, rank, score in doc_scores:
            results.append({
                'query_id': int(qid),
                'doc_id': int(pid),
                'rank': int(rank),
                'score': float(score)
            })
    
    results_df = pd.DataFrame(results)
    print(f"Results DataFrame: {len(results_df):,} rows")
    
    # Save to parquet
    output_path = output_dir / 'search_results.parquet'
    results_df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    # Save metrics summary
    metrics_path = output_dir / 'search_metrics.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'nprobe': config.nprobe,
            't_prime': config.t_prime,
            'k': config.k,
            'nbits': config.nbits,
            'num_threads': NUM_THREADS
        },
        'timing': {
            'total_search_seconds': search_elapsed,
            'avg_ms_per_query': search_elapsed / total_queries * 1000
        },
        'metrics': {k: float(v) for k, v in metrics.items()},
        'data_stats': {
            'total_queries': total_queries,
            'total_results': len(results_df),
            'avg_results_per_query': len(results_df) / total_queries
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Print timing stats
    print("\n" + "=" * 50)
    print("Timing Statistics:")
    print("=" * 50)
    timing_dict = tracker.as_dict()
    for step, times in timing_dict.items():
        if isinstance(times, dict) and 'mean' in times:
            print(f"  {step}: {times['mean']*1000:.2f}ms avg")
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal script time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\nDone! Output files:")
    print(f"  {output_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
