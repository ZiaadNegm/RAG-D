#!/usr/bin/env python3
"""
Simple example script to run WARP search on BEIR Quora dataset.

Usage:
    conda activate warp
    python run_warp_example.py
"""

import os
import sys
# WARP's search engine is CPU-only; hide GPUs to force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure threading environment BEFORE importing torch
NUM_THREADS = 1  # Adjust based on your CPU cores (you have 16)

os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)

# Prevent thread oversubscription
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_AFFINITY"] = "disabled"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import torch
# WARP's search engine is CPU-optimized; use multiple threads for parallel execution
torch.set_num_threads(NUM_THREADS)

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries
from warp.utils.tracker import ExecutionTracker

def main(num_queries_to_run: int):
    # Configure the run
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=16,          # Number of centroids to probe
        t_prime=10000,      # Token budget for candidate generation
        k=100,              # Top-k documents to retrieve
        bound=128,          # Max tokens per document
        runtime=None,       # Use default PyTorch (or set to ONNX config)
        centroid_only=False
    )
    
    print(f"Index path: {config.index_root}/{config.index_name}")
    print(f"Collection path: {config.collection_path}")
    print(f"Queries path: {config.queries_path}")
    print()
    
    # Initialize the searcher
    print("Loading searcher...")
    searcher = WARPSearcher(config)
    
    # Load queries
    print("Loading queries...")
    queries = WARPQueries(config)
    
    # Limit to first 5 queries for testing
    queries.queries.data = dict(list(queries.queries.data.items())[:num_queries_to_run])
    
    # Setup execution tracking
    steps = ["Query Encoding", "Candidate Generation", "top-k Precompute", "Decompression", "Build Matrix"]
    tracker = ExecutionTracker(name="XTR/WARP", steps=steps)
    
    # Run search on all queries
    print(f"\nRunning search on {len(queries)} queries (limited to {num_queries_to_run} for testing)...")
    rankings = searcher.search_all(queries, k=config.k, batched=False, tracker=tracker, show_progress=True)
    
    # Evaluate results
    metrics = rankings.evaluate(queries.qrels, k=config.k)
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("\n" + "="*50)
    print("Timing Statistics:")
    print("="*50)
    print(tracker.as_dict())
    
    print("\n" + "="*50)
    print("Special Metrics (per query):")
    print("="*50)
    for i, metrics in enumerate(tracker.specialMetrics._all_iterations):
        print(f"  Query {i+1}: {metrics}")

if __name__ == "__main__":
    num_queries_to_run = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(num_queries_to_run)
