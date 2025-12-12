#!/usr/bin/env python3
"""
Simple example script to run WARP search on BEIR Quora dataset.

Usage:
    conda activate warp
    python run_warp_example.py
"""

import os
# WARP's search engine is CPU-only; hide GPUs to force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import torch
# WARP's search engine is CPU-optimized; set to 1 thread for single-threaded mode
torch.set_num_threads(1)

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries
from warp.utils.tracker import ExecutionTracker

def main():
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
    NUM_TEST_QUERIES = 1
    queries.queries.data = dict(list(queries.queries.data.items())[:NUM_TEST_QUERIES])
    
    # Setup execution tracking
    steps = ["Query Encoding", "Candidate Generation", "top-k Precompute", "Decompression", "Build Matrix"]
    tracker = ExecutionTracker(name="XTR/WARP", steps=steps)
    
    # Run search on all queries
    print(f"\nRunning search on {len(queries)} queries (limited to {NUM_TEST_QUERIES} for testing)...")
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
    main()
