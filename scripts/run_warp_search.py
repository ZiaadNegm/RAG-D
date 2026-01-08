#!/usr/bin/env python3
"""
Run WARP search on BEIR-Quora and save results for H3 Phase II analysis.

This script:
1. Loads the WARP index
2. Loads queries from BEIR-Quora test set
3. Runs search for all queries
4. Saves document rankings with scores
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from colbert import Searcher
from colbert.infra import ColBERTConfig


def load_queries(queries_path: str) -> dict:
    """Load queries from BEIR format (JSONL or TSV)."""
    queries = {}
    
    if queries_path.endswith('.jsonl'):
        with open(queries_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                queries[item['_id']] = item['text']
    elif queries_path.endswith('.tsv'):
        with open(queries_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
    else:
        raise ValueError(f"Unsupported queries format: {queries_path}")
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Run WARP search and save results")
    parser.add_argument("--index-path", required=True, help="Path to WARP index")
    parser.add_argument("--queries-path", required=True, help="Path to queries file (JSONL or TSV)")
    parser.add_argument("--output-path", required=True, help="Path to save search results parquet")
    parser.add_argument("--k", type=int, default=1000, help="Number of results per query")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for search")
    parser.add_argument("--nprobe", type=int, default=10, help="Number of centroids to probe")
    parser.add_argument("--checkpoint", help="ColBERT checkpoint path (optional)")
    parser.add_argument("--collection-map", help="Path to collection_map.json for doc ID translation")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("WARP Search for H3 Phase II Analysis")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Index: {args.index_path}")
    print(f"Queries: {args.queries_path}")
    print(f"Output: {args.output_path}")
    print(f"k={args.k}, nprobe={args.nprobe}, batch_size={args.batch_size}")
    print("=" * 70)
    
    # Load queries
    print("\nLoading queries...")
    queries = load_queries(args.queries_path)
    print(f"Loaded {len(queries):,} queries")
    
    # Load collection map if provided (for translating doc IDs back to external format)
    internal_to_external = {}
    if args.collection_map:
        print(f"Loading collection map from {args.collection_map}...")
        with open(args.collection_map, 'r') as f:
            collection_map = json.load(f)
        internal_to_external = {int(k): v for k, v in collection_map.items()}
        print(f"Loaded {len(internal_to_external):,} doc ID mappings")
    
    # Initialize searcher
    print("\nInitializing WARP searcher...")
    
    # Try to find checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        # Look for common checkpoint locations
        possible_paths = [
            "/mnt/checkpoints/colbertv2.0",
            "/mnt/datasets/checkpoints/colbertv2.0",
            os.path.join(args.index_path, "checkpoint"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                checkpoint = p
                break
    
    config = ColBERTConfig(
        index_path=args.index_path,
        nprobe=args.nprobe,
    )
    
    searcher = Searcher(
        index=args.index_path,
        checkpoint=checkpoint,
        config=config,
    )
    print("Searcher initialized")
    
    # Run search
    print(f"\nRunning search for {len(queries):,} queries...")
    start_time = time.time()
    
    results = []
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    # Process in batches
    for i in tqdm(range(0, len(query_texts), args.batch_size), desc="Searching"):
        batch_ids = query_ids[i:i+args.batch_size]
        batch_texts = query_texts[i:i+args.batch_size]
        
        # Search batch
        batch_results = searcher.search_all(
            {qid: text for qid, text in zip(batch_ids, batch_texts)},
            k=args.k,
        )
        
        # Collect results
        for qid in batch_ids:
            if qid in batch_results.data:
                ranking = batch_results.data[qid]
                for rank, (doc_id, score, passage_id) in enumerate(ranking):
                    # Convert doc_id if collection map provided
                    external_doc_id = internal_to_external.get(doc_id, str(doc_id))
                    
                    results.append({
                        'query_id': int(qid) if qid.isdigit() else qid,
                        'doc_id': doc_id,  # Internal ID for joining with golden metrics
                        'doc_id_external': external_doc_id,  # External ID for BEIR eval
                        'rank': rank + 1,
                        'score': float(score),
                    })
    
    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed:.1f}s ({len(queries)/elapsed:.1f} queries/sec)")
    
    # Save results
    print(f"\nSaving {len(results):,} results to {args.output_path}...")
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(args.output_path, index=False)
    print(f"Saved to {args.output_path}")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("Search Results Summary")
    print("=" * 70)
    print(f"Queries: {df['query_id'].nunique():,}")
    print(f"Total results: {len(df):,}")
    print(f"Results per query: {len(df) / df['query_id'].nunique():.1f}")
    print(f"Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
