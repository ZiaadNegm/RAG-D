#!/usr/bin/env python3
"""
M3 Verification Experiments

This script runs verification experiments before implementing M3 Tier B:
- E0: Verify influential_counts works correctly
- E3: Verify embedding_pos → centroid_id mapping
- E5: Verify embeddings are sorted by doc_id within centroids
- E6: Trace single query through pipeline

Usage:
    cd /home/azureuser/repos/RAG-D
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    python experiments/m3_verification_experiments.py
"""

import os
import sys
from datetime import datetime
from collections import Counter

# WARP's search engine is CPU-only; hide GPUs to force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure threading environment BEFORE importing torch
NUM_THREADS = 1

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

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries
from warp.utils.tracker import ExecutionTracker


def experiment_e0_influential_counts(searcher, queries, config):
    """
    E0: Verify influential_counts works correctly.
    
    Expected behavior:
    - influential_counts[i] = number of query tokens with observed evidence for doc i
    - Range: 1 to num_query_tokens
    """
    print("\n" + "="*80)
    print("EXPERIMENT E0: Verify influential_counts")
    print("="*80)
    
    results = {}
    ranker = searcher.searcher.ranker
    
    for qid, qtext in list(queries)[:5]:
        print(f"\n--- Query {qid}: '{qtext[:50]}...' ---")
        
        with torch.inference_mode():
            # Encode query
            Q = searcher.searcher.encode(qtext)
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            num_query_tokens = Q_mask.sum().item()
            
            # Run through pipeline manually to get influential_counts
            centroid_scores = Q.squeeze(0) @ ranker.centroids.T
            cells, scores, mse = ranker._warp_select_centroids(
                Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[config.k]
            )
            capacities, candidate_sizes, candidate_pids, candidate_scores = ranker._decompress_centroids(
                Q.squeeze(0), cells, scores, ranker.nprobe
            )
            pids, scores_out, influential_counts, unique_docs = ranker._merge_candidate_scores(
                capacities, candidate_sizes, candidate_pids, candidate_scores, mse, config.k
            )
        
        print(f"  num_query_tokens: {num_query_tokens}")
        print(f"  influential_counts length: {len(influential_counts)}")
        print(f"  Top-10 influential_counts: {influential_counts[:10]}")
        print(f"  Min: {min(influential_counts)}, Max: {max(influential_counts)}")
        print(f"  Sum: {sum(influential_counts)}")
        
        # Validation
        valid_range = all(1 <= c <= num_query_tokens for c in influential_counts)
        print(f"  All counts in valid range [1, {num_query_tokens}]: {valid_range}")
        
        if not valid_range:
            bad_counts = [c for c in influential_counts if c < 1 or c > num_query_tokens]
            print(f"  ✗ Invalid counts found: {bad_counts[:10]}")
        
        results[qid] = {
            "num_query_tokens": num_query_tokens,
            "count_min": min(influential_counts),
            "count_max": max(influential_counts),
            "count_sum": sum(influential_counts),
            "num_docs": len(influential_counts),
            "valid_range": valid_range
        }
    
    return results


def experiment_e3_embedding_to_centroid_mapping(searcher):
    """
    E3: Verify embedding_pos → centroid_id mapping works.
    
    We use: centroid_id = searchsorted(offsets_compacted, embedding_pos, side='right') - 1
    """
    print("\n" + "="*80)
    print("EXPERIMENT E3: Verify embedding_pos → centroid_id mapping")
    print("="*80)
    
    ranker = searcher.searcher.ranker
    offsets = ranker.offsets_compacted
    sizes = ranker.sizes_compacted
    
    print(f"\nIndex statistics:")
    print(f"  Number of centroids: {len(sizes)}")
    print(f"  Total embeddings: {offsets[-1].item()}")
    print(f"  offsets_compacted shape: {offsets.shape}")
    
    # Test various embedding positions
    test_positions = [0, 100, 1000, 10000, 100000, 1000000, offsets[-1].item() - 1]
    test_positions = [p for p in test_positions if p < offsets[-1].item()]
    
    results = []
    print(f"\nTesting {len(test_positions)} embedding positions:")
    
    for pos in test_positions:
        # Method: searchsorted with side='right', then subtract 1
        centroid_id = torch.searchsorted(offsets, pos, side='right').item() - 1
        
        # Verify: offsets[centroid_id] <= pos < offsets[centroid_id + 1]
        begin = offsets[centroid_id].item()
        end = offsets[centroid_id + 1].item()
        valid = begin <= pos < end
        
        result = {
            "embedding_pos": pos,
            "centroid_id": centroid_id,
            "centroid_begin": begin,
            "centroid_end": end,
            "valid": valid
        }
        results.append(result)
        
        status = "✓" if valid else "✗"
        print(f"  {status} pos={pos:>10} → centroid={centroid_id:>6} (range [{begin}, {end}))")
    
    all_valid = all(r["valid"] for r in results)
    print(f"\nAll mappings valid: {all_valid}")
    
    return results


def experiment_e5_centroid_sorting(searcher):
    """
    E5: Verify embeddings are sorted by doc_id within centroids.
    
    The decompression code assumes embeddings within a centroid are sorted by doc_id.
    """
    print("\n" + "="*80)
    print("EXPERIMENT E5: Verify embeddings sorted by doc_id within centroids")
    print("="*80)
    
    ranker = searcher.searcher.ranker
    codes = ranker.codes_compacted
    offsets = ranker.offsets_compacted
    sizes = ranker.sizes_compacted
    
    num_centroids = len(sizes)
    print(f"\nIndex statistics:")
    print(f"  Number of centroids: {num_centroids}")
    print(f"  Total embeddings: {len(codes)}")
    
    # Test a sample of centroids
    test_centroids = [0, 100, 1000, 5000, 10000, num_centroids // 2, num_centroids - 1]
    test_centroids = [c for c in test_centroids if c < num_centroids]
    
    results = []
    print(f"\nTesting {len(test_centroids)} centroids:")
    
    for cid in test_centroids:
        begin = offsets[cid].item()
        end = offsets[cid + 1].item()
        
        if end <= begin:
            print(f"  Centroid {cid}: EMPTY")
            results.append({"centroid_id": cid, "is_sorted": True, "is_empty": True})
            continue
            
        doc_ids = codes[begin:end].tolist()
        
        # Check if sorted
        is_sorted = all(doc_ids[i] <= doc_ids[i+1] for i in range(len(doc_ids)-1))
        
        # Check for duplicates (same doc can have multiple embeddings in same centroid)
        counts = Counter(doc_ids)
        max_dups = max(counts.values())
        num_unique = len(counts)
        
        result = {
            "centroid_id": cid,
            "num_embeddings": len(doc_ids),
            "num_unique_docs": num_unique,
            "max_dups": max_dups,
            "is_sorted": is_sorted
        }
        results.append(result)
        
        status = "✓" if is_sorted else "✗"
        print(f"  {status} Centroid {cid:>6}: {len(doc_ids):>6} embeddings, "
              f"{num_unique:>6} unique docs, max_dups={max_dups}, sorted={is_sorted}")
    
    all_sorted = all(r["is_sorted"] for r in results)
    print(f"\nAll tested centroids sorted: {all_sorted}")
    
    # Additional: check a few with duplicates for proper max handling
    print("\n--- Checking duplicate handling ---")
    for r in results:
        if r.get("max_dups", 0) > 1:
            cid = r["centroid_id"]
            begin = offsets[cid].item()
            end = offsets[cid + 1].item()
            doc_ids = codes[begin:end].tolist()
            
            # Find first duplicate
            for i, (d1, d2) in enumerate(zip(doc_ids[:-1], doc_ids[1:])):
                if d1 == d2:
                    print(f"  Centroid {cid}: duplicate doc_id={d1} at positions {i},{i+1}")
                    print(f"    This is expected - same doc can have multiple embeddings in same centroid")
                    print(f"    Decompression code will keep MAX score for this doc")
                    break
    
    return results


def experiment_e6_trace_query(searcher, queries, config):
    """
    E6: Trace single query through pipeline to understand data flow.
    """
    print("\n" + "="*80)
    print("EXPERIMENT E6: Trace single query through pipeline")
    print("="*80)
    
    # Pick first query
    qid, qtext = list(queries)[0]
    print(f"\nQuery: {qid} = '{qtext}'")
    
    ranker = searcher.searcher.ranker
    
    with torch.inference_mode():
        # Encode query
        print("\n--- Step 1: Query Encoding ---")
        Q = searcher.searcher.encode(qtext)
        print(f"Q.shape: {Q.shape}")  # Expected: [1, 32, 128]
        
        # Get Q_mask
        Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
        num_query_tokens = Q_mask.sum().item()
        print(f"Q_mask.shape: {Q_mask.shape}")
        print(f"Q_mask.sum() (actual query tokens): {num_query_tokens}")
        
        # Candidate generation: Q @ centroids.T
        print("\n--- Step 2: Candidate Generation ---")
        centroid_scores = Q.squeeze(0) @ ranker.centroids.T
        print(f"centroid_scores.shape: {centroid_scores.shape}")  # [32, num_centroids]
        
        # Top centroids per token
        print("\n--- Step 3: Centroid Selection ---")
        cells, scores, mse = ranker._warp_select_centroids(
            Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[config.k]
        )
        print(f"cells.shape: {cells.shape}")  # [32 * nprobe]
        print(f"cells[:16] (first 2 tokens × {ranker.nprobe} probes): {cells[:16].tolist()}")
        print(f"scores[:16]: {scores[:16].tolist()}")
        
        # Check for dummy centroids (masked tokens)
        num_dummy = (cells == ranker.kdummy_centroid).sum().item()
        print(f"Dummy centroid (masked tokens): {ranker.kdummy_centroid}, count: {num_dummy}")
        
        # Decompression
        print("\n--- Step 4: Decompression ---")
        capacities, candidate_sizes, candidate_pids, candidate_scores = ranker._decompress_centroids(
            Q.squeeze(0), cells, scores, ranker.nprobe
        )
        print(f"capacities.shape: {capacities.shape}")
        print(f"capacities[:16] (embeddings per centroid): {capacities[:16].tolist()}")
        print(f"capacities.sum() (total embeddings decompressed): {capacities.sum().item()}")
        print(f"candidate_sizes.shape: {candidate_sizes.shape}")
        print(f"candidate_pids.shape: {candidate_pids.shape}")
        print(f"candidate_scores.shape: {candidate_scores.shape}")
        
        # Look at first stride (token 0, first centroid)
        stride_size = candidate_sizes[0].item()
        print(f"\nFirst stride (token 0, centroid 0):")
        print(f"  Size: {stride_size}")
        if stride_size > 0:
            print(f"  First 10 pids: {candidate_pids[:min(10, stride_size)].tolist()}")
            print(f"  First 10 scores: {candidate_scores[:min(10, stride_size)].tolist()}")
        
        # Merge
        print("\n--- Step 5: Merge (Build Matrix) ---")
        pids, scores_out, influential_counts, unique_docs = ranker._merge_candidate_scores(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse, config.k
        )
        print(f"unique_docs (after Phase 2): {unique_docs}")
        print(f"len(pids) (top-k returned): {len(pids)}")
        print(f"Top-10 pids: {pids[:10]}")
        print(f"Top-10 scores: {scores_out[:10]}")
        print(f"Top-10 influential_counts: {influential_counts[:10]}")
    
    # Analyze influential counts
    print("\n--- Analysis: Influential Counts ---")
    print(f"Min count: {min(influential_counts)}")
    print(f"Max count: {max(influential_counts)}")
    print(f"Mean count: {sum(influential_counts)/len(influential_counts):.2f}")
    
    # Distribution
    count_dist = Counter(influential_counts)
    print(f"Distribution of counts:")
    for c in sorted(count_dist.keys()):
        print(f"  {c} tokens: {count_dist[c]} docs")
    
    return {
        "query_id": qid,
        "query_text": qtext,
        "num_query_tokens": num_query_tokens,
        "num_centroids_probed": ranker.nprobe * num_query_tokens,
        "total_embeddings_decompressed": capacities.sum().item(),
        "unique_docs_scored": unique_docs,
        "top_k_returned": len(pids),
        "influential_count_range": (min(influential_counts), max(influential_counts))
    }


def main():
    print("="*80)
    print("M3 VERIFICATION EXPERIMENTS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    # Configure run
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=8,
        t_prime=10000,
        k=100,
        bound=128,
        centroid_only=False
    )
    
    print(f"\nConfiguration:")
    print(f"  Index: {config.index_root}/{config.index_name}")
    print(f"  nprobe: {config.nprobe}")
    print(f"  nbits: {config.nbits}")
    print(f"  k: {config.k}")
    
    # Initialize searcher
    print("\n--- Loading Searcher ---")
    searcher = WARPSearcher(config)
    
    # Load queries
    print("\n--- Loading Queries ---")
    queries = WARPQueries(config)
    queries.queries.data = dict(list(queries.queries.data.items())[:10])
    print(f"Loaded {len(queries)} queries")
    
    # Run experiments
    all_results = {}
    
    # E0: influential_counts
    all_results["E0"] = experiment_e0_influential_counts(searcher, queries, config)
    
    # E3: embedding_pos → centroid_id
    all_results["E3"] = experiment_e3_embedding_to_centroid_mapping(searcher)
    
    # E5: Centroid sorting
    all_results["E5"] = experiment_e5_centroid_sorting(searcher)
    
    # E6: Trace query
    all_results["E6"] = experiment_e6_trace_query(searcher, queries, config)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nE0 (influential_counts): ", end="")
    e0_errors = [r for r in all_results["E0"].values() if "error" in r]
    if e0_errors:
        print(f"✗ {len(e0_errors)} queries had errors")
    else:
        print("✓ All queries recorded influential_counts")
    
    print("E3 (embedding→centroid): ", end="")
    e3_valid = all(r["valid"] for r in all_results["E3"])
    print(f"{'✓' if e3_valid else '✗'} All mappings valid: {e3_valid}")
    
    print("E5 (centroid sorting): ", end="")
    e5_sorted = all(r["is_sorted"] for r in all_results["E5"])
    print(f"{'✓' if e5_sorted else '✗'} All centroids sorted: {e5_sorted}")
    
    print("E6 (query trace): ", end="")
    e6 = all_results["E6"]
    print(f"✓ Traced query with {e6['num_query_tokens']} tokens, {e6['unique_docs_scored']} docs scored")
    
    return all_results


if __name__ == "__main__":
    results = main()
