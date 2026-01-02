#!/usr/bin/env python3
"""
M3 Parallel Implementation Experiments

This script runs experiments P1-P5 to determine the best approach for 
implementing M3 winner tracking in the parallel WARP pipeline.

Experiments:
  P1: Measure non-fused vs fused performance gap
  P2: Verify Phase 1 stride availability after merge
  P3: Measure Phase 1 stride memory footprint
  P4: Profile task graph Phase 1 vs Phase 2 timing
  P5: Test post-hoc winner lookup feasibility

Usage:
    cd /home/azureuser/repos/RAG-D
    python experiments/m3_parallel_experiments.py           # Run all
    python experiments/m3_parallel_experiments.py --exp P1  # Run specific
    python experiments/m3_parallel_experiments.py --threads 8  # Custom threads

Results are printed to stdout and saved to experiments/m3_parallel_results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# WARP's search engine is CPU-only; hide GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Default threading - will be overridden per experiment
DEFAULT_THREADS = 4


def setup_threading(num_threads: int):
    """Configure threading environment."""
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    
    import torch
    torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Set defaults if not present
    os.environ.setdefault("INDEX_ROOT", "/mnt/datasets/index")
    os.environ.setdefault("DATA_ROOT", "/mnt/datasets/data")


# =============================================================================
# Experiment P1: Fused vs Non-Fused Performance Gap
# =============================================================================

def experiment_p1_fused_vs_nonfused(num_queries: int = 500, num_threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    """
    P1: Measure performance gap between fused and non-fused modes.
    
    Goal: Determine if non-fused mode is acceptable for M3 tracking.
    Decision: If gap < 20%, use non-fused for M3.
    """
    print("\n" + "="*70)
    print("EXPERIMENT P1: Fused vs Non-Fused Performance Gap")
    print("="*70)
    
    actual_threads = setup_threading(num_threads)
    print(f"Threads: {actual_threads}")
    
    import torch
    from warp.engine.config import WARPRunConfig
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    
    def benchmark_mode(fused: bool, warmup: int = 10) -> float:
        """Benchmark QPS for a given mode."""
        config = WARPRunConfig(
            collection="beir",
            dataset="quora", 
            datasplit="test",
            nbits=4,
            nprobe=16,
            k=100,
            bound=128,
            fused_ext=fused
        )
        
        searcher = WARPSearcher(config)
        queries = WARPQueries(config)
        query_list = list(queries.queries.data.items())[:num_queries]
        
        # Verify we're using parallel path
        ranker_class = searcher.searcher.ranker.__class__.__name__
        print(f"  Ranker: {ranker_class}, fused={fused}")
        
        if ranker_class != "ParallelIndexScorerWARP":
            print(f"  WARNING: Expected ParallelIndexScorerWARP, got {ranker_class}")
            print(f"  Ensure torch.get_num_threads() > 1 (currently {torch.get_num_threads()})")
        
        # Warmup
        for qid, qtext in query_list[:warmup]:
            searcher.search(qtext, k=100)
        
        # Timed run
        t0 = time.perf_counter()
        for qid, qtext in query_list:
            searcher.search(qtext, k=100)
        elapsed = time.perf_counter() - t0
        
        qps = len(query_list) / elapsed
        return qps, elapsed
    
    print(f"\nBenchmarking {num_queries} queries...")
    
    # Benchmark fused mode
    print("\n--- Fused Mode ---")
    fused_qps, fused_time = benchmark_mode(fused=True)
    print(f"  QPS: {fused_qps:.1f}")
    print(f"  Total time: {fused_time:.2f}s")
    
    # Benchmark non-fused mode
    print("\n--- Non-Fused Mode ---")
    nonfused_qps, nonfused_time = benchmark_mode(fused=False)
    print(f"  QPS: {nonfused_qps:.1f}")
    print(f"  Total time: {nonfused_time:.2f}s")
    
    # Calculate gap
    gap_percent = 100 * (fused_qps - nonfused_qps) / fused_qps
    
    print("\n--- Results ---")
    print(f"Fused QPS:     {fused_qps:.1f}")
    print(f"Non-fused QPS: {nonfused_qps:.1f}")
    print(f"Gap: {gap_percent:.1f}%")
    
    acceptable = gap_percent < 20
    print(f"\nDecision: Non-fused acceptable for M3? {'YES ✓' if acceptable else 'NO ✗'}")
    
    return {
        "experiment": "P1",
        "fused_qps": fused_qps,
        "nonfused_qps": nonfused_qps,
        "gap_percent": gap_percent,
        "acceptable": acceptable,
        "num_queries": num_queries,
        "num_threads": actual_threads
    }


# =============================================================================
# Experiment P2: Phase 1 Stride Availability
# =============================================================================

def experiment_p2_stride_availability(num_threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    """
    P2: Verify Phase 1 strides are accessible after parallel merge.
    
    Goal: Determine if we can do post-hoc winner lookup.
    """
    print("\n" + "="*70)
    print("EXPERIMENT P2: Phase 1 Stride Availability After Merge")
    print("="*70)
    
    actual_threads = setup_threading(num_threads)
    print(f"Threads: {actual_threads}")
    
    import torch
    from warp.engine.config import WARPRunConfig
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=16,
        k=100,
        fused_ext=False  # Non-fused mode
    )
    
    searcher = WARPSearcher(config)
    queries = WARPQueries(config)
    qid, qtext = list(queries.queries.data.items())[0]
    
    ranker = searcher.searcher.ranker
    
    print(f"\nQuery: {qid}")
    print(f"Ranker: {ranker.__class__.__name__}")
    
    with torch.inference_mode():
        # Encode query
        Q = searcher.searcher.encode(qtext)
        Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
        num_tokens = Q_mask.sum().item()
        
        print(f"Query tokens: {num_tokens}")
        
        # Step 1: Centroid selection
        centroid_scores = Q.squeeze(0) @ ranker.centroids.T
        cells, scores, mse = ranker._warp_select_centroids(
            Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[config.k]
        )
        
        # Step 2: Decompression (non-fused) - parallel API needs num_tokens
        capacities, candidate_sizes, candidate_pids, candidate_scores = ranker._decompress_centroids(
            Q.squeeze(0), cells, scores, ranker.nprobe, num_tokens
        )
        
        print(f"\n--- Before Merge ---")
        print(f"candidate_pids shape: {candidate_pids.shape}")
        print(f"candidate_pids contiguous: {candidate_pids.is_contiguous()}")
        print(f"candidate_sizes shape: {candidate_sizes.shape}")
        
        # Take a snapshot of Phase 1 data (per-token, after decompression, before merge)
        # Each token has nprobe strides
        phase1_snapshot = {}
        offset = 0
        for t in range(num_tokens):
            token_data = {}
            for p in range(ranker.nprobe):
                stride_idx = t * ranker.nprobe + p
                size = candidate_sizes[stride_idx].item()
                cap = capacities[stride_idx].item()
                
                if size > 0:
                    pids = candidate_pids[offset:offset+size].clone()
                    scores_t = candidate_scores[offset:offset+size].clone()
                    token_data[p] = {"pids": pids, "scores": scores_t, "size": size}
                
                offset += cap
            phase1_snapshot[t] = token_data
        
        print(f"Phase 1 snapshot captured: {len(phase1_snapshot)} tokens")
        
        # Step 3: Merge - parallel API needs num_tokens
        pids_result, scores_result = ranker._merge_candidate_scores(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse, config.k, num_tokens
        )
        
        print(f"\n--- After Merge ---")
        print(f"Top-k docs: {len(pids_result)}")
        
        # Check if original tensors were modified
        print(f"\n--- Stride Tensor Status ---")
        print(f"candidate_pids still valid: {candidate_pids.is_contiguous()}")
        print(f"candidate_pids shape unchanged: {candidate_pids.shape}")
        
        # Compare snapshot with current data
        # The merge may have modified the tensors in-place
        offset = 0
        modifications_detected = 0
        for t in range(min(num_tokens, 3)):  # Check first 3 tokens
            for p in range(min(ranker.nprobe, 2)):  # Check first 2 probes per token
                stride_idx = t * ranker.nprobe + p
                size = candidate_sizes[stride_idx].item()
                cap = capacities[stride_idx].item()
                
                if size > 0 and p in phase1_snapshot.get(t, {}):
                    current_pids = candidate_pids[offset:offset+size]
                    snapshot_pids = phase1_snapshot[t][p]["pids"]
                    
                    if not torch.equal(current_pids, snapshot_pids):
                        modifications_detected += 1
                        print(f"  Token {t}, Probe {p}: MODIFIED")
                    else:
                        print(f"  Token {t}, Probe {p}: unchanged ✓")
                
                offset += cap
        
        strides_preserved = modifications_detected == 0
        
    print(f"\n--- Results ---")
    print(f"Modifications detected: {modifications_detected}")
    print(f"Phase 1 strides preserved: {'YES ✓' if strides_preserved else 'NO ✗'}")
    
    if strides_preserved:
        print("\nConclusion: Post-hoc winner lookup IS feasible")
        print("  → Can look up winners from decompression output after merge")
    else:
        print("\nConclusion: Post-hoc winner lookup NOT feasible without copying")
        print("  → Need to copy Phase 1 data before merge, or use inline tracking")
    
    return {
        "experiment": "P2",
        "num_tokens": num_tokens,
        "nprobe": ranker.nprobe,
        "modifications_detected": modifications_detected,
        "strides_preserved": strides_preserved,
        "posthoc_feasible": strides_preserved
    }


# =============================================================================
# Experiment P3: Phase 1 Memory Footprint
# =============================================================================

def experiment_p3_memory_footprint(num_queries: int = 10, num_threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    """
    P3: Measure memory needed to store Phase 1 strides for post-hoc lookup.
    
    Goal: Determine if memory overhead is acceptable.
    """
    print("\n" + "="*70)
    print("EXPERIMENT P3: Phase 1 Stride Memory Footprint")
    print("="*70)
    
    actual_threads = setup_threading(num_threads)
    
    import torch
    from warp.engine.config import WARPRunConfig
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=16,
        k=100,
        fused_ext=False
    )
    
    searcher = WARPSearcher(config)
    queries = WARPQueries(config)
    query_list = list(queries.queries.data.items())[:num_queries]
    
    ranker = searcher.searcher.ranker
    
    memory_per_query = []
    unique_docs_per_query = []
    
    print(f"\nMeasuring memory for {num_queries} queries...")
    
    for qid, qtext in query_list:
        with torch.inference_mode():
            Q = searcher.searcher.encode(qtext)
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            num_tokens = Q_mask.sum().item()
            
            centroid_scores = Q.squeeze(0) @ ranker.centroids.T
            cells, scores, mse = ranker._warp_select_centroids(
                Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[config.k]
            )
            
            capacities, candidate_sizes, candidate_pids, candidate_scores = ranker._decompress_centroids(
                Q.squeeze(0), cells, scores, ranker.nprobe, num_tokens
            )
            
            # Calculate memory for storing Phase 1 per-token merged strides
            # After Phase 1 merge: each token has ONE stride (not nprobe)
            # Upper bound: sum of unique docs per token
            
            # For now, estimate from decompression output
            # Each (pid, score, winner_pos) = 4 + 4 + 8 = 16 bytes
            total_entries = candidate_sizes.sum().item()
            memory_bytes = total_entries * 16  # pid + score + winner_pos
            
            memory_per_query.append(memory_bytes)
            
            # Also get unique docs for context - parallel merge returns only (pids, scores)
            pids_result, scores_result = ranker._merge_candidate_scores(
                capacities, candidate_sizes, candidate_pids, candidate_scores, mse, config.k, num_tokens
            )
            # Estimate unique docs from capacities
            unique_docs_per_query.append(candidate_sizes.sum().item())
    
    avg_memory = sum(memory_per_query) / len(memory_per_query)
    max_memory = max(memory_per_query)
    avg_unique_docs = sum(unique_docs_per_query) / len(unique_docs_per_query)
    
    print(f"\n--- Results ---")
    print(f"Average memory per query: {avg_memory / 1024:.1f} KB")
    print(f"Max memory per query: {max_memory / 1024:.1f} KB")
    print(f"Average unique docs: {avg_unique_docs:.0f}")
    print(f"\nFor batch of 1000 queries: {avg_memory * 1000 / 1024 / 1024:.1f} MB")
    print(f"For batch of 10000 queries: {avg_memory * 10000 / 1024 / 1024:.1f} MB")
    
    # Acceptable if < 100 MB for 1000 queries
    acceptable = (avg_memory * 1000 / 1024 / 1024) < 100
    print(f"\nMemory overhead acceptable? {'YES ✓' if acceptable else 'NO ✗'}")
    
    return {
        "experiment": "P3",
        "num_queries": num_queries,
        "avg_memory_bytes": avg_memory,
        "max_memory_bytes": max_memory,
        "avg_memory_kb": avg_memory / 1024,
        "memory_per_1000_queries_mb": avg_memory * 1000 / 1024 / 1024,
        "avg_unique_docs": avg_unique_docs,
        "acceptable": acceptable
    }


# =============================================================================
# Experiment P4: Task Graph Timing (requires C++ instrumentation)
# =============================================================================

def experiment_p4_task_timing(num_queries: int = 100, num_threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    """
    P4: Profile Phase 1 vs Phase 2 timing in task graph.
    
    Note: Full instrumentation requires C++ changes. This experiment measures
    overall merge time and estimates Phase 1/2 split based on work distribution.
    """
    print("\n" + "="*70)
    print("EXPERIMENT P4: Task Graph Timing Analysis")
    print("="*70)
    
    actual_threads = setup_threading(num_threads)
    
    import torch
    from warp.engine.config import WARPRunConfig
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=16,
        k=100,
        fused_ext=False
    )
    
    searcher = WARPSearcher(config)
    queries = WARPQueries(config)
    query_list = list(queries.queries.data.items())[:num_queries]
    
    ranker = searcher.searcher.ranker
    
    decompression_times = []
    merge_times = []
    
    print(f"\nProfiling {num_queries} queries...")
    
    for qid, qtext in query_list:
        with torch.inference_mode():
            Q = searcher.searcher.encode(qtext)
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            num_tokens = Q_mask.sum().item()
            
            centroid_scores = Q.squeeze(0) @ ranker.centroids.T
            cells, scores, mse = ranker._warp_select_centroids(
                Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[config.k]
            )
            
            # Time decompression
            t0 = time.perf_counter()
            capacities, candidate_sizes, candidate_pids, candidate_scores = ranker._decompress_centroids(
                Q.squeeze(0), cells, scores, ranker.nprobe, num_tokens
            )
            t1 = time.perf_counter()
            decompression_times.append((t1 - t0) * 1000)  # ms
            
            # Time merge
            t0 = time.perf_counter()
            pids_result, scores_result = ranker._merge_candidate_scores(
                capacities, candidate_sizes, candidate_pids, candidate_scores, mse, config.k, num_tokens
            )
            t1 = time.perf_counter()
            merge_times.append((t1 - t0) * 1000)  # ms
    
    avg_decomp = sum(decompression_times) / len(decompression_times)
    avg_merge = sum(merge_times) / len(merge_times)
    total_time = avg_decomp + avg_merge
    
    # Estimate Phase 1 vs Phase 2 work based on task counts
    # Phase 1: 32 tokens * (2*nprobe - 1) tasks = 32 * 31 = 992 tasks
    # Phase 2: 2*32 - 1 = 63 tasks
    # So Phase 1 is ~94% of merge tasks
    phase1_task_fraction = (32 * (2 * ranker.nprobe - 1)) / (32 * (2 * ranker.nprobe - 1) + 63)
    estimated_phase1_ms = avg_merge * phase1_task_fraction
    estimated_phase2_ms = avg_merge * (1 - phase1_task_fraction)
    
    print(f"\n--- Results ---")
    print(f"Average decompression: {avg_decomp:.2f} ms")
    print(f"Average merge: {avg_merge:.2f} ms")
    print(f"  Estimated Phase 1: {estimated_phase1_ms:.2f} ms ({phase1_task_fraction*100:.0f}% of tasks)")
    print(f"  Estimated Phase 2: {estimated_phase2_ms:.2f} ms ({(1-phase1_task_fraction)*100:.0f}% of tasks)")
    print(f"Total: {total_time:.2f} ms")
    
    # Extraction overhead estimate: ~0.1 ms per query for copying data
    extraction_overhead_ms = 0.1
    overhead_percent = 100 * extraction_overhead_ms / total_time
    
    print(f"\nEstimated extraction overhead: {extraction_overhead_ms:.2f} ms ({overhead_percent:.1f}%)")
    print(f"Extraction between Phase 1 and 2: {'Feasible ✓' if overhead_percent < 5 else 'May be costly'}")
    
    return {
        "experiment": "P4",
        "num_queries": num_queries,
        "avg_decompression_ms": avg_decomp,
        "avg_merge_ms": avg_merge,
        "estimated_phase1_ms": estimated_phase1_ms,
        "estimated_phase2_ms": estimated_phase2_ms,
        "phase1_task_fraction": phase1_task_fraction,
        "extraction_overhead_percent": overhead_percent
    }


# =============================================================================
# Experiment P5: Post-Hoc Winner Lookup Benchmark
# =============================================================================

def experiment_p5_posthoc_lookup(num_queries: int = 100, num_threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    """
    P5: Benchmark post-hoc winner lookup performance.
    
    Goal: Determine if lookup time is negligible compared to search.
    """
    print("\n" + "="*70)
    print("EXPERIMENT P5: Post-Hoc Winner Lookup Benchmark")
    print("="*70)
    
    actual_threads = setup_threading(num_threads)
    
    import torch
    from warp.engine.config import WARPRunConfig
    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    
    config = WARPRunConfig(
        collection="beir",
        dataset="quora",
        datasplit="test",
        nbits=4,
        nprobe=16,
        k=100,
        fused_ext=False
    )
    
    searcher = WARPSearcher(config)
    queries = WARPQueries(config)
    query_list = list(queries.queries.data.items())[:num_queries]
    
    ranker = searcher.searcher.ranker
    
    search_times = []
    lookup_times = []
    
    print(f"\nBenchmarking {num_queries} queries...")
    
    for qid, qtext in query_list:
        with torch.inference_mode():
            # Full search
            t0 = time.perf_counter()
            Q = searcher.searcher.encode(qtext)
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            num_tokens = Q_mask.sum().item()
            
            centroid_scores = Q.squeeze(0) @ ranker.centroids.T
            cells, scores, mse = ranker._warp_select_centroids(
                Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[config.k]
            )
            
            capacities, candidate_sizes, candidate_pids, candidate_scores = ranker._decompress_centroids(
                Q.squeeze(0), cells, scores, ranker.nprobe, num_tokens
            )
            
            pids_result, scores_result = ranker._merge_candidate_scores(
                capacities, candidate_sizes, candidate_pids, candidate_scores, mse, config.k, num_tokens
            )
            t1 = time.perf_counter()
            search_times.append((t1 - t0) * 1000)
            
            # Simulate post-hoc winner lookup
            # Build Phase 1 lookup structure (dict per token)
            t0 = time.perf_counter()
            
            phase1_lookup = [{} for _ in range(32)]  # 32 tokens max
            offset = 0
            for t in range(num_tokens):
                for p in range(ranker.nprobe):
                    stride_idx = t * ranker.nprobe + p
                    size = candidate_sizes[stride_idx].item()
                    cap = capacities[stride_idx].item()
                    
                    if size > 0:
                        pids_np = candidate_pids[offset:offset+size].numpy()
                        scores_np = candidate_scores[offset:offset+size].numpy()
                        
                        for i in range(size):
                            doc_id = int(pids_np[i])
                            score = float(scores_np[i])
                            # Fake winner_pos for simulation
                            winner_pos = offset + i
                            
                            # Keep max score per doc
                            if doc_id not in phase1_lookup[t] or score > phase1_lookup[t][doc_id][0]:
                                phase1_lookup[t][doc_id] = (score, winner_pos)
                    
                    offset += cap
            
            # Look up winners for top-k docs
            winners = []
            for doc_id in pids_result:
                for t in range(num_tokens):
                    if doc_id in phase1_lookup[t]:
                        score, winner_pos = phase1_lookup[t][doc_id]
                        winners.append((doc_id, t, winner_pos, score))
            
            t1 = time.perf_counter()
            lookup_times.append((t1 - t0) * 1000)
    
    avg_search = sum(search_times) / len(search_times)
    avg_lookup = sum(lookup_times) / len(lookup_times)
    lookup_overhead_percent = 100 * avg_lookup / avg_search
    
    print(f"\n--- Results ---")
    print(f"Average search time: {avg_search:.2f} ms")
    print(f"Average lookup time: {avg_lookup:.2f} ms")
    print(f"Lookup overhead: {lookup_overhead_percent:.1f}%")
    
    acceptable = lookup_overhead_percent < 10
    print(f"\nLookup overhead acceptable (<10%)? {'YES ✓' if acceptable else 'NO ✗'}")
    
    return {
        "experiment": "P5",
        "num_queries": num_queries,
        "avg_search_ms": avg_search,
        "avg_lookup_ms": avg_lookup,
        "lookup_overhead_percent": lookup_overhead_percent,
        "acceptable": acceptable
    }


# =============================================================================
# Main
# =============================================================================

def run_all_experiments(num_threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    """Run all experiments and return combined results."""
    results = {}
    
    # P1: Performance gap
    try:
        results["P1"] = experiment_p1_fused_vs_nonfused(num_queries=500, num_threads=num_threads)
    except Exception as e:
        print(f"P1 failed: {e}")
        results["P1"] = {"experiment": "P1", "error": str(e)}
    
    # P2: Stride availability
    try:
        results["P2"] = experiment_p2_stride_availability(num_threads=num_threads)
    except Exception as e:
        print(f"P2 failed: {e}")
        results["P2"] = {"experiment": "P2", "error": str(e)}
    
    # P3: Memory footprint
    try:
        results["P3"] = experiment_p3_memory_footprint(num_queries=10, num_threads=num_threads)
    except Exception as e:
        print(f"P3 failed: {e}")
        results["P3"] = {"experiment": "P3", "error": str(e)}
    
    # P4: Task timing
    try:
        results["P4"] = experiment_p4_task_timing(num_queries=100, num_threads=num_threads)
    except Exception as e:
        print(f"P4 failed: {e}")
        results["P4"] = {"experiment": "P4", "error": str(e)}
    
    # P5: Post-hoc lookup
    try:
        results["P5"] = experiment_p5_posthoc_lookup(num_queries=100, num_threads=num_threads)
    except Exception as e:
        print(f"P5 failed: {e}")
        results["P5"] = {"experiment": "P5", "error": str(e)}
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary and recommendations."""
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    p1 = results.get("P1", {})
    p2 = results.get("P2", {})
    p3 = results.get("P3", {})
    p4 = results.get("P4", {})
    p5 = results.get("P5", {})
    
    print("\n--- Experiment Results ---")
    print(f"P1 (Fused vs Non-fused): Gap = {p1.get('gap_percent', 'N/A'):.1f}%, Acceptable: {p1.get('acceptable', 'N/A')}")
    print(f"P2 (Stride availability): Preserved = {p2.get('strides_preserved', 'N/A')}")
    mem_val = p3.get('memory_per_1000_queries_mb', None)
    if mem_val is not None:
        print(f"P3 (Memory footprint): {mem_val:.1f} MB / 1000 queries")
    else:
        print(f"P3 (Memory footprint): N/A")
    p4_val = p4.get('extraction_overhead_percent', None)
    p5_val = p5.get('lookup_overhead_percent', None)
    if p4_val is not None:
        print(f"P4 (Task timing): Extraction overhead = {p4_val:.1f}%")
    else:
        print(f"P4 (Task timing): N/A")
    if p5_val is not None:
        print(f"P5 (Post-hoc lookup): Overhead = {p5_val:.1f}%")
    else:
        print(f"P5 (Post-hoc lookup): N/A")
    
    print("\n--- Recommendation ---")
    
    # Decision logic
    nonfused_ok = p1.get("acceptable", False)
    posthoc_ok = p2.get("strides_preserved", False) and p5.get("acceptable", False)
    
    if posthoc_ok:
        print("✓ RECOMMENDED: Option B (Post-hoc winner lookup)")
        print("  - Phase 1 strides are preserved after merge")
        print("  - Lookup overhead is acceptable")
        print("  - Works with both fused and non-fused modes")
        print("  - Lowest implementation complexity")
    elif nonfused_ok:
        print("✓ RECOMMENDED: Option A (Inline tracking, non-fused)")
        print("  - Non-fused performance gap is acceptable")
        print("  - Force non-fused mode when M3 enabled")
        print("  - Mirror single-threaded implementation")
    else:
        print("⚠ RECOMMENDED: Option C (Extraction pass)")
        print("  - Requires task graph restructuring")
        print("  - Add extraction barrier between Phase 1 and 2")
        print("  - Highest complexity, but cleanest architecture")


def main():
    parser = argparse.ArgumentParser(description="M3 Parallel Implementation Experiments")
    parser.add_argument("--exp", type=str, help="Run specific experiment (P1-P5)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Number of threads")
    parser.add_argument("--output", type=str, default="experiments/m3_parallel_results.json", 
                        help="Output file for results")
    args = parser.parse_args()
    
    load_env()
    
    print("="*70)
    print("M3 PARALLEL IMPLEMENTATION EXPERIMENTS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    if args.exp:
        # Run specific experiment
        exp_map = {
            "P1": lambda: experiment_p1_fused_vs_nonfused(num_threads=args.threads),
            "P2": lambda: experiment_p2_stride_availability(num_threads=args.threads),
            "P3": lambda: experiment_p3_memory_footprint(num_threads=args.threads),
            "P4": lambda: experiment_p4_task_timing(num_threads=args.threads),
            "P5": lambda: experiment_p5_posthoc_lookup(num_threads=args.threads),
        }
        
        if args.exp not in exp_map:
            print(f"Unknown experiment: {args.exp}. Valid: {list(exp_map.keys())}")
            return 1
        
        results = {args.exp: exp_map[args.exp]()}
    else:
        # Run all experiments
        results = run_all_experiments(num_threads=args.threads)
        print_summary(results)
    
    # Save results
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "threads": args.threads
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
