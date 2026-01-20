#!/usr/bin/env python
"""
Thread Scaling Analysis for WARP Pipeline

This script analyzes where threads make a difference in the WARP pipeline.
Based on the M4 optimization document, we know:
- M4 oracle gets SLOWER with more threads (468ms@4 → 615ms@12)
- Search pipeline is stable (~150ms regardless of threads)

Goal: Break down each step and identify:
1. Which steps actually parallelize and benefit from threads
2. Thread synchronization overhead in each step
3. Optimal thread count per step

Pipeline Steps to Analyze:
1. Candidate Generation: Q @ centroids.T (matrix multiplication)
2. top-k Precompute: WARP centroid selection (partial_sort per token)
3. Decompression: decompress residuals per centroid (parallel_for over nprobe*tokens)
4. Build Matrix: merge scores across tokens (task graph with thread pool)
5. M4 Oracle: compute oracle per document (parallel_for over documents)
"""

import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('INDEX_ROOT', '/mnt/datasets/index')
os.environ.setdefault('DATA_ROOT', '/mnt/datasets')
os.environ.setdefault('BEIR_COLLECTION_PATH', '/mnt/datasets/BEIR')
os.environ.setdefault('LOTTE_COLLECTION_PATH', '/mnt/datasets/LOTTE')

sys.path.insert(0, '/home/azureuser/repos/RAG-D')

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries


@dataclass
class StepTiming:
    """Timing data for a single pipeline step."""
    name: str
    times_ms: List[float] = field(default_factory=list)
    
    @property
    def mean_ms(self) -> float:
        return np.mean(self.times_ms) if self.times_ms else 0.0
    
    @property
    def std_ms(self) -> float:
        return np.std(self.times_ms) if self.times_ms else 0.0
    
    @property
    def p50_ms(self) -> float:
        return np.percentile(self.times_ms, 50) if self.times_ms else 0.0
    
    @property
    def p99_ms(self) -> float:
        return np.percentile(self.times_ms, 99) if self.times_ms else 0.0


@dataclass 
class ThreadExperiment:
    """Results for a specific thread count."""
    num_threads: int
    steps: Dict[str, StepTiming] = field(default_factory=dict)
    total_times_ms: List[float] = field(default_factory=list)
    
    @property
    def total_mean_ms(self) -> float:
        return np.mean(self.total_times_ms) if self.total_times_ms else 0.0


def run_microbenchmark_candidate_generation(searcher, Q, num_iterations=50):
    """Benchmark: Q @ centroids.T (dense matrix multiplication)"""
    centroids = searcher.searcher.ranker.centroids
    times = []
    
    # Warmup
    for _ in range(5):
        _ = Q.squeeze(0) @ centroids.T
    
    for _ in range(num_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _ = Q.squeeze(0) @ centroids.T
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - start) * 1000)
    
    return times


def run_microbenchmark_topk_precompute(searcher, centroid_scores, num_iterations=50):
    """Benchmark: WARP centroid selection (parallel partial_sort)"""
    ranker = searcher.searcher.ranker
    Q_mask = torch.ones(32, dtype=torch.bool)
    Q_mask[20:] = False  # Typical ~20 tokens
    
    times = []
    
    # Warmup (must be in inference mode to avoid inplace update error)
    with torch.inference_mode():
        for _ in range(5):
            _ = ranker._warp_select_centroids(Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[1000])
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = ranker._warp_select_centroids(Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[1000])
            times.append((time.perf_counter() - start) * 1000)
    
    return times


def run_microbenchmark_decompression(searcher, Q, cells, centroid_scores, num_tokens, num_iterations=50):
    """Benchmark: Decompression of residuals (parallel_for over cells)"""
    ranker = searcher.searcher.ranker
    times = []
    
    # Warmup
    for _ in range(5):
        _ = ranker._decompress_centroids(Q.squeeze(0), cells, centroid_scores, ranker.nprobe, num_tokens)
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = ranker._decompress_centroids(Q.squeeze(0), cells, centroid_scores, ranker.nprobe, num_tokens)
        times.append((time.perf_counter() - start) * 1000)
    
    return times


def run_microbenchmark_fused(searcher, Q, cells, centroid_scores, mse_estimates, num_tokens, k, num_iterations=50):
    """Benchmark: Fused decompression + merge (task graph with thread pool)"""
    ranker = searcher.searcher.ranker
    times = []
    
    # Warmup
    for _ in range(5):
        _ = ranker._fused_decompress_merge_scores(
            Q.squeeze(0), cells, centroid_scores, ranker.nprobe, num_tokens, mse_estimates, k
        )
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = ranker._fused_decompress_merge_scores(
            Q.squeeze(0), cells, centroid_scores, ranker.nprobe, num_tokens, mse_estimates, k
        )
        times.append((time.perf_counter() - start) * 1000)
    
    return times


def run_thread_scaling_experiment(thread_counts=[1, 2, 4, 6, 8, 12, 16], 
                                  num_queries=20, 
                                  num_iterations_per_step=30,
                                  k=1000, 
                                  nprobe=32):
    """
    Run comprehensive thread scaling experiment.
    
    Tests each pipeline step independently at different thread counts.
    Note: With 1 thread, WARP uses a different non-parallel IndexScorer that
    doesn't have the same decompress API. We skip microbenchmarks for 1 thread
    but still do E2E timing.
    """
    results = {}
    
    print("=" * 80)
    print("THREAD SCALING ANALYSIS FOR WARP PIPELINE")
    print("=" * 80)
    print(f"Configuration: k={k}, nprobe={nprobe}, queries={num_queries}")
    print(f"Thread counts to test: {thread_counts}")
    print()
    
    for num_threads in thread_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_threads} thread(s)")
        print(f"{'='*60}")
        
        # Set thread count BEFORE creating searcher
        torch.set_num_threads(num_threads)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        
        # Force nranks > 1 to use parallel scorer even with few threads
        effective_nranks = max(num_threads, 2)  # Parallel scorer requires nranks > 1
        
        # Reload searcher with new thread count
        config = WARPRunConfig(
            nbits=4, 
            collection='beir', 
            dataset='quora', 
            datasplit='test',
            k=k, 
            nprobe=nprobe, 
            fused_ext=False,  # Use non-fused for separate step timing
            centroid_only=False, 
            nranks=effective_nranks,  # Force parallel mode
        )
        
        searcher = WARPSearcher(config)
        queries = WARPQueries(config)
        query_list = list(queries.queries.data.items())[:num_queries]
        
        experiment = ThreadExperiment(num_threads=num_threads)
        
        # Initialize step timings
        steps = ['Candidate Generation', 'top-k Precompute', 'Decompression', 'Fused Pipeline']
        for step in steps:
            experiment.steps[step] = StepTiming(name=step)
        
        # Get sample encoded query for microbenchmarks
        sample_query_text = query_list[0][1]
        Q = searcher.searcher.encode(sample_query_text)
        
        ranker = searcher.searcher.ranker
        
        # Verify we got the parallel ranker
        ranker_class = ranker.__class__.__name__
        print(f"  Ranker class: {ranker_class}")
        
        # Get intermediate results for step-by-step benchmarking
        with torch.inference_mode():
            centroid_scores = Q.squeeze(0) @ ranker.centroids.T
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            num_tokens = Q_mask.sum().item()
            cells, scores, mse_estimates = ranker._warp_select_centroids(
                Q_mask, centroid_scores, ranker.nprobe, ranker.t_prime[k]
            )
        
        print(f"\n  Sample query has {num_tokens} tokens")
        print(f"  Centroids shape: {ranker.centroids.shape}")
        print(f"  nprobe={nprobe}, cells shape: {cells.shape}")
        
        # === Step 1: Candidate Generation (Q @ centroids.T) ===
        print(f"\n  [1/4] Benchmarking Candidate Generation...")
        times = run_microbenchmark_candidate_generation(searcher, Q, num_iterations_per_step)
        experiment.steps['Candidate Generation'].times_ms = times
        print(f"        Mean: {np.mean(times):.3f}ms, Std: {np.std(times):.3f}ms")
        
        # === Step 2: top-k Precompute (WARP centroid selection) ===
        print(f"  [2/4] Benchmarking top-k Precompute...")
        times = run_microbenchmark_topk_precompute(searcher, centroid_scores, num_iterations_per_step)
        experiment.steps['top-k Precompute'].times_ms = times
        print(f"        Mean: {np.mean(times):.3f}ms, Std: {np.std(times):.3f}ms")
        
        # Check if parallel ranker API is available
        if hasattr(ranker, '_decompress_centroids') and ranker_class == 'ParallelIndexScorerWARP':
            # === Step 3: Decompression only ===
            print(f"  [3/4] Benchmarking Decompression...")
            times = run_microbenchmark_decompression(searcher, Q, cells, scores, num_tokens, num_iterations_per_step)
            experiment.steps['Decompression'].times_ms = times
            print(f"        Mean: {np.mean(times):.3f}ms, Std: {np.std(times):.3f}ms")
            
            # === Step 4: Fused Pipeline (decompression + merge) ===
            print(f"  [4/4] Benchmarking Fused Pipeline...")
            times = run_microbenchmark_fused(searcher, Q, cells, scores, mse_estimates, num_tokens, k, num_iterations_per_step)
            experiment.steps['Fused Pipeline'].times_ms = times
            print(f"        Mean: {np.mean(times):.3f}ms, Std: {np.std(times):.3f}ms")
        else:
            print(f"  [3/4] Skipping Decompression microbenchmark (non-parallel ranker)")
            print(f"  [4/4] Skipping Fused Pipeline microbenchmark (non-parallel ranker)")
        
        # === Full E2E timing ===
        print(f"\n  [E2E] Running full pipeline on {num_queries} queries...")
        total_times = []
        for qid, query_text in query_list:
            start = time.perf_counter()
            _ = searcher.search(query_text)
            total_times.append((time.perf_counter() - start) * 1000)
        experiment.total_times_ms = total_times
        print(f"        Mean: {np.mean(total_times):.3f}ms/query")
        
        results[num_threads] = experiment
        
        # Cleanup
        del searcher
    
    return results


def print_summary(results: Dict[int, ThreadExperiment]):
    """Print a summary table of results."""
    print("\n" + "=" * 100)
    print("SUMMARY: Thread Scaling Analysis")
    print("=" * 100)
    
    thread_counts = sorted(results.keys())
    steps = list(results[thread_counts[0]].steps.keys())
    
    # Header
    header = f"{'Threads':>8} | "
    for step in steps:
        header += f"{step[:15]:>15} | "
    header += f"{'E2E Total':>12}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for num_threads in thread_counts:
        exp = results[num_threads]
        row = f"{num_threads:>8} | "
        for step in steps:
            row += f"{exp.steps[step].mean_ms:>12.2f}ms | "
        row += f"{exp.total_mean_ms:>9.2f}ms"
        print(row)
    
    # Speedup analysis
    print("\n" + "-" * 100)
    print("SPEEDUP vs 1 thread:")
    print("-" * 100)
    
    if 1 in results:
        baseline = results[1]
        header = f"{'Threads':>8} | "
        for step in steps:
            header += f"{step[:15]:>15} | "
        header += f"{'E2E Total':>12}"
        print(header)
        print("-" * len(header))
        
        for num_threads in thread_counts:
            exp = results[num_threads]
            row = f"{num_threads:>8} | "
            for step in steps:
                baseline_time = baseline.steps[step].mean_ms
                current_time = exp.steps[step].mean_ms
                if baseline_time > 0 and current_time > 0:
                    speedup = baseline_time / current_time
                    row += f"{speedup:>12.2f}x | "
                else:
                    row += f"{'N/A':>12} | "
            
            baseline_e2e = baseline.total_mean_ms
            current_e2e = exp.total_mean_ms
            if baseline_e2e > 0 and current_e2e > 0:
                speedup = baseline_e2e / current_e2e
                row += f"{speedup:>9.2f}x"
            else:
                row += f"{'N/A':>9}"
            print(row)
    
    # Analysis insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS:")
    print("=" * 100)
    
    # Find optimal thread count for each step
    for step in steps:
        times = {t: results[t].steps[step].mean_ms for t in thread_counts}
        optimal_threads = min(times, key=times.get)
        print(f"  {step}: Optimal at {optimal_threads} threads ({times[optimal_threads]:.2f}ms)")
    
    # E2E optimal
    e2e_times = {t: results[t].total_mean_ms for t in thread_counts}
    optimal_e2e = min(e2e_times, key=e2e_times.get)
    print(f"  E2E Total: Optimal at {optimal_e2e} threads ({e2e_times[optimal_e2e]:.2f}ms)")


def save_results(results: Dict[int, ThreadExperiment], output_path: str):
    """Save results to JSON."""
    data = {}
    for num_threads, exp in results.items():
        data[num_threads] = {
            'num_threads': num_threads,
            'total_mean_ms': exp.total_mean_ms,
            'total_times_ms': exp.total_times_ms,
            'steps': {
                name: {
                    'mean_ms': step.mean_ms,
                    'std_ms': step.std_ms,
                    'p50_ms': step.p50_ms,
                    'p99_ms': step.p99_ms,
                    'times_ms': step.times_ms
                }
                for name, step in exp.steps.items()
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Thread Scaling Analysis for WARP Pipeline")
    parser.add_argument('--threads', type=str, default='1,2,4,6,8,12',
                        help='Comma-separated list of thread counts to test')
    parser.add_argument('--queries', type=int, default=20,
                        help='Number of queries for E2E timing')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Iterations per microbenchmark')
    parser.add_argument('--k', type=int, default=1000,
                        help='Top-k documents to retrieve')
    parser.add_argument('--nprobe', type=int, default=32,
                        help='Number of centroids to probe')
    parser.add_argument('--output', type=str, default='/tmp/thread_scaling_results.json',
                        help='Output path for JSON results')
    
    args = parser.parse_args()
    
    thread_counts = [int(t) for t in args.threads.split(',')]
    
    results = run_thread_scaling_experiment(
        thread_counts=thread_counts,
        num_queries=args.queries,
        num_iterations_per_step=args.iterations,
        k=args.k,
        nprobe=args.nprobe
    )
    
    print_summary(results)
    save_results(results, args.output)
