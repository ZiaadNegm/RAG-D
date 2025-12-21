# Parallel WARP Search Flow

This document describes the execution flow when `torch.get_num_threads() > 1`.

## Entry Point

When running with multiple threads, the searcher instantiates:
- **Python**: `ParallelIndexScorerWARP` from `warp/engine/search/parallel/parallel_index_storage.py`

## Two Execution Modes

The parallel implementation has two modes controlled by `fused_decompression_merge`:

1. **Fused Mode** (`fused_decompression_merge=True`, default): Decompression and merge are combined in a single task graph
2. **Non-Fused Mode** (`fused_decompression_merge=False`): Separate decompression and merge steps

---

## Fused Mode Flow (Default)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WARPSearcher.search_all()  (warp/engine/searcher.py)                       │
│      │                                                                      │
│      ▼                                                                      │
│  _search_all_unbatched()                                                    │
│      │  tracker.record("query_id", qid)                                     │
│      ▼                                                                      │
│  search() → encode query → ranker.rank()                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ParallelIndexScorerWARP.rank()                                             │
│  (warp/engine/search/parallel/parallel_index_storage.py)                    │
│                                                                             │
│  Step 1: Candidate Generation                                               │
│      centroid_scores = Q @ centroids.T                                      │
│      tracker.record("query_length", query_tokens)                           │
│                                                                             │
│  Step 2: top-k Precompute                                                   │
│      cells, scores, mse = _warp_select_centroids()                          │
│          └── parallel_warp_select_centroids_cpp  ────────────────┐          │
│      tracker.record("n_clusters_selected", n_clusters_selected)  │          │
│                                                                  │          │
│  Step 3 & 4: FUSED Decompression + Build Matrix                  │          │
│      pids, scores, unique_docs = _fused_decompress_merge_scores()│          │
│          └── parallel_fused_decompress_merge_cpp  ───────────────┼──┐       │
│      tracker.record("unique_docs", unique_docs)                  │  │       │
│                                                                  │  │       │
│  Return: top-k pids and scores                                   │  │       │
└──────────────────────────────────────────────────────────────────┼──┼───────┘
                                                                   │  │
                        ┌──────────────────────────────────────────┘  │
                        ▼                                             │
┌───────────────────────────────────────────────────────┐             │
│  parallel_warp_select_centroids.cpp                   │             │
│  Location: warp/engine/search/parallel/              │             │
│            parallel_warp_select_centroids.cpp         │             │
│                                                       │             │
│  Purpose: Select top centroids (parallelized)         │             │
│  Parallelism: OpenMP across query tokens              │             │
│  Output: cells, scores, mse_estimates                 │             │
└───────────────────────────────────────────────────────┘             │
                                                                      │
                        ┌─────────────────────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  parallel_fused_decompress_merge.cpp                  │
│  Location: warp/engine/search/parallel/              │
│            parallel_fused_decompress_merge.cpp        │
│                                                       │
│  Purpose: FUSED decompression + merge in task graph   │
│                                                       │
│  Task Graph Structure:                                │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Decompression Tasks (per centroid)             │  │
│  │    └── decompress_centroid_stride<nbits>()      │  │
│  │        (runs in parallel across centroids)      │  │
│  └─────────────────────────────────────────────────┘  │
│              │ dependencies                           │
│              ▼                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Token-Level Merge Tasks                        │  │
│  │    └── max-reduce nprobe lists per token        │  │
│  │        (runs as decompression completes)        │  │
│  └─────────────────────────────────────────────────┘  │
│              │ dependencies                           │
│              ▼                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Document-Level Merge Tasks                     │  │
│  │    └── sum-reduce across tokens with MSE        │  │
│  └─────────────────────────────────────────────────┘  │
│              │                                        │
│              ▼                                        │
│  partial_sort_results() → top-k                       │
│                                                       │
│  Output: pids, scores, unique_docs                    │
│          (unique_docs = views[0].size_ after merge)   │
└───────────────────────────────────────────────────────┘
```

---

## Non-Fused Mode Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ParallelIndexScorerWARP.rank()  (fused_decompression_merge=False)          │
│                                                                             │
│  Step 1: Candidate Generation                                               │
│      centroid_scores = Q @ centroids.T                                      │
│      tracker.record("query_length", query_tokens)                           │
│                                                                             │
│  Step 2: top-k Precompute                                                   │
│      cells, scores, mse = _warp_select_centroids()                          │
│          └── parallel_warp_select_centroids_cpp                             │
│      tracker.record("n_clusters_selected", n_clusters_selected)             │
│                                                                             │
│  Step 3: Decompression (SEPARATE)                                           │
│      capacities, sizes, pids, scores = _decompress_centroids()              │
│          └── parallel_decompress_centroids_cpp  ─────────────────┐          │
│      tracker.record("total_token_scores", capacities.sum())      │          │
│      unique_docs = torch.unique(candidate_pids[...]).numel()     │          │
│      tracker.record("unique_docs", unique_docs)                  │          │
│                                                                  │          │
│  Step 4: Build Matrix (SEPARATE)                                 │          │
│      pids, scores = _merge_candidate_scores()                    │          │
│          └── parallel_merge_candidate_scores_cpp  ───────────────┼──┐       │
│                                                                  │  │       │
│  Return: top-k pids and scores                                   │  │       │
└──────────────────────────────────────────────────────────────────┼──┼───────┘
                                                                   │  │
                        ┌──────────────────────────────────────────┘  │
                        ▼                                             │
┌───────────────────────────────────────────────────────┐             │
│  parallel_decompress_centroids.cpp                    │             │
│  Location: warp/engine/search/parallel/              │             │
│            parallel_decompress_centroids.cpp          │             │
│                                                       │             │
│  Purpose: Decompress embeddings (parallelized)        │             │
│  Parallelism: OpenMP across centroids                 │             │
│  Output: capacities, candidate_sizes,                 │             │
│          candidate_pids, candidate_scores             │             │
└───────────────────────────────────────────────────────┘             │
                                                                      │
                        ┌─────────────────────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  parallel_merge_candidate_scores.cpp                  │
│  Location: warp/engine/search/parallel/              │
│            parallel_merge_candidate_scores.cpp        │
│                                                       │
│  Purpose: Merge candidates using task graph           │
│  Parallelism: Task-based parallelism                  │
│  Output: pids, scores                                 │
│                                                       │
│  Note: Does NOT return unique_docs                    │
│        (computed in Python from candidate_pids)       │
└───────────────────────────────────────────────────────┘
```

---

## C++ Files Summary

| Step | C++ File (Parallel) | Python Method |
|------|---------------------|---------------|
| Centroid Selection | `parallel/parallel_warp_select_centroids.cpp` | `_warp_select_centroids()` |
| Decompression (non-fused) | `parallel/parallel_decompress_centroids.cpp` | `_decompress_centroids()` |
| Merge (non-fused) | `parallel/parallel_merge_candidate_scores.cpp` | `_merge_candidate_scores()` |
| **Fused Decompress+Merge** | `parallel/parallel_fused_decompress_merge.cpp` | `_fused_decompress_merge_scores()` |

---

## Comparison: Single-Threaded vs Parallel

| Aspect | Single-Threaded | Parallel (Fused) | Parallel (Non-Fused) |
|--------|-----------------|------------------|----------------------|
| Centroid Selection | `warp_select_centroids.cpp` | `parallel_warp_select_centroids.cpp` | Same |
| Decompression | `decompress_centroids.cpp` | Combined in fused | `parallel_decompress_centroids.cpp` |
| Merge | `merge_candidate_scores.cpp` | Combined in fused | `parallel_merge_candidate_scores.cpp` |
| `unique_docs` source | C++ returns it | C++ returns it | Python computes from `candidate_pids` |
| `total_token_scores` | Available | Not available | Available |

---

## Metrics Recorded

| Metric | Fused Mode | Non-Fused Mode | Description |
|--------|------------|----------------|-------------|
| `query_id` | ✅ | ✅ | Query identifier |
| `query_length` | ✅ | ✅ | Number of non-zero query tokens |
| `n_clusters_selected` | ✅ | ✅ | Unique centroids selected |
| `total_token_scores` | ❌ | ✅ | Sum of capacities (no intermediate tensor in fused) |
| `unique_docs` | ✅ | ✅ | Unique documents after merge |

---

## Key Differences from Single-Threaded

1. **Task Graph Parallelism**: The parallel version uses a task graph (`task_graph.hpp`) that schedules decompression and merge tasks across threads with proper dependencies.

2. **Fused Execution**: The default fused mode overlaps decompression with merging — as soon as a centroid is decompressed, its merge tasks can begin.

3. **OpenMP**: Low-level parallelism uses OpenMP (`#pragma omp parallel for`) for loop parallelization.

4. **Per-Thread Buffers**: `centroid_idx` is pre-allocated per thread to avoid contention.
