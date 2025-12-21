# Single-Threaded WARP Search Flow

This document describes the execution flow when `torch.get_num_threads() == 1`.

## Entry Point

When running with a single thread, the searcher instantiates:
- **Python**: `IndexScorerWARP` from `warp/engine/search/index_storage.py`

## Flow Diagram

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
│  IndexScorerWARP.rank()  (warp/engine/search/index_storage.py)              │
│                                                                             │
│  Step 1: Candidate Generation                                               │
│      centroid_scores = Q @ centroids.T                                      │
│      tracker.record("query_length", query_tokens)                           │
│                                                                             │
│  Step 2: top-k Precompute                                                   │
│      cells, scores, mse = _warp_select_centroids()                          │
│          └── warp_select_centroids_cpp  ─────────────────────────┐          │
│      tracker.record("n_clusters_selected", n_clusters_selected)  │          │
│                                                                  │          │
│  Step 3: Decompression                                           │          │
│      capacities, sizes, pids, scores = _decompress_centroids()   │          │
│          └── decompress_centroids_cpp  ──────────────────────────┼──┐       │
│      tracker.record("total_token_scores", total_token_scores)    │  │       │
│                                                                  │  │       │
│  Step 4: Build Matrix (Merge)                                    │  │       │
│      pids, scores, unique_docs = _merge_candidate_scores()       │  │       │
│          └── merge_candidate_scores_cpp  ────────────────────────┼──┼──┐    │
│      tracker.record("unique_docs", unique_docs)                  │  │  │    │
│                                                                  │  │  │    │
│  Return: top-k pids and scores                                   │  │  │    │
└──────────────────────────────────────────────────────────────────┼──┼──┼────┘
                                                                   │  │  │
                        ┌──────────────────────────────────────────┘  │  │
                        ▼                                             │  │
┌───────────────────────────────────────────────────────┐             │  │
│  warp_select_centroids.cpp                            │             │  │
│  Location: warp/engine/search/warp_select_centroids.cpp             │  │
│                                                       │             │  │
│  Purpose: Select top centroids per query token        │             │  │
│  Input: Q_mask, centroid_scores, sizes, nprobe, t_prime, bound      │  │
│  Output: cells (centroid IDs), scores, mse_estimates  │             │  │
└───────────────────────────────────────────────────────┘             │  │
                                                                      │  │
                        ┌─────────────────────────────────────────────┘  │
                        ▼                                                │
┌───────────────────────────────────────────────────────┐                │
│  decompress_centroids.cpp                             │                │
│  Location: warp/engine/search/decompress_centroids.cpp│                │
│                                                       │                │
│  Purpose: Decompress embeddings from selected centroids                │
│  Input: begins, ends, capacities, centroid_scores,    │                │
│         codes_compacted, residuals_compacted,         │                │
│         bucket_weights, Q, nprobe                     │                │
│  Output: capacities, candidate_sizes,                 │                │
│          candidate_pids, candidate_scores             │                │
│                                                       │                │
│  Note: Returns per-centroid document lists with scores│                │
│        (duplicates across centroids possible)         │                │
└───────────────────────────────────────────────────────┘                │
                                                                         │
                        ┌────────────────────────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────┐
│  merge_candidate_scores.cpp                           │
│  Location: warp/engine/search/merge_candidate_scores.cpp
│                                                       │
│  Purpose: Merge candidate lists and compute final scores
│                                                       │
│  Algorithm:                                           │
│  1. For each token: merge nprobe centroid lists       │
│     (max-reduce duplicate docs within token)          │
│  2. Across tokens: merge token-level scores           │
│     (sum-reduce with MSE estimates)                   │
│  3. Partial sort to get top-k                         │
│                                                       │
│  Input: capacities, candidate_sizes,                  │
│         candidate_pids, candidate_scores,             │
│         mse_estimates, nprobe, k                      │
│                                                       │
│  Output: top-k pids, scores, unique_docs              │
│          (unique_docs = views[0].size_ after merge)   │
└───────────────────────────────────────────────────────┘
```

## C++ Files Summary

| Step | C++ File | Python Method |
|------|----------|---------------|
| Centroid Selection | `warp/engine/search/warp_select_centroids.cpp` | `_warp_select_centroids()` |
| Decompression | `warp/engine/search/decompress_centroids.cpp` | `_decompress_centroids()` |
| Merge & Score | `warp/engine/search/merge_candidate_scores.cpp` | `_merge_candidate_scores()` |

## Metrics Recorded

| Metric | Location | Description |
|--------|----------|-------------|
| `query_id` | searcher.py | Query identifier |
| `query_length` | index_storage.py (rank) | Number of non-zero query tokens |
| `n_clusters_selected` | index_storage.py (rank) | Unique centroids selected |
| `total_token_scores` | index_storage.py (rank) | Sum of capacities (decompression work) |
| `unique_docs` | merge_candidate_scores.cpp | Unique documents after merge, before top-k |
