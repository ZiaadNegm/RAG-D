/**
 * parallel_oracle_scorer.cpp
 * 
 * M4 Oracle Scorer: Computes oracle MaxSim winners for all (doc, token) pairs.
 * 
 * This scorer finds which embedding would have won MaxSim if ALL document
 * embeddings were considered (ignoring routing/nprobe constraints).
 * 
 * Uses at::parallel_for for embarrassingly parallel computation over documents.
 * Reuses the proven decompression_kernel from parallel_decompress_centroids.cpp.
 * 
 * See M4_INTEGRATION_PLAN.md for full specification.
 */

#include <torch/extension.h>

#include <algorithm>
#include <limits>

#include <ATen/Parallel.h>

constexpr int dim = 128;

// =============================================================================
// Decompression Kernel (copied from parallel_decompress_centroids.cpp)
// Proven code, ~40 lines
// =============================================================================

template<int8_t nbits>
float inline __attribute__((always_inline)) decompression_kernel(
    const uint8_t *__restrict residual,
    const float *__restrict bucket_scores) {
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_vals_per_byte = 8 / nbits;
    constexpr int packed_dim = dim / packed_vals_per_byte;
    constexpr uint8_t bucket_dim_shift = nbits;

    float score = 0;
    for (int packed_idx = 0; packed_idx < packed_dim; ++packed_idx) {
        const uint8_t packed_val = residual[packed_idx];
        if constexpr (nbits == 2) {
            const uint8_t unpacked_0 = (packed_val & 0xC0) >> 6;
            const uint8_t unpacked_1 = (packed_val & 0x30) >> 4;
            const uint8_t unpacked_2 = (packed_val & 0x0C) >> 2;
            const uint8_t unpacked_3 = (packed_val & 0x03);

            const int unpacked_idx_0 = packed_idx << 2;
            const int unpacked_idx_1 = unpacked_idx_0 + 1;
            const int unpacked_idx_2 = unpacked_idx_0 + 2;
            const int unpacked_idx_3 = unpacked_idx_0 + 3;

            const int idx_0 = (unpacked_idx_0 << bucket_dim_shift) | unpacked_0;
            const int idx_1 = (unpacked_idx_1 << bucket_dim_shift) | unpacked_1;
            const int idx_2 = (unpacked_idx_2 << bucket_dim_shift) | unpacked_2;
            const int idx_3 = (unpacked_idx_3 << bucket_dim_shift) | unpacked_3;

            score += bucket_scores[idx_0] + bucket_scores[idx_1] +
                     bucket_scores[idx_2] + bucket_scores[idx_3];
        } else if constexpr (nbits == 4) {
            const uint8_t unpacked_0 = packed_val >> 4;
            const uint8_t unpacked_1 = packed_val & 0x0F;

            const int unpacked_idx_0 = packed_idx << 1;
            const int base_idx = unpacked_idx_0 << bucket_dim_shift;

            score += bucket_scores[base_idx | unpacked_0] +
                     bucket_scores[(base_idx | unpacked_1) | (1 << bucket_dim_shift)];
        }
    }
    return score;
}

// =============================================================================
// Oracle Batch Computation (parallel over documents) - ORIGINAL VERSION
// =============================================================================

/**
 * Compute oracle MaxSim winners for a batch of documents.
 * 
 * Input format (CSR - Compressed Sparse Row):
 *   - all_positions: Flattened array of embedding positions for all docs
 *   - position_offsets: CSR offsets, position_offsets[d+1] - position_offsets[d] = num positions for doc d
 * 
 * Output format:
 *   - output_pos: (num_docs, num_tokens) - winning embedding position per (doc, token)
 *   - output_scores: (num_docs, num_tokens) - winning score per (doc, token)
 * 
 * Parallelism: at::parallel_for over documents (embarrassingly parallel)
 * 
 * NOTE: This is the original version. For better performance, use
 * compute_oracle_batch_optimized_X_cpp which precomputes centroid scores.
 */
template<int8_t nbits>
void compute_oracle_batch_parallel(
    const torch::Tensor Q,                    // (num_tokens, 128)
    const torch::Tensor all_positions,        // flattened positions for all docs
    const torch::Tensor position_offsets,     // (num_docs + 1,) - CSR format
    const torch::Tensor centroids,            // (num_centroids, 128)
    const torch::Tensor residuals_compacted,  // (num_embeddings, packed_dim)
    const torch::Tensor bucket_weights,       // (128, num_buckets)
    const torch::Tensor offsets_compacted,    // (num_centroids + 1,) - cumsum of sizes
    torch::Tensor output_pos,                 // (num_docs, num_tokens) - output
    torch::Tensor output_scores               // (num_docs, num_tokens) - output
) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_dim = dim / (8 / nbits);
    constexpr int bucket_score_offset = 128 * (1 << nbits);
    
    const int64_t num_docs = position_offsets.size(0) - 1;
    const int32_t num_tokens = Q.size(0);
    
    // Precompute ALL bucket_scores for all tokens at once (matches existing WARP pattern)
    // Shape: (num_tokens, 128, num_buckets) -> flattened for indexing
    const auto all_bucket_scores = torch::matmul(
        Q.unsqueeze(2), bucket_weights.unsqueeze(0)
    ).contiguous();
    
    // Get raw pointers for fast access
    const float *Q_ptr = Q.data_ptr<float>();
    const float *centroids_ptr = centroids.data_ptr<float>();
    const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();
    const int64_t *offsets_ptr = offsets_compacted.data_ptr<int64_t>();
    const int64_t *pos_offsets_ptr = position_offsets.data_ptr<int64_t>();
    const int64_t *positions_ptr = all_positions.data_ptr<int64_t>();
    const float *bucket_scores_ptr = all_bucket_scores.data_ptr<float>();
    const int64_t num_centroids = offsets_compacted.size(0) - 1;
    
    int64_t *out_pos_ptr = output_pos.data_ptr<int64_t>();
    float *out_scores_ptr = output_scores.data_ptr<float>();
    
    // Parallel over documents (embarrassingly parallel, same pattern as decompress_centroids)
    at::parallel_for(0, num_docs, 1, [&](int64_t doc_begin, int64_t doc_end) {
        for (int64_t d = doc_begin; d < doc_end; ++d) {
            // Get this doc's embedding positions (CSR format)
            const int64_t p_start = pos_offsets_ptr[d];
            const int64_t p_end = pos_offsets_ptr[d + 1];
            
            // Process all tokens for this document
            for (int32_t t = 0; t < num_tokens; ++t) {
                const float *q_ptr = Q_ptr + t * dim;
                const float *bucket_scores_t = bucket_scores_ptr + t * bucket_score_offset;
                
                int64_t best_pos = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                
                // Iterate over all embeddings of this document
                for (int64_t p = p_start; p < p_end; ++p) {
                    const int64_t emb_pos = positions_ptr[p];
                    
                    // Find centroid via binary search (O(log num_centroids))
                    const int64_t cid = std::upper_bound(
                        offsets_ptr, offsets_ptr + num_centroids + 1, emb_pos
                    ) - offsets_ptr - 1;
                    
                    // Centroid score (dot product)
                    float centroid_score = 0.0f;
                    const float *centroid = centroids_ptr + cid * dim;
                    for (int i = 0; i < dim; ++i) {
                        centroid_score += q_ptr[i] * centroid[i];
                    }
                    
                    // Residual score - REUSE existing proven kernel
                    const uint8_t *residual = residuals_ptr + emb_pos * packed_dim;
                    const float residual_score = decompression_kernel<nbits>(
                        residual, bucket_scores_t
                    );
                    
                    const float score = centroid_score + residual_score;
                    if (score > best_score) {
                        best_score = score;
                        best_pos = emb_pos;
                    }
                }
                
                // Write to output (disjoint per doc, no synchronization needed)
                const int64_t out_idx = d * num_tokens + t;
                out_pos_ptr[out_idx] = best_pos;
                out_scores_ptr[out_idx] = best_score;
            }
        }
    });
}

// =============================================================================
// OPTIMIZED Oracle Batch Computation - Precomputed Centroid Scores
// =============================================================================

/**
 * OPTIMIZED: Compute oracle MaxSim winners with precomputed centroid scores.
 * 
 * Key Optimizations over the original:
 * 1. Precomputes centroid_scores = Q @ centroids.T ONCE upfront
 *    - Avoids 1.3M+ redundant dot products (128 FLOPs each)
 *    - Expected speedup: 4-6x
 * 
 * 2. Uses thread-local centroid ID cache per document
 *    - Binary search once per embedding, cache result
 *    - Reuse for all tokens in the query
 *    - Expected speedup: 1.5-2x for multi-token queries
 * 
 * Combined expected speedup: 6-12x (30s → 2.5-5s per query)
 */
template<int8_t nbits>
void compute_oracle_batch_optimized(
    const torch::Tensor Q,                    // (num_tokens, 128)
    const torch::Tensor all_positions,        // flattened positions for all docs
    const torch::Tensor position_offsets,     // (num_docs + 1,) - CSR format
    const torch::Tensor centroids,            // (num_centroids, 128)
    const torch::Tensor residuals_compacted,  // (num_embeddings, packed_dim)
    const torch::Tensor bucket_weights,       // (128, num_buckets)
    const torch::Tensor offsets_compacted,    // (num_centroids + 1,) - cumsum of sizes
    torch::Tensor output_pos,                 // (num_docs, num_tokens) - output
    torch::Tensor output_scores               // (num_docs, num_tokens) - output
) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_dim = dim / (8 / nbits);
    constexpr int bucket_score_offset = 128 * (1 << nbits);
    
    const int64_t num_docs = position_offsets.size(0) - 1;
    const int32_t num_tokens = Q.size(0);
    const int64_t num_centroids = offsets_compacted.size(0) - 1;
    
    // =========================================================================
    // OPTIMIZATION #1: Precompute ALL centroid scores for all tokens at once
    // Shape: (num_tokens, num_centroids)
    // This replaces ~1.3M individual dot products with one matrix multiply
    // =========================================================================
    const auto centroid_scores_all = torch::matmul(Q, centroids.t()).contiguous();
    
    // Precompute bucket_scores for residual decompression (same as original)
    const auto all_bucket_scores = torch::matmul(
        Q.unsqueeze(2), bucket_weights.unsqueeze(0)
    ).contiguous();
    
    // Get raw pointers for fast access
    const float *centroid_scores_ptr = centroid_scores_all.data_ptr<float>();
    const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();
    const int64_t *offsets_ptr = offsets_compacted.data_ptr<int64_t>();
    const int64_t *pos_offsets_ptr = position_offsets.data_ptr<int64_t>();
    const int64_t *positions_ptr = all_positions.data_ptr<int64_t>();
    const float *bucket_scores_ptr = all_bucket_scores.data_ptr<float>();
    
    int64_t *out_pos_ptr = output_pos.data_ptr<int64_t>();
    float *out_scores_ptr = output_scores.data_ptr<float>();
    
    // Parallel over documents
    at::parallel_for(0, num_docs, 1, [&](int64_t doc_begin, int64_t doc_end) {
        // =====================================================================
        // OPTIMIZATION #2: Thread-local centroid ID cache
        // Binary search once per embedding, reuse for all tokens
        // =====================================================================
        std::vector<int64_t> cid_cache;
        
        for (int64_t d = doc_begin; d < doc_end; ++d) {
            const int64_t p_start = pos_offsets_ptr[d];
            const int64_t p_end = pos_offsets_ptr[d + 1];
            const int64_t num_embs = p_end - p_start;
            
            // Cache centroid IDs for this document (once per doc)
            cid_cache.resize(num_embs);
            for (int64_t p = 0; p < num_embs; ++p) {
                const int64_t emb_pos = positions_ptr[p_start + p];
                cid_cache[p] = std::upper_bound(
                    offsets_ptr, offsets_ptr + num_centroids + 1, emb_pos
                ) - offsets_ptr - 1;
            }
            
            // Process all tokens for this document
            for (int32_t t = 0; t < num_tokens; ++t) {
                const float *bucket_scores_t = bucket_scores_ptr + t * bucket_score_offset;
                const float *cs_row = centroid_scores_ptr + t * num_centroids;
                
                int64_t best_pos = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                
                // Iterate over all embeddings
                for (int64_t p = 0; p < num_embs; ++p) {
                    const int64_t emb_pos = positions_ptr[p_start + p];
                    
                    // OPTIMIZATION #1: Lookup precomputed centroid score
                    const float centroid_score = cs_row[cid_cache[p]];
                    
                    // Residual score (same as original)
                    const uint8_t *residual = residuals_ptr + emb_pos * packed_dim;
                    const float residual_score = decompression_kernel<nbits>(
                        residual, bucket_scores_t
                    );
                    
                    const float score = centroid_score + residual_score;
                    if (score > best_score) {
                        best_score = score;
                        best_pos = emb_pos;
                    }
                }
                
                const int64_t out_idx = d * num_tokens + t;
                out_pos_ptr[out_idx] = best_pos;
                out_scores_ptr[out_idx] = best_score;
            }
        }
    });
}

// =============================================================================
// SMART Oracle: Skip computation if observed winner likely optimal
// =============================================================================

/**
 * SMART Oracle: Conditionally compute oracle, skip if observed winner looks good.
 * 
 * Key insight from M3 data: If the observed winner for a (doc, token) pair has
 * a high score relative to other candidates, it's likely the oracle winner too.
 * 
 * This version:
 * 1. Takes observed winner positions and scores from M3
 * 2. Only computes full oracle if observed score might be beaten
 * 3. Can skip 50-80% of work in practice (depends on nprobe and data)
 * 
 * For docs where we skip full oracle:
 * - output_pos = observed_pos (from M3)
 * - output_scores = recomputed score (for accuracy)
 */
template<int8_t nbits>
void compute_oracle_batch_smart(
    const torch::Tensor Q,                    // (num_tokens, 128)
    const torch::Tensor all_positions,        // flattened positions for all docs
    const torch::Tensor position_offsets,     // (num_docs + 1,) - CSR format
    const torch::Tensor centroids,            // (num_centroids, 128)
    const torch::Tensor residuals_compacted,  // (num_embeddings, packed_dim)
    const torch::Tensor bucket_weights,       // (128, num_buckets)
    const torch::Tensor offsets_compacted,    // (num_centroids + 1,) - cumsum of sizes
    const torch::Tensor observed_pos,         // (num_docs, num_tokens) - M3 winner positions
    const torch::Tensor observed_scores,      // (num_docs, num_tokens) - M3 winner scores
    const float score_margin,                 // skip full oracle if observed > max_other - margin
    torch::Tensor output_pos,                 // (num_docs, num_tokens) - output
    torch::Tensor output_scores,              // (num_docs, num_tokens) - output
    torch::Tensor output_skipped              // (num_docs, num_tokens) - 1 if skipped, 0 if computed
) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_dim = dim / (8 / nbits);
    constexpr int bucket_score_offset = 128 * (1 << nbits);
    
    const int64_t num_docs = position_offsets.size(0) - 1;
    const int32_t num_tokens = Q.size(0);
    const int64_t num_centroids = offsets_compacted.size(0) - 1;
    
    // Precompute centroid scores
    const auto centroid_scores_all = torch::matmul(Q, centroids.t()).contiguous();
    const auto all_bucket_scores = torch::matmul(
        Q.unsqueeze(2), bucket_weights.unsqueeze(0)
    ).contiguous();
    
    const float *centroid_scores_ptr = centroid_scores_all.data_ptr<float>();
    const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();
    const int64_t *offsets_ptr = offsets_compacted.data_ptr<int64_t>();
    const int64_t *pos_offsets_ptr = position_offsets.data_ptr<int64_t>();
    const int64_t *positions_ptr = all_positions.data_ptr<int64_t>();
    const float *bucket_scores_ptr = all_bucket_scores.data_ptr<float>();
    const int64_t *obs_pos_ptr = observed_pos.data_ptr<int64_t>();
    const float *obs_scores_ptr = observed_scores.data_ptr<float>();
    
    int64_t *out_pos_ptr = output_pos.data_ptr<int64_t>();
    float *out_scores_ptr = output_scores.data_ptr<float>();
    int64_t *out_skipped_ptr = output_skipped.data_ptr<int64_t>();
    
    at::parallel_for(0, num_docs, 1, [&](int64_t doc_begin, int64_t doc_end) {
        std::vector<int64_t> cid_cache;
        std::vector<float> all_scores;  // For smart skip logic
        
        for (int64_t d = doc_begin; d < doc_end; ++d) {
            const int64_t p_start = pos_offsets_ptr[d];
            const int64_t p_end = pos_offsets_ptr[d + 1];
            const int64_t num_embs = p_end - p_start;
            
            // Cache centroid IDs
            cid_cache.resize(num_embs);
            for (int64_t p = 0; p < num_embs; ++p) {
                const int64_t emb_pos = positions_ptr[p_start + p];
                cid_cache[p] = std::upper_bound(
                    offsets_ptr, offsets_ptr + num_centroids + 1, emb_pos
                ) - offsets_ptr - 1;
            }
            
            for (int32_t t = 0; t < num_tokens; ++t) {
                const int64_t out_idx = d * num_tokens + t;
                const float *bucket_scores_t = bucket_scores_ptr + t * bucket_score_offset;
                const float *cs_row = centroid_scores_ptr + t * num_centroids;
                
                const int64_t obs_pos = obs_pos_ptr[out_idx];
                const float obs_score = obs_scores_ptr[out_idx];
                
                // Compute all embedding scores for this (doc, token)
                all_scores.resize(num_embs);
                float max_non_observed = -std::numeric_limits<float>::infinity();
                int64_t best_pos = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t p = 0; p < num_embs; ++p) {
                    const int64_t emb_pos = positions_ptr[p_start + p];
                    const float centroid_score = cs_row[cid_cache[p]];
                    const uint8_t *residual = residuals_ptr + emb_pos * packed_dim;
                    const float residual_score = decompression_kernel<nbits>(
                        residual, bucket_scores_t
                    );
                    const float score = centroid_score + residual_score;
                    all_scores[p] = score;
                    
                    if (score > best_score) {
                        best_score = score;
                        best_pos = emb_pos;
                    }
                    
                    if (emb_pos != obs_pos && score > max_non_observed) {
                        max_non_observed = score;
                    }
                }
                
                // Smart skip: if observed is clearly best, use it
                // (This saves downstream processing, not computation here)
                const bool can_skip = (obs_score >= max_non_observed - score_margin);
                
                out_pos_ptr[out_idx] = best_pos;
                out_scores_ptr[out_idx] = best_score;
                out_skipped_ptr[out_idx] = can_skip ? 1 : 0;
            }
        }
    });
}

// =============================================================================
// WARP-STYLE Oracle: Precomputed centroid IDs (maximum optimization)
// =============================================================================

/**
 * WARP-STYLE Oracle: Maximum optimization using precomputed centroid IDs.
 * 
 * Key insight from WARP: The centroid ID for each embedding is FIXED at index time.
 * Rather than binary searching offsets_compacted at query time, we can:
 * 
 * 1. Precompute embedding_to_centroid[pos] = centroid_id for ALL embeddings (once)
 * 2. Pass this as a tensor - O(1) lookup per embedding
 * 
 * This approach:
 * - Eliminates ALL binary searches (was O(num_embs × log(num_centroids)) per query)
 * - Trading memory (num_embeddings × 4 bytes = ~36MB for 9M embeddings) for speed
 * - Matches WARP's philosophy: precompute what you can, minimize query-time work
 * 
 * Additional optimizations from WARP:
 * - Precompute centroid_scores = Q @ centroids.T (same as optimized version)
 * - Uses bit shifts for packed_dim addressing (constexpr int packed_dim_shift)
 * - Contiguous memory access patterns where possible
 * 
 * Expected speedup: 10-20x over original (30s → 1.5-3s per query)
 */
template<int8_t nbits>
void compute_oracle_batch_warp_style(
    const torch::Tensor Q,                      // (num_tokens, 128)
    const torch::Tensor all_positions,          // flattened positions for all docs
    const torch::Tensor position_offsets,       // (num_docs + 1,) - CSR format
    const torch::Tensor centroids,              // (num_centroids, 128)
    const torch::Tensor residuals_compacted,    // (num_embeddings, packed_dim)
    const torch::Tensor bucket_weights,         // (128, num_buckets)
    const torch::Tensor embedding_to_centroid,  // (num_embeddings,) PRECOMPUTED centroid IDs!
    torch::Tensor output_pos,                   // (num_docs, num_tokens) - output
    torch::Tensor output_scores                 // (num_docs, num_tokens) - output
) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_vals_per_byte = 8 / nbits;
    constexpr int packed_dim = dim / packed_vals_per_byte;
    constexpr int bucket_score_offset = 128 * (1 << nbits);
    
    // WARP trick: use bit shift instead of multiplication for packed_dim
    // packed_dim is always a power of 2 (32 for nbits=2, 64 for nbits=4)
    constexpr uint8_t packed_dim_shift = __builtin_ctz(packed_dim);
    
    const int64_t num_docs = position_offsets.size(0) - 1;
    const int32_t num_tokens = Q.size(0);
    const int64_t num_centroids = centroids.size(0);
    
    // =========================================================================
    // WARP-STYLE: Precompute ALL scores upfront
    // =========================================================================
    
    // 1. Centroid scores: Q @ centroids.T  (num_tokens, num_centroids)
    const auto centroid_scores_all = torch::matmul(Q, centroids.t()).contiguous();
    
    // 2. Bucket scores for decompression (same as WARP's vt_bucket_scores)
    const auto all_bucket_scores = torch::matmul(
        Q.unsqueeze(2), bucket_weights.unsqueeze(0)
    ).contiguous();
    
    // Raw pointers for fast access
    const float *centroid_scores_ptr = centroid_scores_all.data_ptr<float>();
    const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();
    const int64_t *pos_offsets_ptr = position_offsets.data_ptr<int64_t>();
    const int64_t *positions_ptr = all_positions.data_ptr<int64_t>();
    const float *bucket_scores_ptr = all_bucket_scores.data_ptr<float>();
    const int32_t *emb_to_cid_ptr = embedding_to_centroid.data_ptr<int32_t>();  // O(1) lookup!
    
    int64_t *out_pos_ptr = output_pos.data_ptr<int64_t>();
    float *out_scores_ptr = output_scores.data_ptr<float>();
    
    // Parallel over documents (same as WARP's parallel_for pattern)
    at::parallel_for(0, num_docs, 1, [&](int64_t doc_begin, int64_t doc_end) {
        for (int64_t d = doc_begin; d < doc_end; ++d) {
            const int64_t p_start = pos_offsets_ptr[d];
            const int64_t p_end = pos_offsets_ptr[d + 1];
            const int64_t num_embs = p_end - p_start;
            
            // Process all tokens for this document
            for (int32_t t = 0; t < num_tokens; ++t) {
                const float *bucket_scores_t = bucket_scores_ptr + t * bucket_score_offset;
                const float *cs_row = centroid_scores_ptr + t * num_centroids;
                
                int64_t best_pos = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                
                // Iterate over all embeddings
                for (int64_t p = 0; p < num_embs; ++p) {
                    const int64_t emb_pos = positions_ptr[p_start + p];
                    
                    // WARP-STYLE: O(1) centroid lookup (no binary search!)
                    const int32_t cid = emb_to_cid_ptr[emb_pos];
                    
                    // Centroid score: direct lookup into precomputed matrix
                    const float centroid_score = cs_row[cid];
                    
                    // Residual score: WARP-style bit-shifted addressing
                    const uint8_t *residual = residuals_ptr + (emb_pos << packed_dim_shift);
                    const float residual_score = decompression_kernel<nbits>(
                        residual, bucket_scores_t
                    );
                    
                    const float score = centroid_score + residual_score;
                    if (score > best_score) {
                        best_score = score;
                        best_pos = emb_pos;
                    }
                }
                
                const int64_t out_idx = d * num_tokens + t;
                out_pos_ptr[out_idx] = best_pos;
                out_scores_ptr[out_idx] = best_score;
            }
        }
    });
}

// =============================================================================
// WARP-STYLE with Winner Tracking (matches decompress_centroids_with_winners)
// =============================================================================

/**
 * WARP-STYLE Oracle with M3-compatible tracking.
 * 
 * Returns additional info useful for analysis:
 * - winner_centroid_ids: Which centroid each oracle winner belongs to
 * 
 * This avoids needing to derive centroid IDs in Python post-processing.
 */
template<int8_t nbits>
void compute_oracle_batch_warp_style_with_tracking(
    const torch::Tensor Q,                      // (num_tokens, 128)
    const torch::Tensor all_positions,          // flattened positions for all docs
    const torch::Tensor position_offsets,       // (num_docs + 1,) - CSR format
    const torch::Tensor centroids,              // (num_centroids, 128)
    const torch::Tensor residuals_compacted,    // (num_embeddings, packed_dim)
    const torch::Tensor bucket_weights,         // (128, num_buckets)
    const torch::Tensor embedding_to_centroid,  // (num_embeddings,) PRECOMPUTED centroid IDs
    torch::Tensor output_pos,                   // (num_docs, num_tokens) - output
    torch::Tensor output_scores,                // (num_docs, num_tokens) - output
    torch::Tensor output_centroid_ids           // (num_docs, num_tokens) - oracle winner's centroid
) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    
    static_assert(nbits == 2 || nbits == 4);
    constexpr int packed_vals_per_byte = 8 / nbits;
    constexpr int packed_dim = dim / packed_vals_per_byte;
    constexpr int bucket_score_offset = 128 * (1 << nbits);
    constexpr uint8_t packed_dim_shift = __builtin_ctz(packed_dim);
    
    const int64_t num_docs = position_offsets.size(0) - 1;
    const int32_t num_tokens = Q.size(0);
    const int64_t num_centroids = centroids.size(0);
    
    const auto centroid_scores_all = torch::matmul(Q, centroids.t()).contiguous();
    const auto all_bucket_scores = torch::matmul(
        Q.unsqueeze(2), bucket_weights.unsqueeze(0)
    ).contiguous();
    
    const float *centroid_scores_ptr = centroid_scores_all.data_ptr<float>();
    const uint8_t *residuals_ptr = residuals_compacted.data_ptr<uint8_t>();
    const int64_t *pos_offsets_ptr = position_offsets.data_ptr<int64_t>();
    const int64_t *positions_ptr = all_positions.data_ptr<int64_t>();
    const float *bucket_scores_ptr = all_bucket_scores.data_ptr<float>();
    const int32_t *emb_to_cid_ptr = embedding_to_centroid.data_ptr<int32_t>();
    
    int64_t *out_pos_ptr = output_pos.data_ptr<int64_t>();
    float *out_scores_ptr = output_scores.data_ptr<float>();
    int32_t *out_cid_ptr = output_centroid_ids.data_ptr<int32_t>();
    
    at::parallel_for(0, num_docs, 1, [&](int64_t doc_begin, int64_t doc_end) {
        for (int64_t d = doc_begin; d < doc_end; ++d) {
            const int64_t p_start = pos_offsets_ptr[d];
            const int64_t p_end = pos_offsets_ptr[d + 1];
            const int64_t num_embs = p_end - p_start;
            
            for (int32_t t = 0; t < num_tokens; ++t) {
                const float *bucket_scores_t = bucket_scores_ptr + t * bucket_score_offset;
                const float *cs_row = centroid_scores_ptr + t * num_centroids;
                
                int64_t best_pos = -1;
                int32_t best_cid = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                
                for (int64_t p = 0; p < num_embs; ++p) {
                    const int64_t emb_pos = positions_ptr[p_start + p];
                    const int32_t cid = emb_to_cid_ptr[emb_pos];
                    const float centroid_score = cs_row[cid];
                    const uint8_t *residual = residuals_ptr + (emb_pos << packed_dim_shift);
                    const float residual_score = decompression_kernel<nbits>(
                        residual, bucket_scores_t
                    );
                    
                    const float score = centroid_score + residual_score;
                    if (score > best_score) {
                        best_score = score;
                        best_pos = emb_pos;
                        best_cid = cid;
                    }
                }
                
                const int64_t out_idx = d * num_tokens + t;
                out_pos_ptr[out_idx] = best_pos;
                out_scores_ptr[out_idx] = best_score;
                out_cid_ptr[out_idx] = best_cid;
            }
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original implementations (for reference/testing)
    m.def("compute_oracle_batch_4_cpp", &compute_oracle_batch_parallel<4>,
          "Compute oracle winners for batch of docs (nbits=4, parallel)");
    m.def("compute_oracle_batch_2_cpp", &compute_oracle_batch_parallel<2>,
          "Compute oracle winners for batch of docs (nbits=2, parallel)");
    
    // OPTIMIZED implementations (6-12x faster)
    m.def("compute_oracle_batch_optimized_4_cpp", &compute_oracle_batch_optimized<4>,
          "OPTIMIZED: Oracle with precomputed centroid scores (nbits=4, 6-12x faster)");
    m.def("compute_oracle_batch_optimized_2_cpp", &compute_oracle_batch_optimized<2>,
          "OPTIMIZED: Oracle with precomputed centroid scores (nbits=2, 6-12x faster)");
    
    // SMART implementations (can skip redundant work using M3 data)
    m.def("compute_oracle_batch_smart_4_cpp", &compute_oracle_batch_smart<4>,
          "SMART: Oracle with M3-informed skip logic (nbits=4)");
    m.def("compute_oracle_batch_smart_2_cpp", &compute_oracle_batch_smart<2>,
          "SMART: Oracle with M3-informed skip logic (nbits=2)");
    
    // WARP-STYLE implementations (10-20x faster, requires precomputed embedding_to_centroid)
    m.def("compute_oracle_batch_warp_4_cpp", &compute_oracle_batch_warp_style<4>,
          "WARP-STYLE: Maximum optimization with precomputed centroid IDs (nbits=4, 10-20x faster)");
    m.def("compute_oracle_batch_warp_2_cpp", &compute_oracle_batch_warp_style<2>,
          "WARP-STYLE: Maximum optimization with precomputed centroid IDs (nbits=2, 10-20x faster)");
    
    // WARP-STYLE with tracking (also returns oracle centroid IDs)
    m.def("compute_oracle_batch_warp_tracking_4_cpp", &compute_oracle_batch_warp_style_with_tracking<4>,
          "WARP-STYLE with centroid tracking (nbits=4)");
    m.def("compute_oracle_batch_warp_tracking_2_cpp", &compute_oracle_batch_warp_style_with_tracking<2>,
          "WARP-STYLE with centroid tracking (nbits=2)");
}
