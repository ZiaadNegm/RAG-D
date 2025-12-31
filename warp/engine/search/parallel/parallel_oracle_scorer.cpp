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
// Oracle Batch Computation (parallel over documents)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_oracle_batch_4_cpp", &compute_oracle_batch_parallel<4>,
          "Compute oracle winners for batch of docs (nbits=4, parallel)");
    m.def("compute_oracle_batch_2_cpp", &compute_oracle_batch_parallel<2>,
          "Compute oracle winners for batch of docs (nbits=2, parallel)");
}
