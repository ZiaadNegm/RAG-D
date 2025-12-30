#include <torch/extension.h>

#include <vector>

#include "annotated_stride_view.hpp"

// NOTE Used to max-reduce token-level of different clusters.
struct reduce_max_combiner {
    float operator()(float lhs, float rhs) const noexcept {
        return std::max(lhs, rhs);
    }
    float lhs(float lhs) const noexcept {
        return lhs;
    }
    float rhs(float rhs) const noexcept {
        return rhs;
    }
};

// NOTE Used to combine token-level scores into document-level scores.
struct reduce_sum_mse_combiner {
    reduce_sum_mse_combiner(float lhs_mse, float rhs_mse)
        : lhs_mse_(lhs_mse), rhs_mse_(rhs_mse) {}
    float operator()(float lhs, float rhs) const noexcept {
        return lhs + rhs;
    }
    float lhs(float lhs) const noexcept {
        return lhs + rhs_mse_;
    }
    float rhs(float rhs) const noexcept {
        return lhs_mse_ + rhs;
    }
private:
    float lhs_mse_, rhs_mse_;
};

// NOTE Used to combine token-level scores into document-level scores WITH influential token counting.
// A token is "influential" for a document if the doc has an observed score (not MSE fallback).
struct reduce_sum_mse_combiner_with_counts {
    reduce_sum_mse_combiner_with_counts(float lhs_mse, float rhs_mse)
        : lhs_mse_(lhs_mse), rhs_mse_(rhs_mse) {}
    
    // Both sides have observed scores: sum scores and sum counts
    float operator()(float lhs, float rhs) const noexcept {
        return lhs + rhs;
    }
    int16_t combine_counts(int16_t lhs_count, int16_t rhs_count) const noexcept {
        return lhs_count + rhs_count;
    }
    
    // Only LHS has observed score: add MSE for RHS, keep LHS count
    float lhs(float lhs) const noexcept {
        return lhs + rhs_mse_;
    }
    int16_t lhs_count(int16_t lhs_count) const noexcept {
        return lhs_count;  // RHS tokens used MSE fallback, not influential
    }
    
    // Only RHS has observed score: add MSE for LHS, keep RHS count
    float rhs(float rhs) const noexcept {
        return lhs_mse_ + rhs;
    }
    int16_t rhs_count(int16_t rhs_count) const noexcept {
        return rhs_count;  // LHS tokens used MSE fallback, not influential
    }
    
private:
    float lhs_mse_, rhs_mse_;
};

template<typename combiner_type>
void merge_candidate_strides(const annotated_stride_view<> stride1,
                             const annotated_stride_view<> stride2,
                             annotated_stride_view<> result,
                             combiner_type combiner) {
    const int32_t c1_size = *stride1.size_, c2_size = *stride2.size_;
    int32_t result_size = 0, i1 = 0, i2 = 0;
    while (i1 < c1_size && i2 < c2_size) {
        const int32_t key1 = stride1.keys_[i1];
        const int32_t key2 = stride2.keys_[i2];
        result.keys_[result_size] = std::min(key1, key2);
        if (key1 == key2) {
            result.data_[result_size] = combiner(stride1.data_[i1++], stride2.data_[i2++]);
        } else if (key1 < key2) {
            result.data_[result_size] = combiner.lhs(stride1.data_[i1++]);
        } else {
            result.data_[result_size] = combiner.rhs(stride2.data_[i2++]);
        }
        ++result_size;
    }
    if (i1 < c1_size) {
        for (; i1 < c1_size; ++i1) {
            result.keys_[result_size] = stride1.keys_[i1];
            result.data_[result_size] = combiner.lhs(stride1.data_[i1]);
            ++result_size;
        }
    }
    if (i2 < c2_size) {
        for (; i2 < c2_size; ++i2) {
            result.keys_[result_size] = stride2.keys_[i2];
            result.data_[result_size] = combiner.rhs(stride2.data_[i2]);
            ++result_size;
        }
    }
    *result.size_ = result_size;
}

void copy_candidate_stride(const annotated_stride_view<> source,
                           annotated_stride_view<> destination) {
    const int32_t size = *source.size_;
    *destination.size_ = size;
    memcpy(destination.keys_, source.keys_, size * sizeof(int32_t));
    memcpy(destination.data_, source.data_, size * sizeof(int32_t));
}

// Copy stride with winner positions for M3 tracking
void copy_candidate_stride_with_winners(const annotated_stride_view<> source,
                                        annotated_stride_view<> destination) {
    const int32_t size = *source.size_;
    *destination.size_ = size;
    memcpy(destination.keys_, source.keys_, size * sizeof(int32_t));
    memcpy(destination.data_, source.data_, size * sizeof(float));
    memcpy(destination.winner_pos_, source.winner_pos_, size * sizeof(int64_t));
}

// Copy stride with counts for influential token tracking
void copy_candidate_stride_with_counts(const annotated_stride_view<> source,
                                       annotated_stride_view<> destination) {
    const int32_t size = *source.size_;
    *destination.size_ = size;
    memcpy(destination.keys_, source.keys_, size * sizeof(int32_t));
    memcpy(destination.data_, source.data_, size * sizeof(float));
    memcpy(destination.counts_, source.counts_, size * sizeof(int16_t));
}

// Merge candidate strides with influential token count propagation
void merge_candidate_strides_with_counts(
        const annotated_stride_view<> stride1,
        const annotated_stride_view<> stride2,
        annotated_stride_view<> result,
        reduce_sum_mse_combiner_with_counts combiner) {
    const int32_t c1_size = *stride1.size_, c2_size = *stride2.size_;
    int32_t result_size = 0, i1 = 0, i2 = 0;
    while (i1 < c1_size && i2 < c2_size) {
        const int32_t key1 = stride1.keys_[i1];
        const int32_t key2 = stride2.keys_[i2];
        result.keys_[result_size] = std::min(key1, key2);
        if (key1 == key2) {
            // Doc in both strides: sum scores and sum counts
            result.data_[result_size] = combiner(stride1.data_[i1], stride2.data_[i2]);
            result.counts_[result_size] = combiner.combine_counts(stride1.counts_[i1], stride2.counts_[i2]);
            ++i1; ++i2;
        } else if (key1 < key2) {
            // Doc only in LHS: add MSE for RHS, keep LHS count
            result.data_[result_size] = combiner.lhs(stride1.data_[i1]);
            result.counts_[result_size] = combiner.lhs_count(stride1.counts_[i1]);
            ++i1;
        } else {
            // Doc only in RHS: add MSE for LHS, keep RHS count
            result.data_[result_size] = combiner.rhs(stride2.data_[i2]);
            result.counts_[result_size] = combiner.rhs_count(stride2.counts_[i2]);
            ++i2;
        }
        ++result_size;
    }
    // Drain remaining LHS
    for (; i1 < c1_size; ++i1) {
        result.keys_[result_size] = stride1.keys_[i1];
        result.data_[result_size] = combiner.lhs(stride1.data_[i1]);
        result.counts_[result_size] = combiner.lhs_count(stride1.counts_[i1]);
        ++result_size;
    }
    // Drain remaining RHS
    for (; i2 < c2_size; ++i2) {
        result.keys_[result_size] = stride2.keys_[i2];
        result.data_[result_size] = combiner.rhs(stride2.data_[i2]);
        result.counts_[result_size] = combiner.rhs_count(stride2.counts_[i2]);
        ++result_size;
    }
    *result.size_ = result_size;
}

// Merge candidate strides with winner position propagation (for M3 Tier B)
// During MAX merge, the winner position from the higher-scoring stride is kept
void merge_candidate_strides_with_winners(
        const annotated_stride_view<> stride1,
        const annotated_stride_view<> stride2,
        annotated_stride_view<> result,
        reduce_max_combiner combiner) {
    const int32_t c1_size = *stride1.size_, c2_size = *stride2.size_;
    int32_t result_size = 0, i1 = 0, i2 = 0;
    while (i1 < c1_size && i2 < c2_size) {
        const int32_t key1 = stride1.keys_[i1];
        const int32_t key2 = stride2.keys_[i2];
        result.keys_[result_size] = std::min(key1, key2);
        if (key1 == key2) {
            // Same doc in both strides: take max score and its winner position
            const float score1 = stride1.data_[i1];
            const float score2 = stride2.data_[i2];
            const bool left_wins = (score1 >= score2);
            result.data_[result_size] = left_wins ? score1 : score2;
            result.winner_pos_[result_size] = left_wins ? stride1.winner_pos_[i1] : stride2.winner_pos_[i2];
            ++i1; ++i2;
        } else if (key1 < key2) {
            result.data_[result_size] = combiner.lhs(stride1.data_[i1]);
            result.winner_pos_[result_size] = stride1.winner_pos_[i1];
            ++i1;
        } else {
            result.data_[result_size] = combiner.rhs(stride2.data_[i2]);
            result.winner_pos_[result_size] = stride2.winner_pos_[i2];
            ++i2;
        }
        ++result_size;
    }
    // Drain remaining LHS
    for (; i1 < c1_size; ++i1) {
        result.keys_[result_size] = stride1.keys_[i1];
        result.data_[result_size] = combiner.lhs(stride1.data_[i1]);
        result.winner_pos_[result_size] = stride1.winner_pos_[i1];
        ++result_size;
    }
    // Drain remaining RHS
    for (; i2 < c2_size; ++i2) {
        result.keys_[result_size] = stride2.keys_[i2];
        result.data_[result_size] = combiner.rhs(stride2.data_[i2]);
        result.winner_pos_[result_size] = stride2.winner_pos_[i2];
        ++result_size;
    }
    *result.size_ = result_size;
}

// Merge the `nprobe` candidate lists associated with a specific token index.
int merge_candidates_nprobe(std::vector<annotated_stride_view<>> &views,
                            std::vector<annotated_stride_view<>> &views_buffer,
                            const int nprobe, const int query_token_idx) {
    int num_iterations = 0;
    const int begin = query_token_idx * nprobe;
    std::vector<annotated_stride_view<>> *buf1 = &views, *buf2 = &views_buffer;
    reduce_max_combiner combiner;
    for (int step_size = 1; step_size < nprobe; step_size <<= 1, ++num_iterations) {
        for (int lhs = 0; lhs < nprobe; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < nprobe) {
                merge_candidate_strides<>((*buf1)[begin + lhs], (*buf1)[begin + rhs], 
                                          (*buf2)[begin + lhs], combiner);
            } else {
                // NOTE If rhs < nprobe we don't have a merge partner for the current index.
                // In this case, move the current view to the next stage without alteration.
                copy_candidate_stride((*buf1)[begin + lhs],
                                      (*buf2)[begin + lhs]);
            }
        }
        // NOTE change which buffer is considered a "scratch" buffer.
        std::swap(buf1, buf2);
    }
    return num_iterations;
}

// Merge nprobe centroids for a token WITH winner position tracking for M3
int merge_candidates_nprobe_with_winners(std::vector<annotated_stride_view<>> &views,
                                         std::vector<annotated_stride_view<>> &views_buffer,
                                         const int nprobe, const int query_token_idx) {
    int num_iterations = 0;
    const int begin = query_token_idx * nprobe;
    std::vector<annotated_stride_view<>> *buf1 = &views, *buf2 = &views_buffer;
    reduce_max_combiner combiner;
    for (int step_size = 1; step_size < nprobe; step_size <<= 1, ++num_iterations) {
        for (int lhs = 0; lhs < nprobe; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < nprobe) {
                merge_candidate_strides_with_winners((*buf1)[begin + lhs], (*buf1)[begin + rhs], 
                                                     (*buf2)[begin + lhs], combiner);
            } else {
                // No merge partner - copy with winners
                copy_candidate_stride_with_winners((*buf1)[begin + lhs],
                                                   (*buf2)[begin + lhs]);
            }
        }
        std::swap(buf1, buf2);
    }
    return num_iterations;
}

// Merge the 32 strides of token-level scores into a single stride of document-level scores.
void merge_candidates_tokens(std::vector<annotated_stride_view<>> &views,
                            std::vector<annotated_stride_view<>> &views_buffer,
                            const int nprobe, const float *mse_estimates) {
    constexpr int num_tokens = 32;
    std::array<float, num_tokens + 1> mse_prefix;
    mse_prefix[0] = 0;
    for (int i = 0; i < num_tokens; ++i) {
        mse_prefix[i + 1] = mse_prefix[i] + mse_estimates[i];
    }
    for (int step_size = 1; step_size < num_tokens; step_size <<= 1) {
        for (int lhs = 0; lhs < num_tokens; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < num_tokens) {
                // NOTE We can just subtract two prefix sums for the range of MSE values!
                const float lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                const float rhs_mse = mse_prefix[std::min(rhs + step_size, num_tokens)] - mse_prefix[rhs];

                reduce_sum_mse_combiner combiner(lhs_mse, rhs_mse);
                merge_candidate_strides<>(views[lhs * nprobe], views[rhs * nprobe],
                                          views_buffer[lhs * nprobe], combiner);
            } else {
                copy_candidate_stride(views[lhs * nprobe], views_buffer[lhs * nprobe]);
            }
        }
        std::swap(views, views_buffer);
    }
}

// Merge token-level scores into document-level scores WITH influential token counting.
// Each doc starts with count=1 (present in that token's candidate list = influential for that token).
// During merge, counts are summed when docs appear in both strides; preserved otherwise.
void merge_candidates_tokens_with_counts(
        std::vector<annotated_stride_view<>> &views,
        std::vector<annotated_stride_view<>> &views_buffer,
        const int nprobe, const float *mse_estimates) {
    constexpr int num_tokens = 32;
    std::array<float, num_tokens + 1> mse_prefix;
    mse_prefix[0] = 0;
    for (int i = 0; i < num_tokens; ++i) {
        mse_prefix[i + 1] = mse_prefix[i] + mse_estimates[i];
    }
    for (int step_size = 1; step_size < num_tokens; step_size <<= 1) {
        for (int lhs = 0; lhs < num_tokens; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < num_tokens) {
                const float lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                const float rhs_mse = mse_prefix[std::min(rhs + step_size, num_tokens)] - mse_prefix[rhs];

                reduce_sum_mse_combiner_with_counts combiner(lhs_mse, rhs_mse);
                merge_candidate_strides_with_counts(views[lhs * nprobe], views[rhs * nprobe],
                                                   views_buffer[lhs * nprobe], combiner);
            } else {
                copy_candidate_stride_with_counts(views[lhs * nprobe], views_buffer[lhs * nprobe]);
            }
        }
        std::swap(views, views_buffer);
    }
}

std::vector<int> partial_sort_results(annotated_stride_view<> stride,
                          const int num_results) {
    std::vector<int> pid_idx(*stride.size_);
    std::iota(pid_idx.begin(), pid_idx.end(), 0);

    const float *scores = stride.data_;
    std::partial_sort(pid_idx.begin(), pid_idx.begin() + num_results,
                      pid_idx.end(), [scores](const int idx1, const int idx2){
        const float score1 = scores[idx1], score2 = scores[idx2];
        return (score1 > score2) || (score1 == score2 && idx1 < idx2);
    });

    return pid_idx;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> merge_candidate_scores(
        const torch::Tensor candidate_capacities,
        const torch::Tensor candidate_sizes,
        const torch::Tensor candidate_pids_strided,
        const torch::Tensor candidate_scores_strided,
        const torch::Tensor mse_estimates,
        const int nprobe, const int k) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    const int num_cells = candidate_capacities.size(0);
    const int num_candidates = candidate_pids_strided.size(0);

    std::vector<annotated_stride_view<>> views = strided_view(
        candidate_capacities, candidate_sizes, 
        candidate_pids_strided, candidate_scores_strided
    );

    // Local buffers used for merging.
    torch::Tensor size_buffer = torch::zeros({num_cells}, torch::kInt32);
    torch::Tensor pid_buffer = torch::zeros({num_candidates}, torch::kInt32);
    torch::Tensor score_buffer = torch::zeros({num_candidates}, torch::kFloat32);

    // NOTE this scheme guarantees non-overlapping partitions
    std::vector<annotated_stride_view<>> views_buffer = strided_view(
        candidate_capacities, size_buffer, pid_buffer, score_buffer
    );

    int num_iterations;
    for (int query_token_idx = 0; query_token_idx < 32; ++query_token_idx) {
        // TODO(jlscheerer) Add early stopping here. In case we don't have 32 tokens!
        num_iterations = merge_candidates_nprobe(views, views_buffer, nprobe, query_token_idx);
    }
    // NOTE If we performed an odd number of iterations the scratch buffer contains the result.
    if (num_iterations % 2 != 0) {
        std::swap(views, views_buffer);
    }

    // Initialize counts to 1 for each doc in each token's merged stride.
    // A doc present in token i's stride has 1 influential token (token i).
    torch::Tensor counts = torch::ones({num_candidates}, torch::kInt16);
    torch::Tensor counts_buffer = torch::zeros({num_candidates}, torch::kInt16);

    // Create views with counts for token-level merge
    std::vector<annotated_stride_view<>> views_with_counts = strided_view_with_counts(
        candidate_capacities, candidate_sizes,
        candidate_pids_strided, candidate_scores_strided, counts
    );
    std::vector<annotated_stride_view<>> views_buffer_with_counts = strided_view_with_counts(
        candidate_capacities, size_buffer, pid_buffer, score_buffer, counts_buffer
    );

    // Copy current merged data into views_with_counts (scores and pids are already there via shared tensors)
    // But we need to sync sizes and repoint views to current data after nprobe merge
    for (int token_idx = 0; token_idx < 32; ++token_idx) {
        const int idx = token_idx * nprobe;
        *views_with_counts[idx].size_ = *views[idx].size_;
        // Copy pids and scores from views to views_with_counts (they share underlying tensors for pids/scores)
        const int32_t size = *views[idx].size_;
        memcpy(views_with_counts[idx].keys_, views[idx].keys_, size * sizeof(int32_t));
        memcpy(views_with_counts[idx].data_, views[idx].data_, size * sizeof(float));
    }

    // Finally merge the results *between* different tokens WITH count tracking.
    merge_candidates_tokens_with_counts(views_with_counts, views_buffer_with_counts, nprobe, mse_estimates.data_ptr<float>());

    // NOTE After all merges have occured the stride at index 0 contains the resulting scores.
    // Capture unique docs count before filtering to top-k
    const int unique_docs_touched = *(views_with_counts[0].size_);
    const int num_results = std::min(unique_docs_touched, k);
    std::vector<int> pid_idx = partial_sort_results(views_with_counts[0], num_results);

    torch::Tensor candidate_pids = torch::zeros({num_results}, torch::kInt32);
    torch::Tensor candidate_scores = torch::zeros({num_results}, torch::kFloat32);
    torch::Tensor influential_counts = torch::zeros({num_results}, torch::kInt16);

    const int32_t *pids_ptr = views_with_counts[0].keys_;
    const float *scores_ptr = views_with_counts[0].data_;
    const int16_t *counts_ptr = views_with_counts[0].counts_;

    int32_t *candidate_pids_ptr = candidate_pids.data_ptr<int32_t>();
    float *candidate_scores_ptr = candidate_scores.data_ptr<float>();
    int16_t *influential_counts_ptr = influential_counts.data_ptr<int16_t>();
    for (int i = 0; i < num_results; ++i) {
        const int idx = pid_idx[i];
        candidate_pids_ptr[i] = pids_ptr[idx];
        candidate_scores_ptr[i] = scores_ptr[idx];
        influential_counts_ptr[i] = counts_ptr[idx];
    }

    return {std::move(candidate_pids), std::move(candidate_scores), std::move(influential_counts), unique_docs_touched};
}

// Variant of merge_candidate_scores that also extracts Phase 1 winners (per-token MaxSim winners)
// before Phase 2 merge. This enables M3 Tier B measurement: tracking which embedding won
// for each (query_token, doc) pair.
//
// Returns: (candidate_pids, candidate_scores, influential_counts, unique_docs, 
//           phase1_pids[32], phase1_scores[32], phase1_winners[32], phase1_sizes)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor> 
merge_candidate_scores_with_winners(
        const torch::Tensor candidate_capacities,
        const torch::Tensor candidate_sizes,
        const torch::Tensor candidate_pids_strided,
        const torch::Tensor candidate_scores_strided,
        const torch::Tensor candidate_winners_strided,  // NEW: winner positions from decompression
        const torch::Tensor mse_estimates,
        const int nprobe, const int k) {
    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;
    const int num_cells = candidate_capacities.size(0);
    const int num_candidates = candidate_pids_strided.size(0);

    // Create views WITH winner tracking
    std::vector<annotated_stride_view<>> views = strided_view_with_winners(
        candidate_capacities, candidate_sizes, 
        candidate_pids_strided, candidate_scores_strided, candidate_winners_strided
    );

    // Local buffers used for merging (also with winner positions)
    torch::Tensor size_buffer = torch::zeros({num_cells}, torch::kInt32);
    torch::Tensor pid_buffer = torch::zeros({num_candidates}, torch::kInt32);
    torch::Tensor score_buffer = torch::zeros({num_candidates}, torch::kFloat32);
    torch::Tensor winner_buffer = torch::zeros({num_candidates}, torch::kInt64);

    std::vector<annotated_stride_view<>> views_buffer = strided_view_with_winners(
        candidate_capacities, size_buffer, pid_buffer, score_buffer, winner_buffer
    );

    // Phase 1: Merge nprobe centroids per token WITH winner tracking
    int num_iterations;
    for (int query_token_idx = 0; query_token_idx < 32; ++query_token_idx) {
        num_iterations = merge_candidates_nprobe_with_winners(views, views_buffer, nprobe, query_token_idx);
    }
    // If odd iterations, swap back
    if (num_iterations % 2 != 0) {
        std::swap(views, views_buffer);
    }

    // === EXTRACTION STEP: Save Phase 1 results before Phase 2 overwrites them ===
    // For each of the 32 tokens, extract the merged stride (doc_ids, scores, winner_positions)
    std::vector<torch::Tensor> phase1_pids(32);
    std::vector<torch::Tensor> phase1_scores(32);
    std::vector<torch::Tensor> phase1_winners(32);
    torch::Tensor phase1_sizes = torch::zeros({32}, torch::kInt32);
    int32_t* phase1_sizes_ptr = phase1_sizes.data_ptr<int32_t>();
    
    for (int token_idx = 0; token_idx < 32; ++token_idx) {
        const int idx = token_idx * nprobe;  // After merge, result is at stride 0 for this token
        const int32_t size = *views[idx].size_;
        phase1_sizes_ptr[token_idx] = size;
        
        // Copy out the data before Phase 2 overwrites it
        phase1_pids[token_idx] = torch::zeros({size}, torch::kInt32);
        phase1_scores[token_idx] = torch::zeros({size}, torch::kFloat32);
        phase1_winners[token_idx] = torch::zeros({size}, torch::kInt64);
        
        memcpy(phase1_pids[token_idx].data_ptr<int32_t>(), views[idx].keys_, size * sizeof(int32_t));
        memcpy(phase1_scores[token_idx].data_ptr<float>(), views[idx].data_, size * sizeof(float));
        memcpy(phase1_winners[token_idx].data_ptr<int64_t>(), views[idx].winner_pos_, size * sizeof(int64_t));
    }

    // === Continue with Phase 2 (same as original, with influential_counts) ===
    
    // Initialize counts to 1 for each doc in each token's merged stride.
    torch::Tensor counts = torch::ones({num_candidates}, torch::kInt16);
    torch::Tensor counts_buffer = torch::zeros({num_candidates}, torch::kInt16);

    // Create fresh views with counts for Phase 2 (reuse existing pids/scores tensors)
    std::vector<annotated_stride_view<>> views_with_counts = strided_view_with_counts(
        candidate_capacities, candidate_sizes,
        candidate_pids_strided, candidate_scores_strided, counts
    );
    std::vector<annotated_stride_view<>> views_buffer_with_counts = strided_view_with_counts(
        candidate_capacities, size_buffer, pid_buffer, score_buffer, counts_buffer
    );

    // Copy Phase 1 merged data into views_with_counts
    for (int token_idx = 0; token_idx < 32; ++token_idx) {
        const int idx = token_idx * nprobe;
        const int32_t size = phase1_sizes_ptr[token_idx];
        *views_with_counts[idx].size_ = size;
        memcpy(views_with_counts[idx].keys_, phase1_pids[token_idx].data_ptr<int32_t>(), size * sizeof(int32_t));
        memcpy(views_with_counts[idx].data_, phase1_scores[token_idx].data_ptr<float>(), size * sizeof(float));
    }

    // Phase 2: Merge tokens with count tracking
    merge_candidates_tokens_with_counts(views_with_counts, views_buffer_with_counts, nprobe, mse_estimates.data_ptr<float>());

    // Extract final results
    const int unique_docs_touched = *(views_with_counts[0].size_);
    const int num_results = std::min(unique_docs_touched, k);
    std::vector<int> pid_idx = partial_sort_results(views_with_counts[0], num_results);

    torch::Tensor candidate_pids = torch::zeros({num_results}, torch::kInt32);
    torch::Tensor candidate_scores = torch::zeros({num_results}, torch::kFloat32);
    torch::Tensor influential_counts = torch::zeros({num_results}, torch::kInt16);

    const int32_t *pids_ptr = views_with_counts[0].keys_;
    const float *scores_ptr = views_with_counts[0].data_;
    const int16_t *counts_ptr = views_with_counts[0].counts_;

    int32_t *candidate_pids_ptr = candidate_pids.data_ptr<int32_t>();
    float *candidate_scores_ptr = candidate_scores.data_ptr<float>();
    int16_t *influential_counts_ptr = influential_counts.data_ptr<int16_t>();
    for (int i = 0; i < num_results; ++i) {
        const int idx = pid_idx[i];
        candidate_pids_ptr[i] = pids_ptr[idx];
        candidate_scores_ptr[i] = scores_ptr[idx];
        influential_counts_ptr[i] = counts_ptr[idx];
    }

    return {std::move(candidate_pids), std::move(candidate_scores), std::move(influential_counts), 
            unique_docs_touched,
            std::move(phase1_pids), std::move(phase1_scores), std::move(phase1_winners), std::move(phase1_sizes)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_candidate_scores_cpp", &merge_candidate_scores,
        "Merge Strided Candidate Scores");
  m.def("merge_candidate_scores_with_winners_cpp", &merge_candidate_scores_with_winners,
        "Merge Strided Candidate Scores with Phase 1 Winner Extraction for M3");
}