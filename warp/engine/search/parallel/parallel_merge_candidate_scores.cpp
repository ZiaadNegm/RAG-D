#include <torch/extension.h>

#include <vector>

#include "task_graph.hpp"
#include "../annotated_stride_view.hpp"

constexpr int max_num_tokens = 32;

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

// Combiner for Phase 2 that also tracks influential token counts
struct reduce_sum_mse_combiner_with_counts {
    reduce_sum_mse_combiner_with_counts(float lhs_mse, float rhs_mse)
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
    // Count operations: sum counts when doc is in both strides, preserve otherwise
    int16_t combine_counts(int16_t lhs, int16_t rhs) const noexcept {
        return lhs + rhs;
    }
    int16_t lhs_count(int16_t lhs) const noexcept {
        return lhs;  // Doc only in LHS, keep LHS count
    }
    int16_t rhs_count(int16_t rhs) const noexcept {
        return rhs;  // Doc only in RHS, keep RHS count
    }
private:
    float lhs_mse_, rhs_mse_;
};

enum class reduction_type {
    kMaxReduce, kSumReduce
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

// Merge the 32 strides of token-level scores into a single stride WITH count tracking.
void merge_candidates_tokens_with_counts(
        std::vector<annotated_stride_view<>> &views,
        std::vector<annotated_stride_view<>> &views_buffer,
        const int nprobe, const float *mse_estimates) {
    std::array<float, max_num_tokens + 1> mse_prefix;
    mse_prefix[0] = 0;
    for (int i = 0; i < max_num_tokens; ++i) {
        mse_prefix[i + 1] = mse_prefix[i] + mse_estimates[i];
    }
    for (int step_size = 1; step_size < max_num_tokens; step_size <<= 1) {
        for (int lhs = 0; lhs < max_num_tokens; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < max_num_tokens) {
                const float lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                const float rhs_mse = mse_prefix[std::min(rhs + step_size, max_num_tokens)] - mse_prefix[rhs];

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

// Merge the 32 strides of token-level scores into a single stride of document-level scores.
void merge_candidates_tokens(std::vector<annotated_stride_view<>> &views,
                            std::vector<annotated_stride_view<>> &views_buffer,
                            const int nprobe, const float *mse_estimates) {
    
    std::array<float, max_num_tokens + 1> mse_prefix;
    mse_prefix[0] = 0;
    for (int i = 0; i < max_num_tokens; ++i) {
        mse_prefix[i + 1] = mse_prefix[i] + mse_estimates[i];
    }
    for (int step_size = 1; step_size < max_num_tokens; step_size <<= 1) {
        for (int lhs = 0; lhs < max_num_tokens; lhs += (step_size << 1)) {
            const int rhs = lhs + step_size;
            if (rhs < max_num_tokens) {
                // NOTE We can just subtract two prefix sums for the range of MSE values!
                const float lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                const float rhs_mse = mse_prefix[std::min(rhs + step_size, max_num_tokens - 1)] - mse_prefix[rhs];

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

struct merge_context {
    int nprobe;
    std::vector<annotated_stride_view<>> *data, *buffer;
    std::array<float, max_num_tokens + 1> *mse_prefix;
};

struct merge_task {
    using context_type = merge_context;

    reduction_type type;
    int begin_or_stepsize, lhs, rhs;

    static void max_reduce_stride(const int begin, const int lhs, const int rhs,
                                  std::vector<annotated_stride_view<>> * __restrict data,
                                  std::vector<annotated_stride_view<>> * __restrict buffer) {
        reduce_max_combiner combiner;
        annotated_stride_view<> &lhs_data = (*data)[begin + lhs];
        annotated_stride_view<> &rhs_data = (*data)[begin + rhs];
        annotated_stride_view<> &lhs_buffer = (*buffer)[begin + lhs];
        merge_candidate_strides<>(lhs_data, rhs_data, lhs_buffer, combiner);
        std::swap(lhs_data, lhs_buffer); // "promote" the result.
    }

    static void execute_task(const context_type &context, const merge_task &task) {
        if (task.type == reduction_type::kMaxReduce) {
            merge_task::max_reduce_stride(task.begin_or_stepsize, task.lhs, task.rhs, context.data, context.buffer);
        } else if (task.type == reduction_type::kSumReduce) {
            const int step_size = task.begin_or_stepsize;
            const float lhs_mse = (*context.mse_prefix)[task.rhs] - (*context.mse_prefix)[task.lhs];
            const float rhs_mse = (*context.mse_prefix)[std::min(task.rhs + step_size, max_num_tokens)] - (*context.mse_prefix)[task.rhs];

            reduce_sum_mse_combiner combiner(lhs_mse, rhs_mse);
            annotated_stride_view<> &lhs_data = (*context.data)[task.lhs * context.nprobe];
            annotated_stride_view<> &rhs_data = (*context.data)[task.rhs * context.nprobe];
            annotated_stride_view<> &lhs_buffer = (*context.buffer)[task.lhs * context.nprobe];
            merge_candidate_strides<>(lhs_data, rhs_data, lhs_buffer, combiner);
            std::swap(lhs_data, lhs_buffer); // "promote" the result.
        } else {
            __builtin_unreachable();
        }
    }
};

// Task struct for winner-aware MAX merge (used in Phase 1 of M3 tracking)
struct merge_task_with_winners {
    using context_type = merge_context;

    int begin, lhs, rhs;  // Only MAX reduce type needed (Phase 1 only)

    static void max_reduce_stride_with_winners(const int begin, const int lhs, const int rhs,
                                               std::vector<annotated_stride_view<>> * __restrict data,
                                               std::vector<annotated_stride_view<>> * __restrict buffer) {
        reduce_max_combiner combiner;
        annotated_stride_view<> &lhs_data = (*data)[begin + lhs];
        annotated_stride_view<> &rhs_data = (*data)[begin + rhs];
        annotated_stride_view<> &lhs_buffer = (*buffer)[begin + lhs];
        merge_candidate_strides_with_winners(lhs_data, rhs_data, lhs_buffer, combiner);
        std::swap(lhs_data, lhs_buffer); // "promote" the result.
    }

    static void execute_task(const context_type &context, const merge_task_with_winners &task) {
        merge_task_with_winners::max_reduce_stride_with_winners(task.begin, task.lhs, task.rhs, context.data, context.buffer);
    }
};

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

std::tuple<torch::Tensor, torch::Tensor> parallel_merge_candidate_scores(
        const torch::Tensor candidate_capacities,
        const torch::Tensor candidate_sizes,
        const torch::Tensor candidate_pids_strided,
        const torch::Tensor candidate_scores_strided,
        const torch::Tensor mse_estimates,
        const int nprobe,
        const int k,
        const int32_t num_query_tokens) {
    using warp::task_graph;
    using warp::task_ref;

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

    const int num_threads = at::get_num_threads();
    if (num_threads == 1) {
        int num_iterations;
        for (int query_token_idx = 0; query_token_idx < 32; ++query_token_idx) {
            // TODO(jlscheerer) Add early stopping here. In case we don't have 32 tokens!
            num_iterations = merge_candidates_nprobe(views, views_buffer, nprobe, query_token_idx);
        }
        // NOTE If we performed an odd number of iterations the scratch buffer contains the result.
        if (num_iterations % 2 != 0) {
            std::swap(views, views_buffer);
        }

        // Finally merge the results *between* different tokens.
        merge_candidates_tokens(views, views_buffer, nprobe, mse_estimates.data_ptr<float>());
    } else {
        const float *mse_estimates_ptr = mse_estimates.data_ptr<float>();
        std::array<float, max_num_tokens + 1> mse_prefix;
        mse_prefix[0] = 0;
        for (int i = 0; i < max_num_tokens; ++i) {
            mse_prefix[i + 1] = mse_prefix[i] + mse_estimates_ptr[i];
        }

        merge_context context = {
            .nprobe = nprobe,
            .data = &views,
            .buffer = &views_buffer,
            .mse_prefix = &mse_prefix
        };

        task_graph<merge_task> graph(
            std::move(context), num_query_tokens * (2 * nprobe - 1) + (2 * num_query_tokens - 1)
        );

        std::vector<task_ref> token_task_map(num_query_tokens, -1);
        std::vector<task_ref> probe_task_map(nprobe);

        // Add tasks for reducing the nprobe token score strides per query token.
        for (int query_token_idx = 0; query_token_idx < num_query_tokens; ++query_token_idx) {
            std::fill(probe_task_map.begin(), probe_task_map.end(), -1);
            const int begin = query_token_idx * nprobe;
            for (int step_size = 1; step_size < nprobe; step_size <<= 1) {
                for (int lhs = 0; lhs < nprobe; lhs += (step_size << 1)) {
                    if (lhs + step_size < nprobe) {
                        const int rhs = lhs + step_size;

                        // TODO(jlscheerer) Actually fix this.
                        task_ref task = graph.add({
                            .type = reduction_type::kMaxReduce,
                            .begin_or_stepsize = begin,
                            .lhs = lhs,
                            .rhs = rhs
                        });

                        const int pred1 = probe_task_map[lhs];
                        if (pred1 != -1) {
                            graph.mark_successor(pred1, task);
                        }
                        const int pred2 = probe_task_map[rhs];
                        if (pred2 != -1) {
                            graph.mark_successor(pred2, task);
                        }

                        probe_task_map[lhs] = task;
                    }
                }
            }
            // Mark "root" of the probe reduction as the start of the token reduction.
            token_task_map[query_token_idx] = probe_task_map[0];
        }

        // Add the token-level to document-level reduction steps
        for (int step_size = 1; step_size < num_query_tokens; step_size <<= 1) {
            for (int lhs = 0; lhs < num_query_tokens; lhs += (step_size << 1)) {
                if (lhs + step_size < num_query_tokens) {
                    const int rhs = lhs + step_size;
                    task_ref task = graph.add({
                        .type = reduction_type::kSumReduce,
                        .begin_or_stepsize = step_size,
                        .lhs = lhs,
                        .rhs = rhs
                    });
                    
                    const int pred1 = token_task_map[lhs];
                    graph.mark_successor(pred1, task);
                    
                    const int pred2 = token_task_map[rhs];
                    graph.mark_successor(pred2, task);

                    token_task_map[lhs] = task;
                }
            }
        }
        graph.run_all_tasks(num_threads);
    }

    // NOTE After all merges have occured the stride at index 0 contains the resulting scores.
    const int num_results = std::min(*(views[0].size_), k);
    std::vector<int> pid_idx = partial_sort_results(views[0], num_results);

    torch::Tensor candidate_pids = torch::zeros({num_results}, torch::kInt32);
    torch::Tensor candidate_scores = torch::zeros({num_results}, torch::kFloat32);

    const int32_t *pids_ptr = views[0].keys_;
    const float *scores_ptr = views[0].data_;

    int32_t *candidate_pids_ptr = candidate_pids.data_ptr<int32_t>();
    float *candidate_scores_ptr = candidate_scores.data_ptr<float>();
    for (int i = 0; i < num_results; ++i) {
        const int idx = pid_idx[i];
        candidate_pids_ptr[i] = pids_ptr[idx];
        candidate_scores_ptr[i] = scores_ptr[idx];
    }

    return {std::move(candidate_pids), std::move(candidate_scores)};
}

// Variant of parallel_merge_candidate_scores that extracts Phase 1 winners (per-token MaxSim winners)
// before Phase 2 merge. This enables M3 Tier B measurement and also adds influential_counts.
//
// This function uses a two-phase approach:
// 1. Phase 1: Merge nprobe centroids per token WITH winner tracking (using task graph for parallelism)
// 2. EXTRACTION: Copy Phase 1 results to persistent tensors before Phase 2 can overwrite them
// 3. Phase 2: Merge tokens into document-level scores WITH influential_counts tracking
//
// Returns: (candidate_pids, candidate_scores, influential_counts, unique_docs, 
//           phase1_pids[32], phase1_scores[32], phase1_winners[32], phase1_sizes)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor> 
parallel_merge_candidate_scores_with_winners(
        const torch::Tensor candidate_capacities,
        const torch::Tensor candidate_sizes,
        const torch::Tensor candidate_pids_strided,
        const torch::Tensor candidate_scores_strided,
        const torch::Tensor candidate_winners_strided,  // winner positions from decompression
        const torch::Tensor mse_estimates,
        const int nprobe,
        const int k,
        const int32_t num_query_tokens) {
    using warp::task_graph;
    using warp::task_ref;

    torch::NoGradGuard no_grad;
    torch::InferenceMode guard;

    const int num_cells = candidate_capacities.size(0);
    const int num_candidates = candidate_pids_strided.size(0);

    // Create views WITH winner tracking for Phase 1
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

    // ========================================================================
    // PHASE 1: Merge nprobe centroids per token WITH winner tracking
    // ========================================================================
    const int num_threads = at::get_num_threads();
    
    if (num_threads == 1) {
        // Single-threaded path
        int num_iterations;
        for (int query_token_idx = 0; query_token_idx < max_num_tokens; ++query_token_idx) {
            num_iterations = merge_candidates_nprobe_with_winners(views, views_buffer, nprobe, query_token_idx);
        }
        if (num_iterations % 2 != 0) {
            std::swap(views, views_buffer);
        }
    } else {
        // Multi-threaded path: use task graph for Phase 1 only
        // We'll run Phase 1 tasks, then extract, then run Phase 2 sequentially with counts
        
        const float *mse_estimates_ptr = mse_estimates.data_ptr<float>();
        std::array<float, max_num_tokens + 1> mse_prefix;
        mse_prefix[0] = 0;
        for (int i = 0; i < max_num_tokens; ++i) {
            mse_prefix[i + 1] = mse_prefix[i] + mse_estimates_ptr[i];
        }

        // Create context for Phase 1 only (MAX reduce tasks)
        merge_context context = {
            .nprobe = nprobe,
            .data = &views,
            .buffer = &views_buffer,
            .mse_prefix = &mse_prefix
        };

        // Only create Phase 1 tasks (nprobe reduction per token) using winner-aware task type
        const int num_phase1_tasks = num_query_tokens * (2 * nprobe - 1);
        task_graph<merge_task_with_winners> graph(std::move(context), num_phase1_tasks);

        std::vector<task_ref> probe_task_map(nprobe);

        // Add tasks for reducing the nprobe token score strides per query token
        for (int query_token_idx = 0; query_token_idx < num_query_tokens; ++query_token_idx) {
            std::fill(probe_task_map.begin(), probe_task_map.end(), -1);
            const int begin = query_token_idx * nprobe;
            for (int step_size = 1; step_size < nprobe; step_size <<= 1) {
                for (int lhs = 0; lhs < nprobe; lhs += (step_size << 1)) {
                    if (lhs + step_size < nprobe) {
                        const int rhs = lhs + step_size;

                        task_ref task = graph.add({
                            .begin = begin,
                            .lhs = lhs,
                            .rhs = rhs
                        });

                        const int pred1 = probe_task_map[lhs];
                        if (pred1 != -1) {
                            graph.mark_successor(pred1, task);
                        }
                        const int pred2 = probe_task_map[rhs];
                        if (pred2 != -1) {
                            graph.mark_successor(pred2, task);
                        }

                        probe_task_map[lhs] = task;
                    }
                }
            }
        }
        
        // Execute Phase 1 task graph (parallel)
        graph.run_all_tasks(num_threads);
    }

    // ========================================================================
    // EXTRACTION: Save Phase 1 results before Phase 2 overwrites them
    // ========================================================================
    std::vector<torch::Tensor> phase1_pids(max_num_tokens);
    std::vector<torch::Tensor> phase1_scores(max_num_tokens);
    std::vector<torch::Tensor> phase1_winners(max_num_tokens);
    torch::Tensor phase1_sizes = torch::zeros({max_num_tokens}, torch::kInt32);
    int32_t* phase1_sizes_ptr = phase1_sizes.data_ptr<int32_t>();
    
    for (int token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
        const int idx = token_idx * nprobe;  // After merge, result is at stride 0 for this token
        const int32_t size = *views[idx].size_;
        phase1_sizes_ptr[token_idx] = size;
        
        // Copy out the data before Phase 2 overwrites it
        phase1_pids[token_idx] = torch::zeros({size}, torch::kInt32);
        phase1_scores[token_idx] = torch::zeros({size}, torch::kFloat32);
        phase1_winners[token_idx] = torch::zeros({size}, torch::kInt64);
        
        if (size > 0) {
            memcpy(phase1_pids[token_idx].data_ptr<int32_t>(), views[idx].keys_, size * sizeof(int32_t));
            memcpy(phase1_scores[token_idx].data_ptr<float>(), views[idx].data_, size * sizeof(float));
            memcpy(phase1_winners[token_idx].data_ptr<int64_t>(), views[idx].winner_pos_, size * sizeof(int64_t));
        }
    }

    // ========================================================================
    // PHASE 2: Merge tokens into document-level scores WITH influential_counts
    // ========================================================================
    
    // Initialize counts to 1 for each doc in each token's merged stride
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
    for (int token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
        const int idx = token_idx * nprobe;
        const int32_t size = phase1_sizes_ptr[token_idx];
        *views_with_counts[idx].size_ = size;
        if (size > 0) {
            memcpy(views_with_counts[idx].keys_, phase1_pids[token_idx].data_ptr<int32_t>(), size * sizeof(int32_t));
            memcpy(views_with_counts[idx].data_, phase1_scores[token_idx].data_ptr<float>(), size * sizeof(float));
        }
    }

    // Phase 2: Merge tokens with count tracking (sequential - simpler and avoids complex task dependencies)
    merge_candidates_tokens_with_counts(views_with_counts, views_buffer_with_counts, nprobe, mse_estimates.data_ptr<float>());

    // ========================================================================
    // EXTRACT FINAL RESULTS
    // ========================================================================
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
  m.def("parallel_merge_candidate_scores_cpp", &parallel_merge_candidate_scores,
        "Merge Strided Candidate Scores");
  m.def("parallel_merge_candidate_scores_with_winners_cpp", &parallel_merge_candidate_scores_with_winners,
        "Merge Strided Candidate Scores with Phase 1 Winner Extraction for M3");
}