import os
import pathlib
import torch
import numpy as np

from warp.infra.config.config import ColBERTConfig
from warp.utils.tracker import NOPTracker
from warp.utils.utils import print_message
from warp.engine.constants import TPrimePolicy, T_PRIME_MAX

from torch.utils.cpp_extension import load

class ParallelIndexLoaderWARP:
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=True,
        load_index_with_mmap=False,
        fused_decompression_merge=True
    ):
        assert not use_gpu and not load_index_with_mmap

        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap
        self.fused_decompression_merge = fused_decompression_merge

        print_message(f"#> Loading buckets...")
        bucket_weights = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_weights.npy"))
        )
        self.bucket_weights = bucket_weights

        self._load_codec()
        print_message(f"#> Loading repacked residuals...")
        self.residuals_compacted = torch.load(
            os.path.join(self.index_path, "residuals.repacked.compacted.pt")
        )

    def _load_codec(self):
        print_message(f"#> Loading codec...")

        centroids = torch.from_numpy(
            np.load(os.path.join(self.index_path, "centroids.npy"))
        )
        sizes_compacted = torch.load(
            os.path.join(self.index_path, "sizes.compacted.pt")
        )
        codes_compacted = torch.load(
            os.path.join(self.index_path, "codes.compacted.pt")
        )

        residuals_compacted = torch.load(
            os.path.join(self.index_path, "residuals.compacted.pt")
        )

        num_centroids = centroids.shape[0]
        assert sizes_compacted.shape == (num_centroids,)

        num_embeddings = residuals_compacted.shape[0]
        assert sizes_compacted.sum() == num_embeddings
        assert codes_compacted.shape == (num_embeddings,)

        self.sizes_compacted = sizes_compacted
        self.codes_compacted = codes_compacted

        offsets_compacted = torch.zeros((num_centroids + 1,), dtype=torch.long)
        torch.cumsum(sizes_compacted, dim=0, out=offsets_compacted[1:])
        self.offsets_compacted = offsets_compacted

        self.kdummy_centroid = sizes_compacted.argmin().item()

        self.centroids = centroids
        print("#> Not averaging centroids.")

        return residuals_compacted


class ParallelIndexScorerWARP(ParallelIndexLoaderWARP):
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=False,
        load_index_with_mmap=False,
        t_prime=None,
        bound=128,
        fused_decompression_merge=True,
        centroid_only=False,
    ):
        assert not use_gpu
        assert not load_index_with_mmap

        super().__init__(
            index_path=index_path,
            config=config,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap,
            fused_decompression_merge=fused_decompression_merge
        )

        num_threads = torch.get_num_threads()
        assert num_threads != 1

        ParallelIndexScorerWARP.try_load_torch_extensions(use_gpu)

        assert config.ncells is not None
        self.nprobe = config.ncells

        (num_centroids, _) = self.centroids.shape
        if t_prime is not None:
            self.t_prime = TPrimePolicy(value=t_prime)
        elif num_centroids <= 2**16:
            (num_embeddings, _) = self.residuals_compacted.shape
            self.t_prime = TPrimePolicy(value=int(np.sqrt(8 * num_embeddings) / 1000) * 1000)
        else: self.t_prime = T_PRIME_MAX

        assert config.nbits in [2, 4]
        self.nbits = config.nbits

        self.centroid_only = centroid_only

        self.centroid_idx = torch.stack(tuple([
            torch.arange(num_centroids, dtype=torch.int32) for _ in range(num_threads)
        ])).contiguous()

        print("nprobe", self.nprobe, "t_prime", self.t_prime, "nbits", config.nbits)
        self.bound = bound or 128

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return
        cflags = [
            "-O3", "-mavx2", "-mfma", "-march=native", "-ffast-math", "-fno-math-errno", "-m64", "-fopenmp", "-std=c++17",
            "-funroll-loops", "-msse", "-msse2", "-msse3", "-msse4.1", "-mbmi2", "-mmmx", "-mavx", "-fomit-frame-pointer",
            "-fno-strict-aliasing"
        ]

        print_message(
            f"Loading parallel_warp_select_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.warp_select_centroids_cpp = load(
            name="parallel_warp_select_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_warp_select_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).parallel_warp_select_centroids_cpp

        print_message(
            f"Loading parallel_decompress_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.decompress_centroids_cpp = dict()
        cls.decompress_centroids_with_winners_cpp = dict()  # NEW: winner tracking variants
        decompress_centroids_cpp = load(
            name="parallel_decompress_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_decompress_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_centroids_cpp[2] = decompress_centroids_cpp.parallel_decompress_centroids_2_cpp
        cls.decompress_centroids_cpp[4] = decompress_centroids_cpp.parallel_decompress_centroids_4_cpp
        cls.decompress_centroids_with_winners_cpp[2] = decompress_centroids_cpp.parallel_decompress_centroids_with_winners_2_cpp
        cls.decompress_centroids_with_winners_cpp[4] = decompress_centroids_cpp.parallel_decompress_centroids_with_winners_4_cpp

        print_message(
            f"Loading parallel_merge_candidate_scores_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        merge_candidate_scores_module = load(
            name="parallel_merge_candidate_scores_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_merge_candidate_scores.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.merge_candidate_scores_cpp = merge_candidate_scores_module.parallel_merge_candidate_scores_cpp
        cls.merge_candidate_scores_with_winners_cpp = merge_candidate_scores_module.parallel_merge_candidate_scores_with_winners_cpp

        print_message(
            f"Loading parallel_fused_decompress_merge extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.fused_decompress_merge_cpp = dict()
        fused_decompress_merge_cpp = load(
            name="parallel_fused_decompress_merge_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_fused_decompress_merge.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.fused_decompress_merge_cpp[2] = fused_decompress_merge_cpp.parallel_fused_decompress_merge_2_cpp
        cls.fused_decompress_merge_cpp[4] = fused_decompress_merge_cpp.parallel_fused_decompress_merge_4_cpp

        cls.loaded_extensions = True

    def rank(self, config, Q, k=100, filter_fn=None, pids=None, tracker=NOPTracker()):
        assert filter_fn is None
        assert pids is None

        with torch.inference_mode():
            tracker.begin("Candidate Generation")
            centroid_scores = Q.squeeze(0) @ self.centroids.T
            tracker.end("Candidate Generation")

            tracker.begin("top-k Precompute")
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            query_tokens = Q_mask.sum().item()
            tracker.record("query_length", query_tokens)
            cells, centroid_scores, mse_estimates = self._warp_select_centroids(
                Q_mask, centroid_scores, self.nprobe, self.t_prime[k]
            )
            # Count unique centroids actually selected
            n_clusters_selected = torch.unique(cells[cells != self.kdummy_centroid]).numel()
            tracker.record("n_clusters_selected", n_clusters_selected)
            tracker.end("top-k Precompute")

            num_tokens = Q_mask.sum().item()
            
            # M3 Measurement path: force non-fused mode for winner tracking
            if tracker.measurements_enabled:
                tracker.begin("Decompression")
                capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners = self._decompress_centroids_with_winners(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe, num_tokens
                )
                total_token_scores = capacities.sum().item()
                tracker.record("total_token_scores", total_token_scores)
                unique_docs = torch.unique(candidate_pids[candidate_pids >= 0]).numel()
                tracker.record("unique_docs", unique_docs)
                tracker.end("Decompression")
                
                tracker.begin("Build Matrix")
                result = self._merge_candidate_scores_with_winners(
                    capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners,
                    mse_estimates, k, num_tokens
                )
                pids, scores, influential_counts, unique_docs_final = result[0], result[1], result[2], result[3]
                phase1_pids, phase1_scores, phase1_winners, phase1_sizes = result[4], result[5], result[6], result[7]
                tracker.end("Build Matrix")
                
                # Record M3 winners for top-k documents
                self._record_m3_winners(tracker, pids, phase1_pids, phase1_scores, phase1_winners, phase1_sizes, num_tokens)
            
            elif self.fused_decompression_merge:
                tracker.begin("Decompression")
                tracker.end("Decompression")

                tracker.begin("Build Matrix")
                pids, scores, unique_docs, total_token_scores = self._fused_decompress_merge_scores(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe, num_tokens, mse_estimates, k
                )
                tracker.record("total_token_scores", total_token_scores)
                tracker.record("unique_docs", unique_docs)
                tracker.end("Build Matrix")
            else:
                tracker.begin("Decompression")
                capacities, candidate_sizes, candidate_pids, candidate_scores = self._decompress_centroids(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe, num_tokens
                )
                # Total token-document similarity evaluations
                total_token_scores = capacities.sum().item()
                tracker.record("total_token_scores", total_token_scores)
                # Count unique documents before merge/truncation
                unique_docs = torch.unique(candidate_pids[candidate_pids >= 0]).numel()
                tracker.record("unique_docs", unique_docs)
                tracker.end("Decompression")
                tracker.begin("Build Matrix")
                pids, scores = self._merge_candidate_scores(
                    capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k, num_tokens
                )
                tracker.end("Build Matrix")

            # M1 Measurement: Record token-level computation per centroid
            # Uses helper function to avoid code duplication between fused/non-fused paths
            # See MEASUREMENT_WISHES.MD and M1_INTEGRATION_PLAN.md
            self._record_m1_measurements(tracker, cells, num_tokens)

            return pids, scores

    def _warp_select_centroids(self, Q_mask, centroid_scores, nprobe, t_prime):
        cells, scores, mse = ParallelIndexScorerWARP.warp_select_centroids_cpp(
            self.centroid_idx, Q_mask, centroid_scores, self.sizes_compacted, nprobe, t_prime, self.bound
        )

        cells = cells.flatten().contiguous()
        scores = scores.flatten().contiguous()

        # NOTE Skip decompression of cells with a zero score centroid.
        # This means that the corresponding query token was 0.0 (i.e., masked out). 
        cells[scores == 0] = self.kdummy_centroid

        return cells, scores, mse

    def _decompress_centroids(
        self, Q, centroid_ids, centroid_scores, nprobe, num_tokens
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores = ParallelIndexScorerWARP.decompress_centroids_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, num_tokens, self.centroid_only
        )
        return capacities, sizes, pids, scores

    def _merge_candidate_scores(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k, num_tokens
    ):
        pids, scores = ParallelIndexScorerWARP.merge_candidate_scores_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, self.nprobe, k, num_tokens
        )
        return pids.tolist(), scores.tolist()

    def _decompress_centroids_with_winners(
        self, Q, centroid_ids, centroid_scores, nprobe, num_tokens
    ):
        """Decompress centroids with winner position tracking for M3."""
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores, winners = ParallelIndexScorerWARP.decompress_centroids_with_winners_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, num_tokens, self.centroid_only
        )
        return capacities, sizes, pids, scores, winners

    def _merge_candidate_scores_with_winners(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners, mse_estimates, k, num_tokens
    ):
        """Merge candidate scores with Phase 1 winner extraction for M3."""
        result = ParallelIndexScorerWARP.merge_candidate_scores_with_winners_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners,
            mse_estimates, self.nprobe, k, num_tokens
        )
        # Returns: (pids, scores, influential_counts, unique_docs, 
        #           phase1_pids[32], phase1_scores[32], phase1_winners[32], phase1_sizes)
        pids, scores, influential_counts, unique_docs = result[0], result[1], result[2], result[3]
        phase1_pids, phase1_scores, phase1_winners, phase1_sizes = result[4], result[5], result[6], result[7]
        return (pids.tolist(), scores.tolist(), influential_counts, unique_docs,
                phase1_pids, phase1_scores, phase1_winners, phase1_sizes)

    def _fused_decompress_merge_scores(
        self, Q, centroid_ids, centroid_scores, nprobe, num_tokens, mse_estimates, k
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        total_token_scores = capacities.sum().item()
        pids, scores, unique_docs = ParallelIndexScorerWARP.fused_decompress_merge_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, num_tokens,
            mse_estimates, k, self.centroid_only
        )
        return pids.tolist(), scores.tolist(), unique_docs, total_token_scores

    def _record_m1_measurements(self, tracker, cells, num_tokens):
        """
        Record M1 measurements for all query tokens.
        
        This helper computes capacities from offsets_compacted and records M1
        metrics for each (query_token, centroid) pair. Thread-safe via the
        MeasurementCollector's internal lock.
        
        Args:
            tracker: ExecutionTracker with measurements_enabled
            cells: Tensor of selected centroid IDs [32 * nprobe]
            num_tokens: Number of actual query tokens (from Q_mask.sum())
        
        See MEASUREMENT_WISHES.MD for M1 specification.
        """
        if not tracker.measurements_enabled:
            return
        
        # Compute capacities from offsets (same as in _fused_decompress_merge_scores)
        centroid_ids = cells.long()
        capacities = self.offsets_compacted[centroid_ids + 1] - self.offsets_compacted[centroid_ids]
        
        # Record M1 for each (query_token, centroid) pair
        for t in range(num_tokens):
            for p in range(self.nprobe):
                idx = t * self.nprobe + p
                centroid_id = cells[idx].item()
                num_sims = capacities[idx].item()
                # Skip dummy centroids (masked out tokens) and empty centroids
                if centroid_id != self.kdummy_centroid and num_sims > 0:
                    tracker.record_m1(
                        q_token_id=t,
                        centroid_id=centroid_id,
                        num_token_token_sims=num_sims
                    )

    def _record_m3_winners(self, tracker, top_k_pids, phase1_pids, phase1_scores, phase1_winners, phase1_sizes, num_tokens):
        """
        Record M3 winners for top-k documents.
        
        For each top-k document, look up its winner in each token's Phase 1 stride
        using binary search. Records (q_token_id, doc_id, winner_embedding_pos, winner_score).
        
        Args:
            tracker: ExecutionTracker with measurements_enabled
            top_k_pids: List of top-k document IDs
            phase1_pids: List of 32 tensors, each containing doc IDs for that token
            phase1_scores: List of 32 tensors, each containing scores for that token
            phase1_winners: List of 32 tensors, each containing winner positions for that token
            phase1_sizes: Tensor of sizes [32]
            num_tokens: Number of actual query tokens
        
        See M3_INTEGRATION_PLAN.md for M3 Tier B specification.
        """
        if not tracker.measurements_enabled:
            return
        
        import numpy as np
        
        phase1_sizes_np = phase1_sizes.numpy()
        
        for doc_id in top_k_pids:
            for t in range(num_tokens):
                size = phase1_sizes_np[t]
                if size == 0:
                    continue
                
                # Binary search for doc_id in this token's sorted stride
                pids_np = phase1_pids[t].numpy()
                idx = np.searchsorted(pids_np, doc_id)
                
                if idx < size and pids_np[idx] == doc_id:
                    # Found: doc had observed evidence for this token
                    winner_pos = phase1_winners[t][idx].item()
                    winner_score = phase1_scores[t][idx].item()
                    
                    tracker.record_m3_winner(
                        q_token_id=t,
                        doc_id=doc_id,
                        winner_embedding_pos=winner_pos,
                        winner_score=winner_score
                    )
                # else: doc didn't have observed evidence for this token (MSE fallback)