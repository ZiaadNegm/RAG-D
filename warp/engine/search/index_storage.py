import os
import pathlib
import torch
import numpy as np

from warp.infra.config.config import ColBERTConfig
from warp.utils.tracker import NOPTracker
from warp.utils.utils import print_message
from warp.engine.constants import TPrimePolicy, T_PRIME_MAX

from torch.utils.cpp_extension import load

class IndexLoaderWARP:
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=True,
        load_index_with_mmap=False,
    ):
        assert not use_gpu and not load_index_with_mmap

        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap

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


class IndexScorerWARP(IndexLoaderWARP):
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=False,
        load_index_with_mmap=False,
        t_prime=None,
        bound=128,
        centroid_only=False,
    ):
        assert not use_gpu
        assert not load_index_with_mmap

        super().__init__(
            index_path=index_path,
            config=config,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap,
        )

        IndexScorerWARP.try_load_torch_extensions(use_gpu)

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

        self.bound = bound or 128
        self.centroid_only = centroid_only

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
            f"Loading warp_select_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.warp_select_centroids_cpp = load(
            name="warp_select_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "warp_select_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).warp_select_centroids_cpp

        print_message(
            f"Loading decompress_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.decompress_centroids_cpp = dict()
        cls.decompress_centroids_with_winners_cpp = dict()  # NEW: winner tracking variants
        decompress_centroids_cpp = load(
            name="decompress_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "decompress_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_centroids_cpp[2] = decompress_centroids_cpp.decompress_centroids_2_cpp
        cls.decompress_centroids_cpp[4] = decompress_centroids_cpp.decompress_centroids_4_cpp
        cls.decompress_centroids_with_winners_cpp[2] = decompress_centroids_cpp.decompress_centroids_with_winners_2_cpp
        cls.decompress_centroids_with_winners_cpp[4] = decompress_centroids_cpp.decompress_centroids_with_winners_4_cpp

        print_message(
            f"Loading merge_candidate_scores_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        merge_candidate_scores_module = load(
            name="merge_candidate_scores_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "merge_candidate_scores.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.merge_candidate_scores_cpp = merge_candidate_scores_module.merge_candidate_scores_cpp
        cls.merge_candidate_scores_with_winners_cpp = merge_candidate_scores_module.merge_candidate_scores_with_winners_cpp

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
            # non_zero_centroid_scores = centroid_scores[centroid_scores != 0]
            # tracker.record("centroid_scores", non_zero_centroid_scores)
            # Count non-zero cells (total centroid lookups, including duplicates)
            # Count unique centroids actually selected
            n_clusters_selected = torch.unique(cells[cells != 0]).numel()
            tracker.record("n_clusters_selected", n_clusters_selected)
            tracker.end("top-k Precompute")

            # Decide whether to use winner-tracking path for M3 measurements
            track_m3_winners = tracker.measurements_enabled and hasattr(tracker, '_m3_tracking_enabled') and tracker._m3_tracking_enabled
            
            tracker.begin("Decompression")
            if track_m3_winners:
                # Use winner-tracking decompression for M3 Tier B
                capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners = self._decompress_centroids_with_winners(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe
                )
            else:
                # Standard decompression
                capacities, candidate_sizes, candidate_pids, candidate_scores = self._decompress_centroids(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe
                )
                candidate_winners = None
            
            # Total token-document similarity evaluations (inner loop iterations in decompress)
            total_token_scores = capacities.sum().item()
            tracker.record("total_token_scores", total_token_scores)
            tracker.end("Decompression")

            # M1 Measurement: Record token-level computation per centroid
            # See MEASUREMENT_WISHES.MD and M1_INTEGRATION_PLAN.md
            if tracker.measurements_enabled:
                for t in range(query_tokens):
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

            tracker.begin("Build Matrix")
            if track_m3_winners:
                # Use winner-tracking merge for M3 Tier B
                pids, scores, influential_counts, unique_docs, phase1_pids, phase1_scores, phase1_winners, phase1_sizes = self._merge_candidate_scores_with_winners(
                    capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners, mse_estimates, k
                )
                
                # Record M3 Tier B: per-token winners for top-k documents
                # Uses binary search to look up winners efficiently
                self._record_m3_winners(
                    tracker=tracker,
                    top_k_pids=pids,
                    phase1_pids=phase1_pids,
                    phase1_scores=phase1_scores, 
                    phase1_winners=phase1_winners,
                    phase1_sizes=phase1_sizes,
                    query_tokens=query_tokens
                )
            else:
                pids, scores, influential_counts, unique_docs = self._merge_candidate_scores(
                    capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
                )
            
            tracker.record("unique_docs", unique_docs)
            tracker.record("influential_counts", influential_counts)
            tracker.end("Build Matrix")

            return pids, scores

    def _warp_select_centroids(self, Q_mask, centroid_scores, nprobe, t_prime):
        cells, scores, mse = IndexScorerWARP.warp_select_centroids_cpp(
            Q_mask, centroid_scores, self.sizes_compacted, nprobe, t_prime, self.bound
        )

        cells = cells.flatten().contiguous()
        scores = scores.flatten().contiguous()

        # NOTE Skip decompression of cells with a zero score centroid.
        # This means that the corresponding query token was 0.0 (i.e., masked out). 
        cells[scores == 0] = self.kdummy_centroid

        return cells, scores, mse

    def _decompress_centroids(
        self, Q, centroid_ids, centroid_scores, nprobe
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores = IndexScorerWARP.decompress_centroids_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, self.centroid_only
        )
        return capacities, sizes, pids, scores
    
    def _decompress_centroids_with_winners(
        self, Q, centroid_ids, centroid_scores, nprobe
    ):
        """Decompress with winner position tracking for M3 Tier B measurement."""
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores, winners = IndexScorerWARP.decompress_centroids_with_winners_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, self.centroid_only
        )
        return capacities, sizes, pids, scores, winners

    def _merge_candidate_scores(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
    ):
        pids, scores, influential_counts, unique_docs = IndexScorerWARP.merge_candidate_scores_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, self.nprobe, k
        )
        return pids.tolist(), scores.tolist(), influential_counts.tolist(), unique_docs
    
    def _merge_candidate_scores_with_winners(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners, mse_estimates, k
    ):
        """Merge with Phase 1 winner extraction for M3 Tier B measurement."""
        result = IndexScorerWARP.merge_candidate_scores_with_winners_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, candidate_winners, 
            mse_estimates, self.nprobe, k
        )
        pids, scores, influential_counts, unique_docs = result[0], result[1], result[2], result[3]
        phase1_pids, phase1_scores, phase1_winners, phase1_sizes = result[4], result[5], result[6], result[7]
        
        return (pids.tolist(), scores.tolist(), influential_counts.tolist(), unique_docs,
                phase1_pids, phase1_scores, phase1_winners, phase1_sizes)
    
    def _record_m3_winners(
        self, tracker, top_k_pids, phase1_pids, phase1_scores, phase1_winners, phase1_sizes, query_tokens
    ):
        """
        Record M3 Tier B measurements: per-token winners for top-k documents.
        
        For each (query_token, doc) pair where doc is in top-k:
        - Look up the winner in that token's Phase 1 stride via binary search
        - Record (doc_id, winner_embedding_pos, winner_score)
        
        Args:
            tracker: ExecutionTracker with measurement collector
            top_k_pids: List of top-k document IDs (already sorted by score)
            phase1_pids: List of 32 tensors, each containing doc_ids for that token
            phase1_scores: List of 32 tensors, each containing scores for that token  
            phase1_winners: List of 32 tensors, each containing winner positions for that token
            phase1_sizes: Tensor of 32 integers, number of valid entries per token
            query_tokens: Number of actual query tokens (not masked out)
        """
        if not hasattr(tracker, 'record_m3_winner'):
            return
            
        # Convert top_k_pids to set for fast lookup
        top_k_set = set(top_k_pids)
        
        for t in range(query_tokens):
            size = phase1_sizes[t].item()
            if size == 0:
                continue
                
            # Get Phase 1 data for this token
            pids_tensor = phase1_pids[t][:size]
            scores_tensor = phase1_scores[t][:size]
            winners_tensor = phase1_winners[t][:size]
            
            # Phase 1 strides are sorted by doc_id, so we can use binary search
            # But it's simpler to iterate since size is typically ~1000-10000
            pids_np = pids_tensor.numpy()
            scores_np = scores_tensor.numpy()
            winners_np = winners_tensor.numpy()
            
            for i in range(size):
                doc_id = int(pids_np[i])
                if doc_id in top_k_set:
                    tracker.record_m3_winner(
                        q_token_id=t,
                        doc_id=doc_id,
                        winner_embedding_pos=int(winners_np[i]),
                        winner_score=float(scores_np[i])
                    )
