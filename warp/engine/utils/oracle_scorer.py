"""
Oracle Scorer for M4 Measurement

This module provides a Python wrapper for the C++ parallel oracle scorer.
It prepares the CSR (Compressed Sparse Row) format required by the C++ code
and calls the parallel computation.

The oracle scorer computes what the MaxSim winner WOULD HAVE BEEN if all
document embeddings were considered (ignoring routing/nprobe constraints).

Three implementations available:
1. Original: Per-embedding centroid dot products (baseline)
2. Optimized: Precomputed centroid scores, 6-12x faster
3. Smart: Uses M3 observed data to skip redundant work (future optimization)

Usage:
    scorer = OracleScorer(
        reverse_index=reverse_index,
        centroids=centroids,
        residuals_compacted=residuals_compacted,
        bucket_weights=bucket_weights,
        offsets_compacted=offsets_compacted,
        nbits=4,
        oracle_cpp=ParallelIndexScorerWARP.oracle_scorer_cpp,
        use_optimized=True  # Use optimized version (default: True)
    )
    
    oracle_pos, oracle_scores = scorer.compute_oracle_batch(
        Q=Q,  # (num_tokens, 128)
        doc_ids=doc_ids  # list of doc IDs
    )
    # Returns: (num_docs, num_tokens) arrays

See M4_INTEGRATION_PLAN.md for full specification.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any


class OracleScorer:
    """
    Oracle scorer for M4 measurement.
    
    Computes oracle MaxSim winners for all (doc, token) pairs.
    Uses C++ parallel implementation for performance.
    """
    
    def __init__(
        self,
        reverse_index,
        centroids: torch.Tensor,
        residuals_compacted: torch.Tensor,
        bucket_weights: torch.Tensor,
        offsets_compacted: torch.Tensor,
        nbits: int,
        oracle_cpp: dict,
        use_optimized: bool = True
    ):
        """
        Initialize the oracle scorer.
        
        Args:
            reverse_index: ReverseIndex instance for doc_id -> embedding positions lookup
            centroids: (num_centroids, 128) centroid embeddings
            residuals_compacted: (num_embeddings, packed_dim) packed residuals
            bucket_weights: (128, num_buckets) bucket weight matrix for decompression
            offsets_compacted: (num_centroids + 1,) cumsum of centroid sizes
            nbits: 2 or 4
            oracle_cpp: Dict mapping function names -> C++ functions
            use_optimized: If True, use optimized version with precomputed centroid scores
        """
        self.reverse_index = reverse_index
        self.centroids = centroids
        self.residuals_compacted = residuals_compacted
        self.bucket_weights = bucket_weights
        self.offsets_compacted = offsets_compacted
        self.nbits = nbits
        self.use_optimized = use_optimized
        
        # Select the appropriate C++ function
        if use_optimized and f'optimized_{nbits}' in oracle_cpp:
            self._oracle_fn = oracle_cpp[f'optimized_{nbits}']
            self._mode = 'optimized'
        else:
            self._oracle_fn = oracle_cpp[nbits]
            self._mode = 'original'
        
        # Smart oracle function (if available)
        self._smart_fn = oracle_cpp.get(f'smart_{nbits}')
    
    @property
    def mode(self) -> str:
        """Return current oracle mode: 'original', 'optimized', or 'smart'"""
        return self._mode
    
    def compute_oracle_batch(
        self,
        Q: torch.Tensor,
        doc_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute oracle MaxSim winners for a batch of documents.
        
        Args:
            Q: (num_tokens, 128) query token embeddings
            doc_ids: List of document IDs to compute oracle for
        
        Returns:
            Tuple of:
                - oracle_pos: (num_docs, num_tokens) winning embedding positions
                - oracle_scores: (num_docs, num_tokens) winning scores
        """
        num_docs = len(doc_ids)
        num_tokens = Q.size(0)
        
        if num_docs == 0:
            return (
                torch.empty((0, num_tokens), dtype=torch.int64),
                torch.empty((0, num_tokens), dtype=torch.float32)
            )
        
        # Build CSR format for document positions
        all_positions, position_offsets = self._build_csr_positions(doc_ids)
        
        # Pre-allocate output tensors
        output_pos = torch.zeros((num_docs, num_tokens), dtype=torch.int64)
        output_scores = torch.zeros((num_docs, num_tokens), dtype=torch.float32)
        
        # Call C++ implementation (optimized or original)
        self._oracle_fn(
            Q.contiguous(),
            all_positions.contiguous(),
            position_offsets.contiguous(),
            self.centroids.contiguous(),
            self.residuals_compacted.contiguous(),
            self.bucket_weights.contiguous(),
            self.offsets_compacted.contiguous(),
            output_pos,
            output_scores
        )
        
        return output_pos, output_scores
    
    def compute_oracle_batch_smart(
        self,
        Q: torch.Tensor,
        doc_ids: List[int],
        observed_pos: torch.Tensor,
        observed_scores: torch.Tensor,
        score_margin: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SMART Oracle: Compute oracle with M3-informed skip logic.
        
        Uses observed winners from M3 to potentially skip work.
        Returns an additional tensor indicating which (doc, token) pairs
        were skipped because the observed winner was clearly optimal.
        
        Args:
            Q: (num_tokens, 128) query token embeddings
            doc_ids: List of document IDs to compute oracle for
            observed_pos: (num_docs, num_tokens) observed winner positions from M3
            observed_scores: (num_docs, num_tokens) observed winner scores from M3
            score_margin: Skip threshold - if observed >= max_other - margin, skip
        
        Returns:
            Tuple of:
                - oracle_pos: (num_docs, num_tokens) winning embedding positions
                - oracle_scores: (num_docs, num_tokens) winning scores
                - skipped: (num_docs, num_tokens) 1 if skipped, 0 if computed
        """
        if self._smart_fn is None:
            raise RuntimeError("Smart oracle C++ function not available")
        
        num_docs = len(doc_ids)
        num_tokens = Q.size(0)
        
        if num_docs == 0:
            return (
                torch.empty((0, num_tokens), dtype=torch.int64),
                torch.empty((0, num_tokens), dtype=torch.float32),
                torch.empty((0, num_tokens), dtype=torch.int64)
            )
        
        all_positions, position_offsets = self._build_csr_positions(doc_ids)
        
        output_pos = torch.zeros((num_docs, num_tokens), dtype=torch.int64)
        output_scores = torch.zeros((num_docs, num_tokens), dtype=torch.float32)
        output_skipped = torch.zeros((num_docs, num_tokens), dtype=torch.int64)
        
        self._smart_fn(
            Q.contiguous(),
            all_positions.contiguous(),
            position_offsets.contiguous(),
            self.centroids.contiguous(),
            self.residuals_compacted.contiguous(),
            self.bucket_weights.contiguous(),
            self.offsets_compacted.contiguous(),
            observed_pos.contiguous(),
            observed_scores.contiguous(),
            score_margin,
            output_pos,
            output_scores,
            output_skipped
        )
        
        return output_pos, output_scores, output_skipped
    
    def _build_csr_positions(
        self,
        doc_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build CSR (Compressed Sparse Row) format for document positions.
        
        Args:
            doc_ids: List of document IDs
        
        Returns:
            Tuple of:
                - all_positions: Flattened tensor of all embedding positions
                - position_offsets: (num_docs + 1,) CSR offsets
        """
        # Gather positions for each document
        all_pos_list = []
        offsets = [0]
        
        for doc_id in doc_ids:
            positions = self.reverse_index.get_embedding_positions(doc_id)
            if positions is not None:
                all_pos_list.append(positions)
                offsets.append(offsets[-1] + len(positions))
            else:
                # Document not found - should not happen if doc_ids come from search
                offsets.append(offsets[-1])
        
        # Concatenate all positions
        if all_pos_list:
            all_positions = torch.cat(all_pos_list)
        else:
            all_positions = torch.empty(0, dtype=torch.int64)
        
        position_offsets = torch.tensor(offsets, dtype=torch.int64)
        
        return all_positions, position_offsets


class OptimizedOracleScorer(OracleScorer):
    """
    Optimized Oracle Scorer - always uses the optimized C++ implementation.
    
    Provides 6-12x speedup over original by:
    1. Precomputing centroid scores (Q @ centroids.T) upfront
    2. Caching centroid IDs per document (binary search once per embedding)
    
    Backwards compatible with OracleScorer API.
    """
    
    def __init__(
        self,
        reverse_index,
        centroids: torch.Tensor,
        residuals_compacted: torch.Tensor,
        bucket_weights: torch.Tensor,
        offsets_compacted: torch.Tensor,
        nbits: int,
        oracle_cpp: dict
    ):
        """Initialize with optimized=True forced."""
        super().__init__(
            reverse_index=reverse_index,
            centroids=centroids,
            residuals_compacted=residuals_compacted,
            bucket_weights=bucket_weights,
            offsets_compacted=offsets_compacted,
            nbits=nbits,
            oracle_cpp=oracle_cpp,
            use_optimized=True
        )


def compute_m4_oracle_and_record(
    tracker,
    oracle_scorer: OracleScorer,
    Q: torch.Tensor,
    all_scored_docs: List[int],
    num_tokens: int
) -> None:
    """
    Compute M4 oracle winners and record them to tracker.
    
    Args:
        tracker: ExecutionTracker with measurements_enabled
        oracle_scorer: OracleScorer instance
        Q: (num_tokens, 128) query token embeddings
        all_scored_docs: List of document IDs (same as M3 scope)
        num_tokens: Number of actual query tokens
    """
    if not tracker.measurements_enabled:
        return
    
    if not all_scored_docs:
        return
    
    # Compute oracle for all docs at once
    oracle_pos, oracle_scores = oracle_scorer.compute_oracle_batch(
        Q[:num_tokens],  # Only use actual tokens
        all_scored_docs
    )
    
    # Record batch in single call (50-100x faster than per-token loop)
    tracker.record_m4_batch(
        query_id=tracker.current_query_id,
        doc_ids=all_scored_docs,
        oracle_pos=oracle_pos,
        oracle_scores=oracle_scores,
        num_tokens=num_tokens
    )


def make_oracle_cpp_dict(oracle_module) -> Dict[Any, Any]:
    """
    Create oracle_cpp dictionary from the compiled C++ module.
    
    This maps both legacy keys (2, 4) and new keys (optimized_2, etc.)
    to the appropriate C++ functions.
    
    Args:
        oracle_module: The compiled torch extension module
    
    Returns:
        Dict mapping keys to C++ functions
    """
    result = {}
    
    # Legacy keys for backwards compatibility
    if hasattr(oracle_module, 'compute_oracle_batch_2_cpp'):
        result[2] = oracle_module.compute_oracle_batch_2_cpp
    if hasattr(oracle_module, 'compute_oracle_batch_4_cpp'):
        result[4] = oracle_module.compute_oracle_batch_4_cpp
    
    # Optimized versions
    if hasattr(oracle_module, 'compute_oracle_batch_optimized_2_cpp'):
        result['optimized_2'] = oracle_module.compute_oracle_batch_optimized_2_cpp
    if hasattr(oracle_module, 'compute_oracle_batch_optimized_4_cpp'):
        result['optimized_4'] = oracle_module.compute_oracle_batch_optimized_4_cpp
    
    # Smart versions
    if hasattr(oracle_module, 'compute_oracle_batch_smart_2_cpp'):
        result['smart_2'] = oracle_module.compute_oracle_batch_smart_2_cpp
    if hasattr(oracle_module, 'compute_oracle_batch_smart_4_cpp'):
        result['smart_4'] = oracle_module.compute_oracle_batch_smart_4_cpp
    
    # WARP-style versions
    if hasattr(oracle_module, 'compute_oracle_batch_warp_2_cpp'):
        result['warp_2'] = oracle_module.compute_oracle_batch_warp_2_cpp
    if hasattr(oracle_module, 'compute_oracle_batch_warp_4_cpp'):
        result['warp_4'] = oracle_module.compute_oracle_batch_warp_4_cpp
    
    # WARP-style with tracking
    if hasattr(oracle_module, 'compute_oracle_batch_warp_tracking_2_cpp'):
        result['warp_tracking_2'] = oracle_module.compute_oracle_batch_warp_tracking_2_cpp
    if hasattr(oracle_module, 'compute_oracle_batch_warp_tracking_4_cpp'):
        result['warp_tracking_4'] = oracle_module.compute_oracle_batch_warp_tracking_4_cpp
    
    return result


class WarpStyleOracleScorer:
    """
    WARP-Style Oracle Scorer - Maximum optimization using precomputed centroid IDs.
    
    Key insight from WARP: The centroid ID for each embedding is FIXED at index time.
    Rather than binary searching offsets_compacted at query time, we precompute:
    
        embedding_to_centroid[pos] = centroid_id for ALL embeddings
    
    This trades ~36MB of memory for a 10-20x speedup.
    
    Building the embedding_to_centroid tensor (one-time, ~10 seconds):
    
        offsets = torch.load("sizes.compacted.pt")
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), offsets.cumsum(0)])
        
        # Method 1: Using searchsorted (simple)
        positions = torch.arange(num_embeddings)
        embedding_to_centroid = torch.searchsorted(offsets, positions, side='right') - 1
        
        # Method 2: Using repeat_interleave (faster for building)
        sizes = torch.load("sizes.compacted.pt")
        embedding_to_centroid = torch.repeat_interleave(
            torch.arange(len(sizes)), sizes
        ).to(torch.int32)
    
    Usage:
        scorer = WarpStyleOracleScorer(
            reverse_index=reverse_index,
            centroids=centroids,
            residuals_compacted=residuals_compacted,
            bucket_weights=bucket_weights,
            embedding_to_centroid=embedding_to_centroid,  # precomputed!
            nbits=4,
            oracle_cpp=ParallelIndexScorerWARP.oracle_scorer_cpp
        )
        
        oracle_pos, oracle_scores = scorer.compute_oracle_batch(Q, doc_ids)
    """
    
    def __init__(
        self,
        reverse_index,
        centroids: torch.Tensor,
        residuals_compacted: torch.Tensor,
        bucket_weights: torch.Tensor,
        embedding_to_centroid: torch.Tensor,
        nbits: int,
        oracle_cpp: dict,
        track_centroids: bool = False
    ):
        """
        Initialize the WARP-style oracle scorer.
        
        Args:
            reverse_index: ReverseIndex instance for doc_id -> embedding positions lookup
            centroids: (num_centroids, 128) centroid embeddings
            residuals_compacted: (num_embeddings, packed_dim) packed residuals
            bucket_weights: (128, num_buckets) bucket weight matrix
            embedding_to_centroid: (num_embeddings,) PRECOMPUTED centroid ID per embedding
            nbits: 2 or 4
            oracle_cpp: Dict mapping function names -> C++ functions
            track_centroids: If True, also return oracle winner centroid IDs
        """
        self.reverse_index = reverse_index
        self.centroids = centroids
        self.residuals_compacted = residuals_compacted
        self.bucket_weights = bucket_weights
        self.embedding_to_centroid = embedding_to_centroid.to(torch.int32).contiguous()
        self.nbits = nbits
        self.track_centroids = track_centroids
        
        # Select WARP-style C++ function
        if track_centroids:
            self._oracle_fn = oracle_cpp[f'warp_tracking_{nbits}']
        else:
            self._oracle_fn = oracle_cpp[f'warp_{nbits}']
    
    @staticmethod
    def build_embedding_to_centroid(index_path: str, verbose: bool = True) -> torch.Tensor:
        """
        Build the embedding_to_centroid lookup tensor.
        
        This is a one-time operation that takes ~10 seconds for 9M embeddings.
        The result can be saved and reloaded for subsequent runs.
        
        Args:
            index_path: Path to WARP index directory
            verbose: Whether to print progress
        
        Returns:
            Tensor of shape (num_embeddings,) with dtype int32
        """
        import os
        
        sizes_path = os.path.join(index_path, "sizes.compacted.pt")
        sizes = torch.load(sizes_path)
        
        if verbose:
            print(f"Building embedding_to_centroid from {sizes_path}...")
            print(f"  {len(sizes):,} centroids, {sizes.sum().item():,} embeddings")
        
        # Use repeat_interleave: each centroid ID is repeated by its size
        embedding_to_centroid = torch.repeat_interleave(
            torch.arange(len(sizes), dtype=torch.int32),
            sizes.to(torch.int64)
        )
        
        if verbose:
            print(f"  Built tensor of shape {embedding_to_centroid.shape}")
            print(f"  Memory: {embedding_to_centroid.numel() * 4 / 1e6:.1f} MB")
        
        return embedding_to_centroid
    
    @staticmethod
    def load_or_build_embedding_to_centroid(
        index_path: str, 
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Load embedding_to_centroid if it exists, otherwise build and save it.
        """
        import os
        
        cache_path = os.path.join(index_path, "embedding_to_centroid.pt")
        
        if os.path.exists(cache_path):
            if verbose:
                print(f"Loading embedding_to_centroid from {cache_path}...")
            return torch.load(cache_path)
        
        if verbose:
            print(f"embedding_to_centroid not found, building...")
        
        tensor = WarpStyleOracleScorer.build_embedding_to_centroid(index_path, verbose)
        
        if verbose:
            print(f"Saving to {cache_path}...")
        torch.save(tensor, cache_path)
        
        return tensor
    
    def compute_oracle_batch(
        self,
        Q: torch.Tensor,
        doc_ids: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute oracle MaxSim winners for a batch of documents.
        
        Returns:
            If track_centroids=False:
                Tuple of (oracle_pos, oracle_scores)
            If track_centroids=True:
                Tuple of (oracle_pos, oracle_scores, oracle_centroid_ids)
        """
        num_docs = len(doc_ids)
        num_tokens = Q.size(0)
        
        if num_docs == 0:
            empty_pos = torch.empty((0, num_tokens), dtype=torch.int64)
            empty_scores = torch.empty((0, num_tokens), dtype=torch.float32)
            if self.track_centroids:
                empty_cids = torch.empty((0, num_tokens), dtype=torch.int32)
                return empty_pos, empty_scores, empty_cids
            return empty_pos, empty_scores
        
        # Build CSR format for document positions
        all_positions, position_offsets = self._build_csr_positions(doc_ids)
        
        # Pre-allocate output tensors
        output_pos = torch.zeros((num_docs, num_tokens), dtype=torch.int64)
        output_scores = torch.zeros((num_docs, num_tokens), dtype=torch.float32)
        
        if self.track_centroids:
            output_cids = torch.zeros((num_docs, num_tokens), dtype=torch.int32)
            
            self._oracle_fn(
                Q.contiguous(),
                all_positions.contiguous(),
                position_offsets.contiguous(),
                self.centroids.contiguous(),
                self.residuals_compacted.contiguous(),
                self.bucket_weights.contiguous(),
                self.embedding_to_centroid,
                output_pos,
                output_scores,
                output_cids
            )
            
            return output_pos, output_scores, output_cids
        else:
            self._oracle_fn(
                Q.contiguous(),
                all_positions.contiguous(),
                position_offsets.contiguous(),
                self.centroids.contiguous(),
                self.residuals_compacted.contiguous(),
                self.bucket_weights.contiguous(),
                self.embedding_to_centroid,
                output_pos,
                output_scores
            )
            
            return output_pos, output_scores
    
    def _build_csr_positions(
        self,
        doc_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build CSR format for document positions (same as OracleScorer)."""
        all_pos_list = []
        offsets = [0]
        
        for doc_id in doc_ids:
            positions = self.reverse_index.get_embedding_positions(doc_id)
            if positions is not None:
                all_pos_list.append(positions)
                offsets.append(offsets[-1] + len(positions))
            else:
                offsets.append(offsets[-1])
        
        if all_pos_list:
            all_positions = torch.cat(all_pos_list)
        else:
            all_positions = torch.empty(0, dtype=torch.int64)
        
        position_offsets = torch.tensor(offsets, dtype=torch.int64)
        
        return all_positions, position_offsets
