"""
Oracle Scorer for M4 Measurement

This module provides a Python wrapper for the C++ parallel oracle scorer.
It prepares the CSR (Compressed Sparse Row) format required by the C++ code
and calls the parallel computation.

The oracle scorer computes what the MaxSim winner WOULD HAVE BEEN if all
document embeddings were considered (ignoring routing/nprobe constraints).

Usage:
    scorer = OracleScorer(
        reverse_index=reverse_index,
        centroids=centroids,
        residuals_compacted=residuals_compacted,
        bucket_weights=bucket_weights,
        offsets_compacted=offsets_compacted,
        nbits=4,
        oracle_cpp=ParallelIndexScorerWARP.oracle_scorer_cpp
    )
    
    oracle_pos, oracle_scores = scorer.compute_oracle_batch(
        Q=Q,  # (num_tokens, 128)
        doc_ids=doc_ids  # list of doc IDs
    )
    # Returns: (num_docs, num_tokens) arrays

See M4_INTEGRATION_PLAN.md for full specification.
"""

import torch
from typing import List, Tuple, Optional


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
        oracle_cpp: dict
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
            oracle_cpp: Dict mapping nbits -> C++ function
        """
        self.reverse_index = reverse_index
        self.centroids = centroids
        self.residuals_compacted = residuals_compacted
        self.bucket_weights = bucket_weights
        self.offsets_compacted = offsets_compacted
        self.nbits = nbits
        self._oracle_fn = oracle_cpp[nbits]
    
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
        
        # Call C++ parallel implementation
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
