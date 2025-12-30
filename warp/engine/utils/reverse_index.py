"""
Reverse Index for WARP: Document ID → Embedding Positions

This module provides a reverse lookup from document IDs to their embedding positions
in the WARP index. While WARP stores embeddings grouped by centroid (for fast search),
some operations like oracle computation (M4) need to access ALL embeddings for a 
specific document.

## The Problem

WARP's index layout is centroid-sorted:

    Position:     0    1    2    3    4    5    6    7    8    9   10
                  ├────────────────┼────────────────────────┼─────────┤
    Centroid:     │   Centroid 0   │      Centroid 1        │ Cent 2  │
    codes:        │  7    7   12  12│  3    3   55  55  99  │  7  101 │

    Document 7's embeddings are scattered: positions [0, 1, 9]

## The Solution: Reverse Index

We build a CSR-like structure (same pattern WARP uses for centroids):

    doc_id → embedding positions

    reverse_sorted_indices:  [all positions sorted by their doc_id]
    reverse_doc_offsets:     [cumsum of doc counts, for O(1) lookup]

## Storage Format

    reverse_sorted_indices.pt  - torch.int64, shape (num_embeddings,)
    reverse_doc_offsets.pt     - torch.int64, shape (num_docs + 1,)

## Usage Example

    from warp.engine.utils.reverse_index import ReverseIndex
    
    # Build and save (one-time)
    reverse_idx = ReverseIndex.build(index_path)
    reverse_idx.save(index_path)
    
    # Load (subsequent uses)
    reverse_idx = ReverseIndex.load(index_path)
    
    # Lookup embeddings for a document
    positions = reverse_idx.get_embedding_positions(doc_id=42)
    residuals_for_doc = residuals_compacted[positions]

"""

import os
import torch
from typing import Optional, Union, List


class ReverseIndex:
    """
    Reverse index mapping document IDs to their embedding positions in the WARP index.
    
    This enables O(1) lookup of all embeddings belonging to a document, which is
    required for oracle computation (M4 metric) where we need to compute MaxSim
    over ALL document tokens, not just those exposed by routing.
    
    Attributes:
        sorted_indices: Embedding positions sorted by document ID.
        doc_offsets: Cumulative sum of document embedding counts.
                     doc_offsets[i] is the start index for doc i in sorted_indices.
        num_docs: Total number of documents.
        num_embeddings: Total number of embeddings.
    
    Example:
        Consider a tiny index with 11 embeddings across 5 documents:
        
        codes_compacted = [7, 7, 12, 12, 3, 3, 55, 55, 99, 7, 101]
                          position: 0  1   2   3  4  5   6   7   8  9   10
        
        After building the reverse index:
        
        sorted_indices = [4, 5,  0, 1, 9,  2, 3,  6, 7,  8,   10]
                         └─doc 3─┘ └─doc 7─┘ └doc12┘ └doc55┘ └99┘ └101┘
        
        doc_offsets[3] = 0, doc_offsets[4] = 2   → doc 3 at sorted_indices[0:2]
        doc_offsets[7] = 2, doc_offsets[8] = 5   → doc 7 at sorted_indices[2:5]
        
        Lookup: get_embedding_positions(7) → tensor([0, 1, 9])
    """
    
    # Filenames for persistence
    SORTED_INDICES_FILE = "reverse_sorted_indices.pt"
    DOC_OFFSETS_FILE = "reverse_doc_offsets.pt"
    
    def __init__(
        self,
        sorted_indices: torch.Tensor,
        doc_offsets: torch.Tensor,
    ):
        """
        Initialize a ReverseIndex from pre-computed tensors.
        
        Args:
            sorted_indices: Embedding positions sorted by document ID.
                           Shape: (num_embeddings,), dtype: torch.int64
            doc_offsets: Cumulative document counts for O(1) lookup.
                        Shape: (num_docs + 1,), dtype: torch.int64
        """
        self.sorted_indices = sorted_indices
        self.doc_offsets = doc_offsets
        self.num_embeddings = sorted_indices.shape[0]
        self.num_docs = doc_offsets.shape[0] - 1
    
    @classmethod
    def build(cls, index_path: str, verbose: bool = True) -> "ReverseIndex":
        """
        Build a reverse index from a WARP index.
        
        This reads codes_compacted.pt (which contains doc_id for each embedding)
        and builds the reverse lookup structure.
        
        Args:
            index_path: Path to the WARP index directory.
            verbose: Whether to print progress information.
        
        Returns:
            A new ReverseIndex instance.
        
        Algorithm:
            1. Load codes_compacted (doc_id per embedding)
            2. Sort embedding positions by their doc_id → sorted_indices
            3. Count embeddings per doc → doc_counts
            4. Cumsum to get offsets → doc_offsets
        
        Time complexity: O(N log N) for sorting, where N = num_embeddings
        Space complexity: O(N) for sorted_indices + O(D) for doc_offsets
        """
        if verbose:
            print(f"Building reverse index from '{index_path}'...")
        
        # Load document IDs for each embedding
        codes_path = os.path.join(index_path, "codes.compacted.pt")
        codes_compacted = torch.load(codes_path)
        
        num_embeddings = codes_compacted.shape[0]
        num_docs = codes_compacted.max().item() + 1
        
        if verbose:
            print(f"  Embeddings: {num_embeddings:,}")
            print(f"  Documents:  {num_docs:,}")
        
        # Step 1: Sort positions by doc_id
        # argsort gives us indices that would sort codes_compacted
        # These indices ARE the embedding positions, now ordered by doc_id
        sorted_indices = torch.argsort(codes_compacted).to(torch.int64)
        
        # Step 2: Count embeddings per document
        doc_counts = torch.bincount(codes_compacted, minlength=num_docs)
        
        # Step 3: Cumulative sum for O(1) offset lookup
        # doc_offsets[i] = sum of counts for docs 0..i-1
        # doc_offsets[i+1] - doc_offsets[i] = count for doc i
        doc_offsets = torch.zeros(num_docs + 1, dtype=torch.int64)
        torch.cumsum(doc_counts, dim=0, out=doc_offsets[1:])
        
        if verbose:
            avg_embeddings = num_embeddings / num_docs
            print(f"  Avg embeddings/doc: {avg_embeddings:.1f}")
            print(f"  Memory: sorted_indices={sorted_indices.numel() * 8 / 1e6:.1f}MB, "
                  f"doc_offsets={doc_offsets.numel() * 8 / 1e6:.1f}MB")
        
        return cls(sorted_indices, doc_offsets)
    
    @classmethod
    def load(cls, index_path: str) -> "ReverseIndex":
        """
        Load a previously saved reverse index.
        
        Args:
            index_path: Path to the WARP index directory containing
                       reverse_sorted_indices.pt and reverse_doc_offsets.pt.
        
        Returns:
            A ReverseIndex instance.
        
        Raises:
            FileNotFoundError: If reverse index files don't exist.
        """
        sorted_indices_path = os.path.join(index_path, cls.SORTED_INDICES_FILE)
        doc_offsets_path = os.path.join(index_path, cls.DOC_OFFSETS_FILE)
        
        if not os.path.exists(sorted_indices_path):
            raise FileNotFoundError(
                f"Reverse index not found at '{index_path}'. "
                f"Build it first with ReverseIndex.build()"
            )
        
        sorted_indices = torch.load(sorted_indices_path)
        doc_offsets = torch.load(doc_offsets_path)
        
        return cls(sorted_indices, doc_offsets)
    
    @classmethod
    def load_or_build(cls, index_path: str, verbose: bool = True) -> "ReverseIndex":
        """
        Load reverse index if it exists, otherwise build and save it.
        
        Args:
            index_path: Path to the WARP index directory.
            verbose: Whether to print progress information.
        
        Returns:
            A ReverseIndex instance.
        """
        try:
            if verbose:
                print(f"Attempting to load reverse index from '{index_path}'...")
            return cls.load(index_path)
        except FileNotFoundError:
            if verbose:
                print("Reverse index not found, building...")
            reverse_idx = cls.build(index_path, verbose=verbose)
            reverse_idx.save(index_path, verbose=verbose)
            return reverse_idx
    
    def save(self, index_path: str, verbose: bool = True) -> None:
        """
        Save the reverse index to disk.
        
        Args:
            index_path: Path to the WARP index directory.
            verbose: Whether to print progress information.
        """
        sorted_indices_path = os.path.join(index_path, self.SORTED_INDICES_FILE)
        doc_offsets_path = os.path.join(index_path, self.DOC_OFFSETS_FILE)
        
        torch.save(self.sorted_indices, sorted_indices_path)
        torch.save(self.doc_offsets, doc_offsets_path)
        
        if verbose:
            print(f"Saved reverse index to '{index_path}':")
            print(f"  {self.SORTED_INDICES_FILE}: {os.path.getsize(sorted_indices_path) / 1e6:.1f}MB")
            print(f"  {self.DOC_OFFSETS_FILE}: {os.path.getsize(doc_offsets_path) / 1e6:.1f}MB")
    
    def get_embedding_positions(self, doc_id: int) -> torch.Tensor:
        """
        Get all embedding positions for a document.
        
        Args:
            doc_id: The document ID to look up.
        
        Returns:
            A tensor of embedding positions (indices into codes_compacted,
            residuals_compacted, etc.).
        
        Time complexity: O(1) for the lookup, O(k) to slice where k = num embeddings for doc
        
        Example:
            positions = reverse_idx.get_embedding_positions(42)
            # positions might be tensor([1205, 8923, 15002, ...])
            
            # Use these to get the actual embeddings:
            doc_residuals = residuals_compacted[positions]
        """
        begin = self.doc_offsets[doc_id].item()
        end = self.doc_offsets[doc_id + 1].item()
        return self.sorted_indices[begin:end]
    
    def get_embedding_positions_batch(
        self, 
        doc_ids: Union[List[int], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get embedding positions for multiple documents.
        
        Args:
            doc_ids: List or tensor of document IDs.
        
        Returns:
            A tuple of (positions, doc_boundaries):
            - positions: Concatenated embedding positions for all docs
            - doc_boundaries: Start index in positions for each doc
                             (length = len(doc_ids) + 1)
        
        Example:
            positions, boundaries = reverse_idx.get_embedding_positions_batch([7, 12, 55])
            
            # Get positions for doc 7:
            doc_7_positions = positions[boundaries[0]:boundaries[1]]
            
            # Get positions for doc 12:
            doc_12_positions = positions[boundaries[1]:boundaries[2]]
        """
        if isinstance(doc_ids, torch.Tensor):
            doc_ids = doc_ids.tolist()
        
        all_positions = []
        boundaries = [0]
        
        for doc_id in doc_ids:
            positions = self.get_embedding_positions(doc_id)
            all_positions.append(positions)
            boundaries.append(boundaries[-1] + len(positions))
        
        return torch.cat(all_positions), torch.tensor(boundaries, dtype=torch.int64)
    
    def get_num_embeddings(self, doc_id: int) -> int:
        """
        Get the number of embeddings for a document (without fetching positions).
        
        Args:
            doc_id: The document ID.
        
        Returns:
            Number of embeddings for this document.
        """
        return (self.doc_offsets[doc_id + 1] - self.doc_offsets[doc_id]).item()
    
    def get_centroid_for_position(
        self, 
        embedding_pos: int,
        offsets_compacted: torch.Tensor
    ) -> int:
        """
        Get the centroid ID for an embedding position.
        
        Since WARP stores embeddings sorted by centroid, we can find the centroid
        via binary search on offsets_compacted.
        
        Args:
            embedding_pos: Position in the compacted arrays.
            offsets_compacted: Cumulative centroid sizes from the WARP index.
        
        Returns:
            The centroid ID that owns this embedding.
        """
        # Binary search: find largest i where offsets_compacted[i] <= embedding_pos
        centroid_id = torch.searchsorted(offsets_compacted, embedding_pos, right=True) - 1
        return centroid_id.item()
    
    def __repr__(self) -> str:
        return (
            f"ReverseIndex(num_docs={self.num_docs:,}, "
            f"num_embeddings={self.num_embeddings:,})"
        )


# =============================================================================
# Standalone usage example
# =============================================================================

def example_usage():
    """
    Demonstrates how to use the ReverseIndex.
    
    Run this example:
        python -m warp.engine.utils.reverse_index
    """
    import numpy as np
    
    # Example with the BEIR-Quora index
    index_path = "/mnt/datasets/index/beir-quora.split=test.nbits=4"
    
    print("=" * 60)
    print("ReverseIndex Example")
    print("=" * 60)
    
    # Build or load the reverse index
    reverse_idx = ReverseIndex.load_or_build(index_path)
    print(f"\n{reverse_idx}")
    
    # Look up embeddings for a specific document
    print("\n" + "-" * 60)
    print("Single Document Lookup")
    print("-" * 60)
    
    doc_id = 42
    positions = reverse_idx.get_embedding_positions(doc_id)
    print(f"Document {doc_id} has {len(positions)} embeddings")
    print(f"Embedding positions: {positions.tolist()}")
    
    # Verify by checking codes_compacted
    codes_compacted = torch.load(f"{index_path}/codes.compacted.pt")
    doc_ids_at_positions = codes_compacted[positions]
    print(f"Doc IDs at those positions: {doc_ids_at_positions.tolist()}")
    assert (doc_ids_at_positions == doc_id).all(), "Mismatch!"
    print("✓ Verified: all positions map back to the correct doc_id")
    
    # Batch lookup
    print("\n" + "-" * 60)
    print("Batch Document Lookup (for oracle computation)")
    print("-" * 60)
    
    candidate_docs = [10, 100, 1000, 10000]
    positions, boundaries = reverse_idx.get_embedding_positions_batch(candidate_docs)
    
    print(f"Total embeddings for {len(candidate_docs)} docs: {len(positions)}")
    for i, doc_id in enumerate(candidate_docs):
        start, end = boundaries[i].item(), boundaries[i + 1].item()
        print(f"  Doc {doc_id}: {end - start} embeddings (positions {start}:{end})")
    
    # Get centroid IDs for the embeddings
    print("\n" + "-" * 60)
    print("Centroid Lookup (for M5/M6 analysis)")
    print("-" * 60)
    
    sizes_compacted = torch.load(f"{index_path}/sizes.compacted.pt")
    offsets_compacted = torch.zeros(len(sizes_compacted) + 1, dtype=torch.int64)
    torch.cumsum(sizes_compacted, dim=0, out=offsets_compacted[1:])
    
    doc_id = 42
    positions = reverse_idx.get_embedding_positions(doc_id)
    print(f"Document {doc_id}'s embeddings are in these centroids:")
    for pos in positions[:5]:  # Show first 5
        centroid = reverse_idx.get_centroid_for_position(pos.item(), offsets_compacted)
        print(f"  Position {pos.item()} → Centroid {centroid}")
    if len(positions) > 5:
        print(f"  ... and {len(positions) - 5} more")


if __name__ == "__main__":
    example_usage()
