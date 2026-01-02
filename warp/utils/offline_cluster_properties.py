"""
Offline Cluster Properties Computation

Computes offline cluster property metrics (A1, A2, A3, A5, B5) directly from
WARP index files without running any queries.

Usage:
    from warp.utils.offline_cluster_properties import OfflineClusterPropertiesComputer
    
    computer = OfflineClusterPropertiesComputer(
        index_path="/mnt/datasets/index/beir-quora.split=test.nbits=4"
    )
    
    # Compute all metrics and save to Parquet
    df = computer.compute_all()
    
    # Or compute individually
    a1_a2_a3 = computer.compute_a1_a2_a3()
    b5 = computer.compute_b5()
    a5 = computer.compute_a5()

Output: {index_path}/cluster_properties_offline.parquet

Metrics:
    - A1 (n_tokens): Number of embeddings per centroid
    - A2 (n_docs): Number of unique documents per centroid  
    - A3 (concentration): Document-token concentration (top-k shares, gini)
    - A5 (dispersion): Within-centroid dispersion (mean squared distance)
    - B5 (isolation): Inter-centroid isolation (1 - mean k-NN similarity)

See docs/CLUSTER_PROPERTIES_OFFLINE.md for full specification.
See docs/OFFLINE_CLUSTER_PROPERTIES_INTEGRATION_PLAN.md for implementation details.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd


@dataclass
class OfflineMetricsConfig:
    """Configuration for offline metrics computation."""
    
    # A5 (dispersion) parameters
    a5_sample_fraction: float = 0.25
    a5_min_samples: int = 100
    a5_max_samples: int = 10000
    a5_seed: int = 42
    
    # B5 (isolation) parameters
    b5_k_neighbors: int = 10
    b5_batch_size: int = 1000  # For batched similarity computation


class OfflineClusterPropertiesComputer:
    """
    Compute offline cluster properties from a WARP index.
    
    This class loads WARP index files and computes various metrics that
    characterize the static structure of the index without running queries.
    
    Attributes:
        index_path: Path to WARP index directory
        config: Configuration for metrics computation
        num_centroids: Number of centroids in the index
        num_embeddings: Total number of embeddings
    """
    
    def __init__(
        self,
        index_path: str,
        config: Optional[OfflineMetricsConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize the offline cluster properties computer.
        
        Args:
            index_path: Path to WARP index directory
            config: Optional configuration for metrics computation
            verbose: Whether to print progress messages
        """
        self.index_path = Path(index_path)
        self.config = config or OfflineMetricsConfig()
        self.verbose = verbose
        
        # Validate index path
        if not self.index_path.exists():
            raise ValueError(f"Index path does not exist: {self.index_path}")
        
        # Lazy-loaded components
        self._centroids: Optional[torch.Tensor] = None
        self._sizes_compacted: Optional[torch.Tensor] = None
        self._offsets_compacted: Optional[torch.Tensor] = None
        self._codes_compacted: Optional[torch.Tensor] = None
        self._residuals_compacted: Optional[torch.Tensor] = None
        self._bucket_weights: Optional[torch.Tensor] = None
        self._codec = None
        
        # Cached values
        self._num_centroids: Optional[int] = None
        self._num_embeddings: Optional[int] = None
        self._nbits: Optional[int] = None
    
    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # Index Loading
    # =========================================================================
    
    @property
    def centroids(self) -> torch.Tensor:
        """Load centroids (C, 128) lazily."""
        if self._centroids is None:
            path = self.index_path / "centroids.npy"
            self._centroids = torch.from_numpy(np.load(str(path)))
            self._log(f"  Loaded centroids: {self._centroids.shape}")
        return self._centroids
    
    @property
    def sizes_compacted(self) -> torch.Tensor:
        """Load sizes_compacted (C,) lazily."""
        if self._sizes_compacted is None:
            path = self.index_path / "sizes.compacted.pt"
            self._sizes_compacted = torch.load(str(path))
            self._num_centroids = len(self._sizes_compacted)
            self._log(f"  Loaded sizes_compacted: {self._sizes_compacted.shape}")
        return self._sizes_compacted
    
    @property
    def offsets_compacted(self) -> torch.Tensor:
        """Compute offsets from sizes lazily."""
        if self._offsets_compacted is None:
            sizes = self.sizes_compacted
            self._offsets_compacted = torch.zeros((len(sizes) + 1,), dtype=torch.long)
            torch.cumsum(sizes, dim=0, out=self._offsets_compacted[1:])
            self._log(f"  Computed offsets_compacted: {self._offsets_compacted.shape}")
        return self._offsets_compacted
    
    @property
    def codes_compacted(self) -> torch.Tensor:
        """Load codes_compacted (N,) lazily - contains document IDs."""
        if self._codes_compacted is None:
            path = self.index_path / "codes.compacted.pt"
            self._codes_compacted = torch.load(str(path))
            self._num_embeddings = len(self._codes_compacted)
            self._log(f"  Loaded codes_compacted: {self._codes_compacted.shape}")
        return self._codes_compacted
    
    @property
    def residuals_compacted(self) -> torch.Tensor:
        """Load residuals_compacted (N, packed_dim) lazily."""
        if self._residuals_compacted is None:
            path = self.index_path / "residuals.compacted.pt"
            self._residuals_compacted = torch.load(str(path))
            # Infer nbits from packed_dim
            packed_dim = self._residuals_compacted.shape[1]
            self._nbits = 4 if packed_dim == 64 else 2
            self._log(f"  Loaded residuals_compacted: {self._residuals_compacted.shape}")
        return self._residuals_compacted
    
    @property 
    def bucket_weights(self) -> torch.Tensor:
        """Load bucket_weights lazily."""
        if self._bucket_weights is None:
            path = self.index_path / "bucket_weights.npy"
            self._bucket_weights = torch.from_numpy(np.load(str(path)))
            self._log(f"  Loaded bucket_weights: {self._bucket_weights.shape}")
        return self._bucket_weights
    
    @property
    def codec(self):
        """Load ResidualCodec lazily for decompression."""
        if self._codec is None:
            from warp.indexing.codecs.residual import ResidualCodec
            self._codec = ResidualCodec.load(str(self.index_path))
            self._log(f"  Loaded ResidualCodec (nbits={self._codec.nbits})")
        return self._codec
    
    @property
    def num_centroids(self) -> int:
        """Get number of centroids."""
        if self._num_centroids is None:
            _ = self.sizes_compacted  # Trigger load
        return self._num_centroids
    
    @property
    def num_embeddings(self) -> int:
        """Get total number of embeddings."""
        if self._num_embeddings is None:
            _ = self.codes_compacted  # Trigger load
        return self._num_embeddings
    
    @property
    def nbits(self) -> int:
        """Get quantization bits (2 or 4)."""
        if self._nbits is None:
            _ = self.residuals_compacted  # Trigger load
        return self._nbits
    
    def load_all(self):
        """Load all index components."""
        self._log("Loading index components...")
        start = time.time()
        
        _ = self.centroids
        _ = self.sizes_compacted
        _ = self.offsets_compacted
        _ = self.codes_compacted
        _ = self.residuals_compacted
        _ = self.bucket_weights
        
        elapsed = time.time() - start
        self._log(f"  Index loading completed in {elapsed:.2f}s")
        self._log(f"  {self.num_centroids:,} centroids, {self.num_embeddings:,} embeddings")
    
    # =========================================================================
    # A1, A2, A3: Token-List Size, Unique Docs, Concentration (Single Pass)
    # =========================================================================
    
    def compute_a1_a2_a3(self) -> Dict[str, np.ndarray]:
        """
        Compute A1 (token-list size), A2 (unique docs), A3 (concentration) in one pass.
        
        Returns:
            Dictionary with keys:
                - n_tokens: (C,) int64 - number of embeddings per centroid
                - n_docs: (C,) int64 - unique documents per centroid
                - top_1_doc_share: (C,) float32 - fraction from top-1 document
                - top_5_doc_share: (C,) float32 - fraction from top-5 documents
                - top_10_doc_share: (C,) float32 - fraction from top-10 documents
                - gini_coefficient: (C,) float32 - Gini coefficient of token distribution
        """
        self._log("\nComputing A1, A2, A3 metrics...")
        start = time.time()
        
        sizes = self.sizes_compacted.numpy()
        offsets = self.offsets_compacted.numpy()
        codes = self.codes_compacted.numpy()
        num_centroids = self.num_centroids
        
        # Allocate output arrays
        n_tokens = sizes.copy().astype(np.int64)  # A1: direct copy
        n_docs = np.zeros(num_centroids, dtype=np.int64)
        top_1_share = np.zeros(num_centroids, dtype=np.float32)
        top_5_share = np.zeros(num_centroids, dtype=np.float32)
        top_10_share = np.zeros(num_centroids, dtype=np.float32)
        gini = np.zeros(num_centroids, dtype=np.float32)
        
        # Process each centroid
        for c in range(num_centroids):
            begin = offsets[c]
            end = offsets[c + 1]
            
            if begin == end:
                continue
            
            doc_ids = codes[begin:end]
            n_total = len(doc_ids)
            
            # A2: Unique document count
            unique_docs, counts = np.unique(doc_ids, return_counts=True)
            n_docs[c] = len(unique_docs)
            
            # A3: Concentration metrics
            sorted_counts = np.sort(counts)[::-1]  # Descending order
            
            # Top-k shares (handle case where fewer than k docs exist)
            top_1_share[c] = sorted_counts[:1].sum() / n_total
            top_5_share[c] = sorted_counts[:5].sum() / n_total
            top_10_share[c] = sorted_counts[:10].sum() / n_total
            
            # Gini coefficient
            gini[c] = self._compute_gini(counts)
        
        elapsed = time.time() - start
        self._log(f"  A1/A2/A3 computed in {elapsed:.2f}s")
        
        # Log summary statistics
        non_empty = n_tokens > 0
        self._log(f"  A1 (n_tokens): min={n_tokens.min()}, max={n_tokens.max()}, "
                  f"mean={n_tokens.mean():.1f}")
        self._log(f"  A2 (n_docs): min={n_docs[non_empty].min()}, max={n_docs.max()}, "
                  f"mean={n_docs[non_empty].mean():.1f}")
        self._log(f"  A3 (gini): min={gini[non_empty].min():.3f}, max={gini[non_empty].max():.3f}, "
                  f"mean={gini[non_empty].mean():.3f}")
        
        return {
            'n_tokens': n_tokens,
            'n_docs': n_docs,
            'top_1_doc_share': top_1_share,
            'top_5_doc_share': top_5_share,
            'top_10_doc_share': top_10_share,
            'gini_coefficient': gini,
        }
    
    @staticmethod
    def _compute_gini(counts: np.ndarray) -> float:
        """Compute Gini coefficient for a distribution."""
        if len(counts) == 0:
            return 0.0
        n = len(counts)
        sorted_counts = np.sort(counts)
        index = np.arange(1, n + 1)
        total = np.sum(sorted_counts)
        if total == 0:
            return 0.0
        return (2 * np.sum(index * sorted_counts) / (n * total)) - (n + 1) / n
    
    # =========================================================================
    # B5: Inter-Centroid Isolation
    # =========================================================================
    
    def compute_b5(self, use_batched: bool = False) -> Dict[str, np.ndarray]:
        """
        Compute B5: Inter-centroid isolation metrics.
        
        Args:
            use_batched: If True, use batched computation (for very large indexes).
                        If False, compute full similarity matrix.
        
        Returns:
            Dictionary with keys:
                - isolation: (C,) float32 - isolation score (1 - mean k-NN similarity)
                - nearest_neighbor_sim: (C,) float32 - similarity to nearest neighbor
                - mean_neighbor_sim: (C,) float32 - mean similarity to k nearest neighbors
        """
        self._log("\nComputing B5 metrics (centroid isolation)...")
        start = time.time()
        
        centroids = self.centroids.float()
        num_centroids = self.num_centroids
        k = min(self.config.b5_k_neighbors, num_centroids - 1)
        
        # Estimate memory for full matrix
        mem_gb = num_centroids * num_centroids * 4 / 1e9
        
        if use_batched or mem_gb > 50:  # Use batched if > 50GB
            self._log(f"  Using batched approach (estimated {mem_gb:.1f}GB for full matrix)")
            return self._compute_b5_batched(centroids, k)
        else:
            self._log(f"  Using full matrix approach ({mem_gb:.2f}GB)")
            return self._compute_b5_full(centroids, k, start)
    
    def _compute_b5_full(
        self, centroids: torch.Tensor, k: int, start_time: float
    ) -> Dict[str, np.ndarray]:
        """Compute B5 using full similarity matrix."""
        num_centroids = centroids.shape[0]
        
        # Compute full similarity matrix (cosine sim = dot product for normalized vectors)
        similarities = centroids @ centroids.T  # (C, C)
        
        # Zero out diagonal (self-similarity)
        similarities.fill_diagonal_(float('-inf'))
        
        # Get top-k neighbors per centroid
        top_k_sims, _ = torch.topk(similarities, k, dim=1)  # (C, k)
        
        # Compute metrics
        mean_neighbor_sim = top_k_sims.mean(dim=1).numpy()  # (C,)
        nearest_neighbor_sim = top_k_sims[:, 0].numpy()  # (C,) - highest sim = nearest
        isolation = 1.0 - mean_neighbor_sim  # (C,)
        
        # Cleanup
        del similarities
        
        elapsed = time.time() - start_time
        self._log(f"  B5 computed in {elapsed:.2f}s")
        self._log(f"  isolation: min={isolation.min():.4f}, max={isolation.max():.4f}, "
                  f"mean={isolation.mean():.4f}")
        
        return {
            'isolation': isolation.astype(np.float32),
            'nearest_neighbor_sim': nearest_neighbor_sim.astype(np.float32),
            'mean_neighbor_sim': mean_neighbor_sim.astype(np.float32),
        }
    
    def _compute_b5_batched(
        self, centroids: torch.Tensor, k: int
    ) -> Dict[str, np.ndarray]:
        """Compute B5 using batched computation for large indexes."""
        num_centroids = centroids.shape[0]
        batch_size = self.config.b5_batch_size
        
        isolation = np.zeros(num_centroids, dtype=np.float32)
        nearest_neighbor_sim = np.zeros(num_centroids, dtype=np.float32)
        mean_neighbor_sim = np.zeros(num_centroids, dtype=np.float32)
        
        for i in range(0, num_centroids, batch_size):
            end_i = min(i + batch_size, num_centroids)
            batch = centroids[i:end_i]  # (batch_size, 128)
            
            # Compute similarities for this batch against all centroids
            sims = batch @ centroids.T  # (batch_size, C)
            
            # Zero out self-similarities
            for j in range(end_i - i):
                sims[j, i + j] = float('-inf')
            
            # Get top-k
            top_k_sims, _ = torch.topk(sims, k, dim=1)
            
            mean_neighbor_sim[i:end_i] = top_k_sims.mean(dim=1).numpy()
            nearest_neighbor_sim[i:end_i] = top_k_sims[:, 0].numpy()
            isolation[i:end_i] = 1.0 - mean_neighbor_sim[i:end_i]
        
        self._log(f"  isolation: min={isolation.min():.4f}, max={isolation.max():.4f}, "
                  f"mean={isolation.mean():.4f}")
        
        return {
            'isolation': isolation,
            'nearest_neighbor_sim': nearest_neighbor_sim,
            'mean_neighbor_sim': mean_neighbor_sim,
        }
    
    # =========================================================================
    # A5: Within-Centroid Dispersion
    # =========================================================================
    
    def compute_a5(self) -> Dict[str, np.ndarray]:
        """
        Compute A5: Within-centroid dispersion (mean squared distance to centroid).
        
        Uses sampling for efficiency (configurable via config.a5_* parameters).
        
        Returns:
            Dictionary with keys:
                - dispersion: (C,) float32 - mean squared distance to centroid
                - samples_used: (C,) int32 - number of samples used per centroid
        """
        self._log("\nComputing A5 metrics (within-centroid dispersion)...")
        start = time.time()
        
        offsets = self.offsets_compacted.numpy()
        residuals_compacted = self.residuals_compacted
        codec = self.codec
        num_centroids = self.num_centroids
        
        # Config
        sample_fraction = self.config.a5_sample_fraction
        min_samples = self.config.a5_min_samples
        max_samples = self.config.a5_max_samples
        rng = np.random.default_rng(self.config.a5_seed)
        
        # Output arrays
        dispersion = np.zeros(num_centroids, dtype=np.float32)
        samples_used = np.zeros(num_centroids, dtype=np.int32)
        
        for c in range(num_centroids):
            begin = offsets[c]
            end = offsets[c + 1]
            n = end - begin
            
            if n == 0:
                continue
            
            # Determine sample size
            n_samples = int(n * sample_fraction)
            n_samples = max(min_samples, min(n_samples, max_samples, n))
            samples_used[c] = n_samples
            
            # Sample indices
            if n_samples < n:
                local_idx = rng.choice(n, size=n_samples, replace=False)
                global_idx = begin + local_idx
            else:
                global_idx = np.arange(begin, end)
                n_samples = n
            
            # Get residuals for sampled embeddings
            sample_residuals = residuals_compacted[global_idx]
            
            # Decompress using codec's CPU path (without final normalization)
            # This gives us: embedding = centroid + residual_contribution
            codes_t = torch.full((n_samples,), c, dtype=torch.int32)
            
            # Raw decompression (before normalization)
            centroids_ = codec.lookup_centroids(codes_t, out_device='cpu')
            residuals_ = codec.reversed_bit_map[sample_residuals.long()]
            residuals_ = codec.decompression_lookup_table[residuals_.long()]
            residuals_ = residuals_.reshape(n_samples, -1)
            residuals_ = codec.bucket_weights[residuals_.long()]
            embeddings = centroids_ + residuals_
            
            # Compute squared distances to centroid
            centroid = codec.centroids[c].float()
            sq_distances = ((embeddings.float() - centroid) ** 2).sum(dim=1)
            dispersion[c] = sq_distances.mean().item()
        
        elapsed = time.time() - start
        self._log(f"  A5 computed in {elapsed:.2f}s")
        
        # Statistics
        non_empty = samples_used > 0
        self._log(f"  dispersion: min={dispersion[non_empty].min():.4f}, "
                  f"max={dispersion[non_empty].max():.4f}, "
                  f"mean={dispersion[non_empty].mean():.4f}")
        self._log(f"  samples_used: mean={samples_used[non_empty].mean():.0f}")
        
        return {
            'dispersion': dispersion,
            'samples_used': samples_used,
        }
    
    # =========================================================================
    # Compute All and Save
    # =========================================================================
    
    def compute_all(self, save: bool = True) -> pd.DataFrame:
        """
        Compute all offline cluster property metrics.
        
        Args:
            save: If True, save results to Parquet file in index directory.
        
        Returns:
            DataFrame with all metrics per centroid.
        """
        self._log("=" * 70)
        self._log("OFFLINE CLUSTER PROPERTIES COMPUTATION")
        self._log("=" * 70)
        self._log(f"Index: {self.index_path}")
        
        total_start = time.time()
        
        # Load index
        self.load_all()
        
        # Compute all metrics
        a1_a2_a3 = self.compute_a1_a2_a3()
        b5 = self.compute_b5()
        a5 = self.compute_a5()
        
        # Combine into DataFrame
        self._log("\nBuilding output DataFrame...")
        
        df = pd.DataFrame({
            'centroid_id': np.arange(self.num_centroids, dtype=np.int32),
            # A1
            'n_tokens': a1_a2_a3['n_tokens'],
            # A2
            'n_docs': a1_a2_a3['n_docs'],
            # A3
            'top_1_doc_share': a1_a2_a3['top_1_doc_share'],
            'top_5_doc_share': a1_a2_a3['top_5_doc_share'],
            'top_10_doc_share': a1_a2_a3['top_10_doc_share'],
            'gini_coefficient': a1_a2_a3['gini_coefficient'],
            # A5
            'dispersion': a5['dispersion'],
            'dispersion_samples_used': a5['samples_used'],
            # B5
            'isolation': b5['isolation'],
            'nearest_neighbor_sim': b5['nearest_neighbor_sim'],
            'mean_neighbor_sim': b5['mean_neighbor_sim'],
        })
        
        # Add derived metrics
        df['tokens_per_doc'] = np.where(
            df['n_docs'] > 0,
            df['n_tokens'] / df['n_docs'],
            0.0
        ).astype(np.float32)
        
        total_elapsed = time.time() - total_start
        self._log(f"\nTotal computation time: {total_elapsed:.2f}s")
        
        # Save to Parquet
        if save:
            output_path = self.index_path / "cluster_properties_offline.parquet"
            df.to_parquet(output_path, index=False)
            self._log(f"\nSaved to: {output_path}")
        
        return df
    
    def print_summary(self, df: Optional[pd.DataFrame] = None):
        """Print summary statistics of computed metrics."""
        if df is None:
            # Try to load from file
            output_path = self.index_path / "cluster_properties_offline.parquet"
            if output_path.exists():
                df = pd.read_parquet(output_path)
            else:
                raise ValueError("No DataFrame provided and no saved file found")
        
        print("\n" + "=" * 70)
        print("OFFLINE CLUSTER PROPERTIES SUMMARY")
        print("=" * 70)
        print(f"Index: {self.index_path}")
        print(f"Centroids: {len(df):,}")
        print(f"Total embeddings: {df['n_tokens'].sum():,}")
        
        non_empty = df['n_tokens'] > 0
        
        print("\n--- A1: Token-List Size ---")
        print(f"  min: {df['n_tokens'].min():,}")
        print(f"  max: {df['n_tokens'].max():,}")
        print(f"  mean: {df['n_tokens'].mean():.1f}")
        print(f"  std: {df['n_tokens'].std():.1f}")
        print(f"  imbalance ratio: {df['n_tokens'].max() / df['n_tokens'].mean():.1f}x")
        
        print("\n--- A2: Unique Documents ---")
        print(f"  min: {df.loc[non_empty, 'n_docs'].min():,}")
        print(f"  max: {df['n_docs'].max():,}")
        print(f"  mean: {df.loc[non_empty, 'n_docs'].mean():.1f}")
        print(f"  correlation(n_tokens, n_docs): {df.loc[non_empty, ['n_tokens', 'n_docs']].corr().iloc[0, 1]:.3f}")
        
        print("\n--- A3: Document-Token Concentration ---")
        print(f"  top_1_doc_share: mean={df.loc[non_empty, 'top_1_doc_share'].mean():.3f}, "
              f"max={df['top_1_doc_share'].max():.3f}")
        print(f"  top_5_doc_share: mean={df.loc[non_empty, 'top_5_doc_share'].mean():.3f}")
        print(f"  top_10_doc_share: mean={df.loc[non_empty, 'top_10_doc_share'].mean():.3f}")
        print(f"  gini: mean={df.loc[non_empty, 'gini_coefficient'].mean():.3f}, "
              f"max={df['gini_coefficient'].max():.3f}")
        
        print("\n--- A5: Within-Centroid Dispersion ---")
        print(f"  min: {df.loc[non_empty, 'dispersion'].min():.4f}")
        print(f"  max: {df['dispersion'].max():.4f}")
        print(f"  mean: {df.loc[non_empty, 'dispersion'].mean():.4f}")
        
        print("\n--- B5: Inter-Centroid Isolation ---")
        print(f"  min: {df['isolation'].min():.4f}")
        print(f"  max: {df['isolation'].max():.4f}")
        print(f"  mean: {df['isolation'].mean():.4f}")
        print(f"  nearest_neighbor_sim: mean={df['nearest_neighbor_sim'].mean():.4f}")
        
        # Identify problematic centroids
        print("\n--- Diagnostics ---")
        dominated = (df['top_1_doc_share'] > 0.5) & non_empty
        print(f"  Dominated centroids (top-1 > 50%): {dominated.sum():,} "
              f"({dominated.sum() / non_empty.sum() * 100:.1f}%)")
        
        high_dispersion = df['dispersion'] > df['dispersion'].quantile(0.95)
        print(f"  High dispersion (top 5%): {high_dispersion.sum():,}")
        
        low_isolation = df['isolation'] < df['isolation'].quantile(0.05)
        print(f"  Low isolation (bottom 5%): {low_isolation.sum():,}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for offline cluster properties computation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute offline cluster properties for a WARP index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute all metrics and save to Parquet
    python -m warp.utils.offline_cluster_properties --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4
    
    # Just print summary of existing metrics
    python -m warp.utils.offline_cluster_properties --index-path /path/to/index --summary-only
        """
    )
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to WARP index directory"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing metrics (don't recompute)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to Parquet file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    parser.add_argument(
        "--a5-sample-fraction",
        type=float,
        default=0.25,
        help="Fraction of embeddings to sample for A5 dispersion (default: 0.25)"
    )
    parser.add_argument(
        "--b5-k-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for B5 isolation (default: 10)"
    )
    
    args = parser.parse_args()
    
    config = OfflineMetricsConfig(
        a5_sample_fraction=args.a5_sample_fraction,
        b5_k_neighbors=args.b5_k_neighbors,
    )
    
    computer = OfflineClusterPropertiesComputer(
        index_path=args.index_path,
        config=config,
        verbose=not args.quiet
    )
    
    if args.summary_only:
        computer.print_summary()
    else:
        df = computer.compute_all(save=not args.no_save)
        computer.print_summary(df)


if __name__ == "__main__":
    main()
