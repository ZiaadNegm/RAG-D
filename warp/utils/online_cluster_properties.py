"""
Online Cluster Properties Computation

Computes online cluster property metrics from WARP measurement data (M1, M3, M4, R0).
These metrics characterize routing behavior during actual query execution.

Usage:
    from warp.utils.online_cluster_properties import OnlineClusterPropertiesComputer
    
    computer = OnlineClusterPropertiesComputer(
        run_dir="/mnt/warp_measurements/runs/my_run",
        index_path="/mnt/datasets/index/beir-quora.split=test.nbits=4"
    )
    
    # Compute all metrics
    results = computer.compute_all()
    
    # Or compute by phase
    phase2 = computer.compute_phase2_core_metrics()  # A6/B1, B2, B3, C1, C2
    phase3 = computer.compute_phase3_yield_metrics()  # A4, B4
    phase4 = computer.compute_phase4_oracle_metrics()  # C3, C6 (parallel)
    phase5 = computer.compute_phase5_routing_fidelity()  # C4, C5 (parallel)

Output: {run_dir}/cluster_properties_online/

Metrics:
    Phase 2 (Light - M1 only):
    - A6/B1: Selection frequency and traffic concentration
    - B2: Anti-hub rate (centroids never selected)
    - B3: Traffic concentration (Gini, entropy, top-k shares)
    - C1: Per-query activation entropy
    - C2: Per-query load imbalance
    
    Phase 3 (Medium - M1 + M3):
    - A4: Yield per centroid (influential / computed)
    - B4: Hub classification (good_hub, bad_hub, normal)
    
    Phase 4 (Heavy - M4, parallelized):
    - C3: Pruning recall at ranking level
    - C6: Evidence dispersion per document
    
    Phase 5 (Heavy - M4 + R0, parallelized):
    - C4/C5: Routing fidelity (oracle hit/miss rates)

See docs/CLUSTER_PROPERTIES_ONLINE.md for metric definitions.
See docs/ONLINE_CLUSTER_PROPERTIES_INTEGRATION_PLAN.md for implementation details.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from warp.utils.chunked_m4 import ChunkedM4Processor, DEFAULT_CHUNK_SIZE


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnlineMetricsConfig:
    """Configuration for online metrics computation."""
    
    # Hub classification (B4)
    hub_percentile: float = 95.0       # Top X% by traffic are "hubs"
    bad_yield_threshold: float = 0.1   # Yield below this → bad hub
    good_yield_threshold: float = 0.3  # Yield above this → good hub
    
    # Parallelization (Phase 4, 5)
    num_workers: int = 8               # Workers for heavy metrics
    chunk_size: int = 100              # Queries per parallel chunk
    
    # Chunked M4 processing (for large runs)
    m4_chunk_size: int = DEFAULT_CHUNK_SIZE  # Queries per M4 chunk (default: 500)
    
    # Recall@k values (C3 - renamed to "oracle_evidence_recall")
    recall_k_values: List[int] = field(default_factory=lambda: [10, 100, 1000])
    
    # Streamlining flags (SQ2 cleanup)
    skip_c4_routing_fidelity: bool = True   # C4 hit/miss redundant with M5
    skip_b2_entropy: bool = True            # Keep Gini + top-p, skip entropy
    c2_internal_only: bool = True           # Mark C2 as internal/debug metric


# =============================================================================
# Reusable Utilities (shared with derived_metrics.py and offline_cluster_properties.py)
# =============================================================================

def compute_gini(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for a distribution.
    
    Reused from OfflineClusterPropertiesComputer._compute_gini()
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1]. 0 = perfect equality, 1 = max inequality
    """
    if len(values) == 0:
        return 0.0
    n = len(values)
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    return (2 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n


def compute_entropy(counts: np.ndarray, normalize: bool = False) -> float:
    """
    Compute Shannon entropy of a distribution.
    
    Args:
        counts: Array of counts (will be normalized to probabilities)
        normalize: If True, return normalized entropy (0-1 scale)
        
    Returns:
        Entropy in nats. If normalize=True, returns H / log(n).
    """
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]  # Remove zeros for log
    entropy = -np.sum(p * np.log(p))
    
    if normalize and len(counts) > 1:
        max_entropy = np.log(len(counts))
        return entropy / max_entropy
    return entropy


def load_offsets_from_index(index_path: Path) -> torch.Tensor:
    """
    Load or compute offsets_compacted from sizes.compacted.pt.
    
    Reused pattern from DerivedMetricsComputer.offsets_compacted
    
    Args:
        index_path: Path to WARP index directory
        
    Returns:
        Tensor of shape (num_centroids + 1,) for centroid lookup
    """
    sizes = torch.load(index_path / "sizes.compacted.pt")
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    torch.cumsum(sizes, dim=0, out=offsets[1:])
    return offsets


def embedding_pos_to_centroid(
    embedding_positions: torch.Tensor,
    offsets: torch.Tensor
) -> torch.Tensor:
    """
    Map embedding positions to centroid IDs via binary search.
    
    Reused pattern from DerivedMetricsComputer.compute_m5()
    
    Args:
        embedding_positions: Tensor of global embedding positions
        offsets: Offsets tensor from load_offsets_from_index()
        
    Returns:
        Tensor of centroid IDs (same shape as embedding_positions)
    """
    return torch.searchsorted(offsets, embedding_positions, side='right') - 1


# =============================================================================
# Parallelization Helper Functions (module-level for pickling)
# =============================================================================

def _compute_pruning_recall_single_query(
    query_id: int,
    actual_scores_q: pd.DataFrame,
    oracle_scores_q: pd.DataFrame,
    k_values: List[int]
) -> Dict[str, Any]:
    """Compute pruning recall for a single query (parallelizable)."""
    actual_q = actual_scores_q.sort_values('actual_score', ascending=False)
    oracle_q = oracle_scores_q.sort_values('oracle_score', ascending=False)
    
    row = {'query_id': query_id}
    
    for k in k_values:
        actual_top_k = set(actual_q.head(k)['doc_id'])
        oracle_top_k = set(oracle_q.head(k)['doc_id'])
        
        overlap = len(actual_top_k & oracle_top_k)
        recall = overlap / min(k, len(oracle_top_k)) if oracle_top_k else 1.0
        row[f'recall@{k}'] = recall
    
    return row


def _compute_routing_fidelity_single_query(
    query_id: int,
    m4_q: pd.DataFrame,
    r0_sets_q: Dict[int, set],
    offsets_np: np.ndarray
) -> Dict[str, int]:
    """Compute routing fidelity for a single query (parallelizable)."""
    # Derive oracle centroid from embedding positions
    oracle_positions = m4_q['oracle_embedding_pos'].values
    offsets_tensor = torch.from_numpy(offsets_np)
    oracle_centroids = torch.searchsorted(
        offsets_tensor, 
        torch.from_numpy(oracle_positions.astype(np.int64)), 
        side='right'
    ) - 1
    oracle_centroids = oracle_centroids.numpy()
    
    hits = 0
    misses = 0
    
    for tid, oc in zip(m4_q['q_token_id'].values, oracle_centroids):
        selected = r0_sets_q.get(tid, set())
        if oc in selected:
            hits += 1
        else:
            misses += 1
    
    return {'query_id': query_id, 'hits': hits, 'misses': misses}


# =============================================================================
# Main Computer Class
# =============================================================================

class OnlineClusterPropertiesComputer:
    """
    Compute online cluster properties from WARP measurement data.
    
    This class loads measurement Parquet files (M1, M3, M4, R0) and computes
    various metrics that characterize routing behavior during query execution.
    
    Follows the same pattern as:
    - DerivedMetricsComputer (for M2, M5, M6)
    - OfflineClusterPropertiesComputer (for A1-A5, B5)
    
    Attributes:
        run_dir: Path to measurement run directory
        index_path: Path to WARP index
        output_dir: Path to output directory
        config: Configuration for metrics computation
    """
    
    def __init__(
        self,
        run_dir: str,
        index_path: str,
        output_dir: Optional[str] = None,
        config: Optional[OnlineMetricsConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize the online cluster properties computer.
        
        Args:
            run_dir: Path to measurement run directory (contains tier_a/, tier_b/)
            index_path: Path to WARP index directory
            output_dir: Output directory (default: {run_dir}/cluster_properties_online)
            config: Configuration for metrics computation
            verbose: Whether to print progress messages
        """
        self.run_dir = Path(run_dir)
        self.index_path = Path(index_path)
        self.output_dir = Path(output_dir) if output_dir else self.run_dir / "cluster_properties_online"
        self.config = config or OnlineMetricsConfig()
        self.verbose = verbose
        
        # Paths to measurement files
        self.tier_a_dir = self.run_dir / "tier_a"
        self.tier_b_dir = self.run_dir / "tier_b"
        
        # Lazy-loaded data
        self._offsets: Optional[torch.Tensor] = None
        self._num_centroids: Optional[int] = None
        self._m1: Optional[pd.DataFrame] = None
        self._m3: Optional[pd.DataFrame] = None
        self._m4: Optional[pd.DataFrame] = None
        self._r0: Optional[pd.DataFrame] = None
        
        # Cached results
        self._centroid_aggregates: Optional[pd.DataFrame] = None
        self._per_query_metrics: Optional[pd.DataFrame] = None
        self._global_summary: Optional[Dict] = None
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.run_dir.exists():
            raise ValueError(f"Run directory not found: {self.run_dir}")
        if not self.index_path.exists():
            raise ValueError(f"Index path not found: {self.index_path}")
    
    def _log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # Lazy Loading (same pattern as DerivedMetricsComputer)
    # =========================================================================
    
    @property
    def offsets(self) -> torch.Tensor:
        """Load offsets_compacted lazily."""
        if self._offsets is None:
            self._offsets = load_offsets_from_index(self.index_path)
            self._num_centroids = len(self._offsets) - 1
            self._log(f"  Loaded offsets: {self._offsets.shape} ({self._num_centroids:,} centroids)")
        return self._offsets
    
    @property
    def num_centroids(self) -> int:
        """Get number of centroids in index."""
        if self._num_centroids is None:
            _ = self.offsets  # Trigger load
        return self._num_centroids
    
    @property
    def m1(self) -> pd.DataFrame:
        """Load M1 (compute per centroid) lazily."""
        if self._m1 is None:
            path = self.tier_a_dir / "M1_compute_per_centroid.parquet"
            if not path.exists():
                raise FileNotFoundError(f"M1 not found: {path}")
            self._m1 = pd.read_parquet(path)
            self._log(f"  Loaded M1: {len(self._m1):,} rows")
        return self._m1
    
    @property
    def m3(self) -> pd.DataFrame:
        """Load M3 (observed winners) lazily."""
        if self._m3 is None:
            path = self.tier_b_dir / "M3_observed_winners.parquet"
            if not path.exists():
                raise FileNotFoundError(f"M3 not found: {path}")
            self._m3 = pd.read_parquet(path)
            self._log(f"  Loaded M3: {len(self._m3):,} rows")
        return self._m3
    
    @property
    def m4(self) -> pd.DataFrame:
        """
        Load M4 (oracle winners) lazily.
        
        WARNING: For large M4 files (>1GB), this will cause OOM errors.
        Use the chunked processing methods instead:
        - _compute_oracle_evidence_recall_chunked()
        - _compute_evidence_dispersion_chunked()
        """
        if self._m4 is None:
            path = self.tier_b_dir / "M4_oracle_winners.parquet"
            if not path.exists():
                raise FileNotFoundError(f"M4 not found: {path}")
            
            # Check file size and warn if large
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 1000:
                warnings.warn(
                    f"M4 file is {size_mb:.0f} MB. Loading into memory may cause OOM. "
                    "Consider using chunked processing methods instead."
                )
            
            self._m4 = pd.read_parquet(path)
            self._log(f"  Loaded M4: {len(self._m4):,} rows")
        return self._m4
    
    def _get_m4_path(self) -> Path:
        """Get path to M4 file."""
        return self.tier_b_dir / "M4_oracle_winners.parquet"
    
    def _should_use_chunked_m4(self) -> bool:
        """Check if M4 file is large enough to warrant chunked processing."""
        m4_path = self._get_m4_path()
        if not m4_path.exists():
            return False
        size_mb = m4_path.stat().st_size / (1024 * 1024)
        return size_mb > 1000  # > 1GB
    
    @property
    def r0(self) -> pd.DataFrame:
        """Load R0 (selected centroids) lazily."""
        if self._r0 is None:
            path = self.tier_b_dir / "R0_selected_centroids.parquet"
            if not path.exists():
                raise FileNotFoundError(f"R0 not found: {path}")
            self._r0 = pd.read_parquet(path)
            self._log(f"  Loaded R0: {len(self._r0):,} rows")
        return self._r0
    
    # =========================================================================
    # Phase 2: Core Metrics (M1 only) - Light
    # =========================================================================
    
    def compute_phase2_core_metrics(self) -> Dict[str, Any]:
        """
        Compute Phase 2 metrics: A6/B1, B2, B3, C1, C2.
        
        These metrics only require M1 data (lightest tier).
        
        Returns:
            Dictionary with:
            - sel_freq: Series of selection frequency per centroid
            - traffic_concentration: Dict with Gini, entropy, top-k shares
            - anti_hub: Dict with rate and centroid IDs
            - per_query_activation: DataFrame with per-query C1, C2 metrics
        """
        self._log("\n=== Phase 2: Core Metrics (M1 only) ===")
        start = time.time()
        
        # A6/B1: Selection frequency per centroid
        sel_freq = self._compute_selection_frequency()
        
        # B2: Anti-hub rate
        anti_hub = self._compute_anti_hub_rate(sel_freq)
        
        # B3: Traffic concentration (global)
        traffic_concentration = self._compute_traffic_concentration(sel_freq)
        
        # C1, C2: Per-query metrics
        per_query = self._compute_per_query_activation()
        
        self._log(f"  Phase 2 completed in {time.time() - start:.1f}s")
        
        return {
            'sel_freq': sel_freq,
            'traffic_concentration': traffic_concentration,
            'anti_hub': anti_hub,
            'per_query_activation': per_query,
        }
    
    def _compute_selection_frequency(self) -> pd.Series:
        """
        Compute A6/B1: Selection frequency per centroid.
        
        Uses M1 which has one row per (query, token, centroid) selection.
        """
        self._log("  Computing A6/B1: Selection frequency...")
        
        # Count selections per centroid across all queries/tokens
        sel_freq = self.m1.groupby('centroid_id').size()
        
        # Reindex to include all centroids (fill missing with 0)
        sel_freq = sel_freq.reindex(range(self.num_centroids), fill_value=0)
        
        self._log(f"    Selected centroids: {(sel_freq > 0).sum():,} / {self.num_centroids:,}")
        self._log(f"    Total selections: {sel_freq.sum():,}")
        
        return sel_freq
    
    def _compute_anti_hub_rate(self, sel_freq: pd.Series) -> Dict[str, Any]:
        """
        Compute B2: Anti-hub rate (centroids never selected).
        """
        self._log("  Computing B2: Anti-hub rate...")
        
        anti_hub_mask = sel_freq == 0
        anti_hub_ids = sel_freq[anti_hub_mask].index.tolist()
        rate = len(anti_hub_ids) / self.num_centroids
        
        self._log(f"    Anti-hubs: {len(anti_hub_ids):,} ({rate:.1%})")
        
        return {
            'rate': rate,
            'count': len(anti_hub_ids),
            'centroid_ids': anti_hub_ids,
        }
    
    def _compute_traffic_concentration(self, sel_freq: pd.Series) -> Dict[str, float]:
        """
        Compute B3: Traffic concentration metrics.
        
        Reuses compute_gini() and compute_entropy() utilities.
        
        Note: Entropy metrics can be skipped via config.skip_b2_entropy
        (Gini + top-p shares are sufficient for SQ2 analysis)
        """
        self._log("  Computing B3: Traffic concentration...")
        
        values = sel_freq.values.astype(np.float64)
        total = values.sum()
        
        # Sort for top-k computation
        sorted_vals = np.sort(values)[::-1]
        
        # Top-k shares
        n = len(sorted_vals)
        top_1pct = int(max(1, n * 0.01))
        top_5pct = int(max(1, n * 0.05))
        top_10pct = int(max(1, n * 0.10))
        
        result = {
            'gini': compute_gini(values),
            'top_1pct_share': sorted_vals[:top_1pct].sum() / total if total > 0 else 0,
            'top_5pct_share': sorted_vals[:top_5pct].sum() / total if total > 0 else 0,
            'top_10pct_share': sorted_vals[:top_10pct].sum() / total if total > 0 else 0,
        }
        
        # Entropy metrics (optional - skip if config.skip_b2_entropy)
        if not self.config.skip_b2_entropy:
            result['entropy'] = compute_entropy(values)
            result['normalized_entropy'] = compute_entropy(values, normalize=True)
        
        self._log(f"    Gini: {result['gini']:.3f}")
        self._log(f"    Top 5% share: {result['top_5pct_share']:.1%}")
        
        return result
    
    def _compute_per_query_activation(self) -> pd.DataFrame:
        """
        Compute C1, C2: Per-query activation entropy and load imbalance.
        """
        self._log("  Computing C1, C2: Per-query activation...")
        
        results = []
        
        for query_id, group in self.m1.groupby('query_id'):
            # C1: Activation entropy
            centroid_counts = group.groupby('centroid_id').size().values
            entropy = compute_entropy(centroid_counts)
            norm_entropy = compute_entropy(centroid_counts, normalize=True)
            
            # C2: Load imbalance (CV and Gini of centroid loads)
            loads = group.groupby('centroid_id')['num_token_token_sims'].sum().values
            cv = loads.std() / loads.mean() if loads.mean() > 0 else 0
            gini = compute_gini(loads)
            
            # Top-k concentration within query
            sorted_loads = np.sort(loads)[::-1]
            total_load = loads.sum()
            top_5 = sorted_loads[:max(1, len(sorted_loads) // 20)]
            top_5_share = top_5.sum() / total_load if total_load > 0 else 0
            
            results.append({
                'query_id': query_id,
                'unique_centroids': len(centroid_counts),
                'activation_entropy': entropy,
                'normalized_activation_entropy': norm_entropy,
                'load_cv': cv,
                'load_gini': gini,
                'top_5_centroid_share': top_5_share,
            })
        
        df = pd.DataFrame(results)
        self._log(f"    Processed {len(df):,} queries")
        
        return df
    
    # =========================================================================
    # Phase 3: Yield and Hub Classification (M1 + M3) - Medium
    # =========================================================================
    
    def compute_phase3_yield_metrics(self) -> Dict[str, Any]:
        """
        Compute Phase 3 metrics: A4 (yield), B4 (hub classification).
        
        Requires M1 and M3 data.
        
        Returns:
            Dictionary with:
            - yield_per_centroid: DataFrame with yield metrics
            - hub_classification: DataFrame with hub types
        """
        self._log("\n=== Phase 3: Yield and Hub Metrics (M1 + M3) ===")
        start = time.time()
        
        # A4: Yield per centroid
        yield_df = self._compute_yield_per_centroid()
        
        # B4: Hub classification (requires yield and sel_freq)
        sel_freq = self.m1.groupby('centroid_id').size().reindex(
            range(self.num_centroids), fill_value=0
        )
        hub_df = self._compute_hub_classification(sel_freq, yield_df['yield'])
        
        self._log(f"  Phase 3 completed in {time.time() - start:.1f}s")
        
        return {
            'yield_per_centroid': yield_df,
            'hub_classification': hub_df,
        }
    
    def _compute_yield_per_centroid(self) -> pd.DataFrame:
        """
        Compute A4: Yield per centroid.
        
        Yield = influential interactions / computed similarities
        
        M3 stores winner_embedding_pos, which needs conversion to centroid_id.
        """
        self._log("  Computing A4: Yield per centroid...")
        
        # M1: Total computations per centroid
        computed = self.m1.groupby('centroid_id')['num_token_token_sims'].sum()
        
        # M3: Derive winner_centroid_id from embedding_pos
        winner_positions = torch.tensor(self.m3['winner_embedding_pos'].values, dtype=torch.long)
        winner_centroids = embedding_pos_to_centroid(winner_positions, self.offsets)
        
        # Count influential interactions per centroid
        influential = pd.Series(winner_centroids.numpy()).value_counts()
        
        # Build result DataFrame
        result = pd.DataFrame({
            'centroid_id': range(self.num_centroids),
            'computed': computed.reindex(range(self.num_centroids), fill_value=0).values,
            'influential': influential.reindex(range(self.num_centroids), fill_value=0).values,
        })
        
        # Compute yield (NaN for never-selected centroids)
        result['yield'] = np.where(
            result['computed'] > 0,
            result['influential'] / result['computed'],
            np.nan
        )
        
        # Statistics
        selected = result[result['computed'] > 0]
        pure_waste = selected[selected['influential'] == 0]
        
        self._log(f"    Mean yield (selected): {selected['yield'].mean():.4f}")
        self._log(f"    Pure waste centroids: {len(pure_waste):,} / {len(selected):,}")
        
        return result
    
    def _compute_hub_classification(
        self,
        sel_freq: pd.Series,
        yield_per_centroid: pd.Series
    ) -> pd.DataFrame:
        """
        Compute B4: Hub classification.
        
        Classifies centroids as: normal, hub, good_hub, bad_hub
        """
        self._log("  Computing B4: Hub classification...")
        
        config = self.config
        
        total_traffic = sel_freq.sum()
        traffic_share = sel_freq / total_traffic if total_traffic > 0 else sel_freq * 0
        
        df = pd.DataFrame({
            'centroid_id': range(self.num_centroids),
            'sel_freq': sel_freq.values,
            'traffic_share': traffic_share.values,
            'yield': yield_per_centroid.values,
        })
        
        # Bad hubness: high traffic × low yield
        df['bad_hubness'] = df['traffic_share'] * (1 - df['yield'].fillna(0))
        
        # Good hubness: high traffic × high yield
        df['good_hubness'] = df['traffic_share'] * df['yield'].fillna(0)
        
        # Classification
        df['hub_type'] = 'normal'
        hub_threshold = df['traffic_share'].quantile(config.hub_percentile / 100)
        
        is_hub = df['traffic_share'] > hub_threshold
        has_yield = df['yield'].notna()
        
        df.loc[is_hub, 'hub_type'] = 'hub'
        df.loc[is_hub & has_yield & (df['yield'] < config.bad_yield_threshold), 'hub_type'] = 'bad_hub'
        df.loc[is_hub & has_yield & (df['yield'] > config.good_yield_threshold), 'hub_type'] = 'good_hub'
        
        # Statistics
        hub_counts = df['hub_type'].value_counts()
        self._log(f"    Hub counts: {hub_counts.to_dict()}")
        
        return df
    
    # =========================================================================
    # Phase 4: Oracle-Based Metrics (M4) - Heavy, Parallelized
    # =========================================================================
    
    def compute_phase4_oracle_metrics(self, parallel: bool = True, use_chunked: bool = True) -> Dict[str, Any]:
        """
        Compute Phase 4 metrics: C3/C5 (oracle evidence recall), C6 (evidence dispersion).
        
        C3/C5 "Oracle Evidence Recall": How much of the oracle-best evidence
        appears in the actual top-k ranking? (Renamed from "pruning recall")
        
        These operate on M4 data which is large (~96K rows per query).
        For large M4 files (>1GB), uses chunked processing to avoid OOM.
        
        Args:
            parallel: Whether to parallelize across queries
            use_chunked: Whether to use chunked processing for large M4 files
            
        Returns:
            Dictionary with:
            - oracle_evidence_recall: DataFrame with recall@k per query (C3/C5)
            - evidence_dispersion_summary: Summary stats for C6
        """
        self._log("\n=== Phase 4: Oracle Metrics (M4) ===")
        start = time.time()
        
        # Check if we should use chunked processing
        should_chunk = use_chunked and self._should_use_chunked_m4()
        
        if should_chunk:
            m4_path = self._get_m4_path()
            size_mb = m4_path.stat().st_size / (1024 * 1024)
            self._log(f"  M4 file is {size_mb:.0f} MB - using chunked processing")
            
            # C3/C5: Oracle evidence recall (chunked)
            recall_df = self._compute_oracle_evidence_recall_chunked()
            
            # C6: Evidence dispersion (chunked)
            dispersion_summary = self._compute_evidence_dispersion_chunked()
        else:
            # Original non-chunked path
            # C3/C5: Oracle evidence recall (renamed from pruning recall)
            recall_df = self._compute_oracle_evidence_recall(parallel=parallel)
            
            # C6: Evidence dispersion (can be large, compute summary only)
            dispersion_summary = self._compute_evidence_dispersion_summary()
        
        self._log(f"  Phase 4 completed in {time.time() - start:.1f}s")
        
        return {
            'oracle_evidence_recall': recall_df,
            'evidence_dispersion_summary': dispersion_summary,
        }
    
    def _compute_oracle_evidence_recall(self, parallel: bool = True) -> pd.DataFrame:
        """
        Compute C3/C5: Oracle evidence recall at ranking level.
        
        (Renamed from "pruning recall" for clarity)
        
        Measures how much of the oracle-best evidence appears in the actual top-k.
        - Oracle ranking: sum of M4 oracle MaxSim scores per document
        - Actual ranking: sum of M3 observed MaxSim scores per document
        
        Args:
            parallel: Whether to parallelize across queries (recommended for large runs)
        """
        self._log("  Computing C3/C5: Oracle evidence recall...")
        
        k_values = self.config.recall_k_values
        num_workers = self.config.num_workers
        
        # Compute actual document scores (sum of observed MaxSim)
        actual_scores = self.m3.groupby(['query_id', 'doc_id'])['winner_score'].sum().reset_index()
        actual_scores.columns = ['query_id', 'doc_id', 'actual_score']
        
        # Compute oracle document scores (sum of oracle MaxSim)
        oracle_scores = self.m4.groupby(['query_id', 'doc_id'])['oracle_score'].sum().reset_index()
        oracle_scores.columns = ['query_id', 'doc_id', 'oracle_score']
        
        query_ids = actual_scores['query_id'].unique()
        
        # Pre-split data by query for parallel processing
        actual_by_query = {qid: actual_scores[actual_scores['query_id'] == qid] 
                          for qid in query_ids}
        oracle_by_query = {qid: oracle_scores[oracle_scores['query_id'] == qid] 
                          for qid in query_ids}
        
        if parallel and len(query_ids) > 10:
            self._log(f"    Parallelizing across {len(query_ids)} queries with {num_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _compute_pruning_recall_single_query,
                        qid, actual_by_query[qid], oracle_by_query[qid], k_values
                    )
                    for qid in query_ids
                ]
                results = [f.result() for f in futures]
        else:
            results = [
                _compute_pruning_recall_single_query(
                    qid, actual_by_query[qid], oracle_by_query[qid], k_values
                )
                for qid in query_ids
            ]
        
        df = pd.DataFrame(results)
        
        for k in k_values:
            col = f'recall@{k}'
            if col in df.columns:
                self._log(f"    {col}: mean={df[col].mean():.3f}")
        
        return df
    
    def _compute_evidence_dispersion_summary(self) -> Dict[str, float]:
        """
        Compute C6: Evidence dispersion summary statistics.
        
        Full per-doc dispersion is expensive to store. Return summary only.
        """
        self._log("  Computing C6: Evidence dispersion (summary)...")
        
        # Derive oracle centroid from embedding position
        oracle_positions = torch.tensor(self.m4['oracle_embedding_pos'].values, dtype=torch.long)
        oracle_centroids = embedding_pos_to_centroid(oracle_positions, self.offsets)
        
        m4_with_centroid = self.m4.copy()
        m4_with_centroid['oracle_centroid_id'] = oracle_centroids.numpy()
        
        # Count unique centroids per (query, doc)
        unique_centroids_per_doc = m4_with_centroid.groupby(
            ['query_id', 'doc_id']
        )['oracle_centroid_id'].nunique()
        
        return {
            'mean_unique_centroids': unique_centroids_per_doc.mean(),
            'median_unique_centroids': unique_centroids_per_doc.median(),
            'max_unique_centroids': unique_centroids_per_doc.max(),
            'docs_with_1_centroid': (unique_centroids_per_doc == 1).sum(),
            'docs_with_multi_centroid': (unique_centroids_per_doc > 1).sum(),
        }
    
    def _compute_oracle_evidence_recall_chunked(self) -> pd.DataFrame:
        """
        Compute C3/C5: Oracle evidence recall using chunked M4 processing.
        
        Processes M4 in query batches to avoid OOM errors for large files.
        """
        self._log("  Computing C3/C5: Oracle evidence recall (chunked)...")
        
        k_values = self.config.recall_k_values
        m4_path = self._get_m4_path()
        
        # Load M3 once (should be smaller than M4)
        m3_path = self.tier_b_dir / "M3_observed_winners.parquet"
        m3 = pd.read_parquet(m3_path)
        self._log(f"    Loaded M3: {len(m3):,} rows")
        
        # Compute actual document scores from M3 (this is fixed, doesn't depend on M4 chunks)
        actual_scores = m3.groupby(['query_id', 'doc_id'])['winner_score'].sum().reset_index()
        actual_scores.columns = ['query_id', 'doc_id', 'actual_score']
        
        # Create chunked processor
        processor = ChunkedM4Processor(
            m4_path,
            chunk_size=self.config.m4_chunk_size,
            verbose=self.verbose
        )
        
        info = processor.get_file_info()
        self._log(f"    Processing {info['num_queries']:,} queries in {info['num_chunks']} chunks")
        
        # Accumulate oracle scores per (query, doc) across chunks
        oracle_scores_accumulated = {}  # (query_id, doc_id) -> total_oracle_score
        
        for m4_chunk in processor.iter_chunks(
            columns=['query_id', 'doc_id', 'oracle_score'],
            show_progress=self.verbose
        ):
            # Aggregate chunk
            chunk_scores = m4_chunk.groupby(['query_id', 'doc_id'])['oracle_score'].sum()
            
            # Accumulate into global dict
            for idx, score in chunk_scores.items():
                key = idx  # idx is (query_id, doc_id) tuple
                if key in oracle_scores_accumulated:
                    oracle_scores_accumulated[key] += score
                else:
                    oracle_scores_accumulated[key] = score
        
        # Convert accumulated scores to DataFrame
        oracle_scores = pd.DataFrame([
            {'query_id': k[0], 'doc_id': k[1], 'oracle_score': v}
            for k, v in oracle_scores_accumulated.items()
        ])
        
        self._log(f"    Accumulated oracle scores for {len(oracle_scores):,} (query, doc) pairs")
        
        # Now compute recall per query
        query_ids = actual_scores['query_id'].unique()
        
        actual_by_query = {qid: actual_scores[actual_scores['query_id'] == qid] 
                          for qid in query_ids}
        oracle_by_query = {qid: oracle_scores[oracle_scores['query_id'] == qid] 
                          for qid in query_ids}
        
        results = []
        for qid in tqdm(query_ids, desc="Computing recall@k", disable=not self.verbose):
            if qid in oracle_by_query:
                row = _compute_pruning_recall_single_query(
                    qid, actual_by_query[qid], oracle_by_query[qid], k_values
                )
                results.append(row)
        
        df = pd.DataFrame(results)
        
        for k in k_values:
            col = f'recall@{k}'
            if col in df.columns:
                self._log(f"    {col}: mean={df[col].mean():.3f}")
        
        return df
    
    def _compute_evidence_dispersion_chunked(self) -> Dict[str, float]:
        """
        Compute C6: Evidence dispersion using chunked M4 processing.
        
        Memory-optimized approach: Since each chunk contains complete queries
        (ChunkedM4Processor partitions by query_id), all rows for any (query_id, doc_id)
        pair are fully contained within a single chunk. This means we can compute
        unique centroid counts directly within each chunk using nunique(), without
        accumulating sets across chunks.
        
        Memory usage: O(total_pairs) integers (~750 MB) instead of O(total_pairs * avg_centroids) 
        in sets (~70 GB).
        """
        self._log("  Computing C6: Evidence dispersion (chunked, memory-optimized)...")
        
        m4_path = self._get_m4_path()
        
        # Create chunked processor
        processor = ChunkedM4Processor(
            m4_path,
            chunk_size=self.config.m4_chunk_size,
            verbose=self.verbose
        )
        
        # Collect unique centroid counts per (query, doc) - just integers, not sets
        # Since chunks contain complete queries, we can compute nunique() within each chunk
        all_unique_counts = []
        
        for m4_chunk in processor.iter_chunks(
            columns=['query_id', 'doc_id', 'oracle_embedding_pos'],
            show_progress=self.verbose
        ):
            # Compute centroid IDs for this chunk
            oracle_positions = torch.tensor(m4_chunk['oracle_embedding_pos'].values, dtype=torch.long)
            oracle_centroids = embedding_pos_to_centroid(oracle_positions, self.offsets).numpy()
            
            m4_chunk = m4_chunk.copy()
            m4_chunk['oracle_centroid_id'] = oracle_centroids
            
            # Compute unique centroid count per (query, doc) WITHIN this chunk
            # This works because all rows for a (query, doc) pair are in the same chunk
            chunk_unique_counts = m4_chunk.groupby(['query_id', 'doc_id'])['oracle_centroid_id'].nunique()
            all_unique_counts.extend(chunk_unique_counts.values)
            
            # Free memory
            del m4_chunk, oracle_positions, oracle_centroids, chunk_unique_counts
        
        # Convert to numpy array for efficient statistics
        unique_counts = np.array(all_unique_counts, dtype=np.int32)
        self._log(f"    Computed dispersion for {len(unique_counts):,} (query, doc) pairs")
        
        return {
            'mean_unique_centroids': float(unique_counts.mean()),
            'median_unique_centroids': float(np.median(unique_counts)),
            'max_unique_centroids': int(unique_counts.max()),
            'docs_with_1_centroid': int((unique_counts == 1).sum()),
            'docs_with_multi_centroid': int((unique_counts > 1).sum()),
        }
    
    # =========================================================================
    # Phase 5: Routing Fidelity (M4 + R0) - Heavy, Parallelized
    # =========================================================================
    
    def compute_phase5_routing_fidelity(self) -> Dict[str, Any]:
        """
        Compute Phase 5 metrics: C4/C5 (routing fidelity).
        
        C4: Oracle hit rate (oracle centroid was selected)
        C5: Oracle miss rate (oracle centroid was NOT selected)
        
        Requires R0 with centroid_score for full analysis.
        
        Returns:
            Dictionary with routing fidelity metrics
        """
        self._log("\n=== Phase 5: Routing Fidelity (M4 + R0) ===")
        start = time.time()
        
        # Check if R0 has centroid_score
        has_score = 'centroid_score' in self.r0.columns
        if not has_score:
            warnings.warn("R0 missing centroid_score - C4/C5 analysis limited")
        
        fidelity = self._compute_routing_fidelity()
        
        self._log(f"  Phase 5 completed in {time.time() - start:.1f}s")
        
        return fidelity
    
    def _compute_routing_fidelity(self, parallel: bool = True) -> Dict[str, Any]:
        """
        Compute routing fidelity: how often is oracle centroid in selected set?
        
        Args:
            parallel: Whether to parallelize across queries (recommended for large runs)
        """
        self._log("  Computing C4/C5: Routing fidelity...")
        
        num_workers = self.config.num_workers
        
        # Pre-compute offsets as numpy for parallel workers
        offsets_np = self.offsets.numpy()
        
        # Build selected centroid sets per (query, token)
        r0_grouped = self.r0.groupby('query_id')
        
        query_ids = self.m4['query_id'].unique()
        
        # Pre-split data by query
        m4_by_query = {qid: self.m4[self.m4['query_id'] == qid] for qid in query_ids}
        
        # Build R0 sets by query -> {token_id: set(centroid_ids)}
        r0_sets_by_query = {}
        for qid in query_ids:
            r0_q = self.r0[self.r0['query_id'] == qid]
            r0_sets_by_query[qid] = r0_q.groupby('q_token_id')['centroid_id'].apply(set).to_dict()
        
        if parallel and len(query_ids) > 10:
            self._log(f"    Parallelizing across {len(query_ids)} queries with {num_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _compute_routing_fidelity_single_query,
                        qid, m4_by_query[qid], r0_sets_by_query.get(qid, {}), offsets_np
                    )
                    for qid in query_ids
                ]
                results = [f.result() for f in futures]
        else:
            results = [
                _compute_routing_fidelity_single_query(
                    qid, m4_by_query[qid], r0_sets_by_query.get(qid, {}), offsets_np
                )
                for qid in query_ids
            ]
        
        # Aggregate results
        total_hits = sum(r['hits'] for r in results)
        total_misses = sum(r['misses'] for r in results)
        total = total_hits + total_misses
        
        hit_rate = total_hits / total if total > 0 else 0
        miss_rate = total_misses / total if total > 0 else 0
        
        self._log(f"    Hit rate: {hit_rate:.1%} ({total_hits:,} / {total:,})")
        self._log(f"    Miss rate: {miss_rate:.1%} ({total_misses:,} / {total:,})")
        
        return {
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'total_interactions': total,
            'hits': total_hits,
            'misses': total_misses,
        }
    
    # =========================================================================
    # Compute All and Save
    # =========================================================================
    
    def compute_all(self, save: bool = True) -> Dict[str, Any]:
        """
        Compute all online cluster property metrics.
        
        Args:
            save: Whether to save results to output directory
            
        Returns:
            Dictionary with all computed metrics
        """
        self._log(f"\n{'='*70}")
        self._log("Online Cluster Properties Computation")
        self._log(f"{'='*70}")
        self._log(f"Run dir: {self.run_dir}")
        self._log(f"Index: {self.index_path}")
        self._log(f"Output: {self.output_dir}")
        
        start = time.time()
        
        results = {}
        
        # Phase 2: Core metrics (M1 only)
        results['phase2'] = self.compute_phase2_core_metrics()
        
        # Phase 3: Yield and hub classification (M1 + M3)
        results['phase3'] = self.compute_phase3_yield_metrics()
        
        # Phase 4: Oracle metrics (M4) - C3 renamed to "oracle_evidence_recall"
        results['phase4'] = self.compute_phase4_oracle_metrics()
        
        # Phase 5: Routing fidelity (M4 + R0) - C4 hit/miss rate
        # Note: Can be skipped via config.skip_c4_routing_fidelity (redundant with M5)
        if not self.config.skip_c4_routing_fidelity:
            results['phase5'] = self.compute_phase5_routing_fidelity()
        else:
            self._log("\n=== Phase 5: Routing Fidelity (SKIPPED - redundant with M5) ===")
            results['phase5'] = {'skipped': True, 'reason': 'redundant with M5'}
        
        total_time = time.time() - start
        self._log(f"\n{'='*70}")
        self._log(f"Total time: {total_time:.1f}s")
        
        if save:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save computed results to output directory."""
        self._log(f"\nSaving results to {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Centroid aggregates (merge yield and hub classification)
        yield_df = results['phase3']['yield_per_centroid']
        hub_df = results['phase3']['hub_classification']
        
        centroid_agg = yield_df.merge(
            hub_df[['centroid_id', 'traffic_share', 'bad_hubness', 'good_hubness', 'hub_type']],
            on='centroid_id'
        )
        centroid_agg['sel_freq'] = results['phase2']['sel_freq'].values
        
        centroid_agg.to_parquet(self.output_dir / "centroid_aggregates.parquet", index=False)
        self._log(f"  Saved centroid_aggregates.parquet ({len(centroid_agg):,} rows)")
        
        # Per-query metrics (merge activation and recall)
        per_query = results['phase2']['per_query_activation']
        recall_df = results['phase4']['oracle_evidence_recall']
        
        per_query_full = per_query.merge(recall_df, on='query_id', how='left')
        per_query_full.to_parquet(self.output_dir / "per_query_metrics.parquet", index=False)
        self._log(f"  Saved per_query_metrics.parquet ({len(per_query_full):,} rows)")
        
        # Global summary
        summary = {
            'traffic_concentration': results['phase2']['traffic_concentration'],
            'anti_hub': {
                'rate': float(results['phase2']['anti_hub']['rate']),
                'count': int(results['phase2']['anti_hub']['count']),
            },
            'yield_summary': {
                'mean_yield': float(yield_df[yield_df['computed'] > 0]['yield'].mean()),
                'median_yield': float(yield_df[yield_df['computed'] > 0]['yield'].median()),
                'pure_waste_centroids': int((yield_df['influential'] == 0).sum()),
            },
            'hub_classification': {k: int(v) for k, v in hub_df['hub_type'].value_counts().to_dict().items()},
            'routing_fidelity': results['phase5'] if results['phase5'].get('skipped') else {
                k: float(v) if isinstance(v, (float, np.floating)) else int(v) 
                for k, v in results['phase5'].items()
            },
            'evidence_dispersion': {k: float(v) for k, v in results['phase4']['evidence_dispersion_summary'].items()},
            'oracle_evidence_recall': {  # Renamed from pruning_recall (C3/C5)
                f'mean_recall@{k}': float(recall_df[f'recall@{k}'].mean())
                for k in self.config.recall_k_values
                if f'recall@{k}' in recall_df.columns
            },
        }
        
        with open(self.output_dir / "global_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        self._log(f"  Saved global_summary.json")
    
    def print_summary(self) -> None:
        """Print summary of existing or computed metrics."""
        summary_path = self.output_dir / "global_summary.json"
        
        if not summary_path.exists():
            self._log("No summary found. Run compute_all() first.")
            return
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        self._log(f"\n{'='*70}")
        self._log("Online Cluster Properties Summary")
        self._log(f"{'='*70}")
        
        self._log("\nTraffic Concentration:")
        tc = summary.get('traffic_concentration', {})
        self._log(f"  Gini: {tc.get('gini', 'N/A'):.3f}")
        self._log(f"  Top 5% share: {tc.get('top_5pct_share', 'N/A'):.1%}")
        
        self._log("\nAnti-Hub:")
        ah = summary.get('anti_hub', {})
        self._log(f"  Rate: {ah.get('rate', 'N/A'):.1%}")
        self._log(f"  Count: {ah.get('count', 'N/A'):,}")
        
        self._log("\nYield:")
        ys = summary.get('yield_summary', {})
        self._log(f"  Mean: {ys.get('mean_yield', 'N/A'):.4f}")
        self._log(f"  Pure waste centroids: {ys.get('pure_waste_centroids', 'N/A'):,}")
        
        self._log("\nRouting Fidelity (C4):")
        rf = summary.get('routing_fidelity', {})
        if rf.get('skipped'):
            self._log(f"  (Skipped - {rf.get('reason', 'N/A')})")
        else:
            self._log(f"  Hit rate: {rf.get('hit_rate', 'N/A'):.1%}")
            self._log(f"  Miss rate: {rf.get('miss_rate', 'N/A'):.1%}")
        
        self._log("\nOracle Evidence Recall (C3/C5):")
        pr = summary.get('oracle_evidence_recall', summary.get('pruning_recall', {}))  # Backward compat
        for k, v in pr.items():
            self._log(f"  {k}: {v:.3f}")
