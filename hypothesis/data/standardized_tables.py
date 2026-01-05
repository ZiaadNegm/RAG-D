"""
Standardized Tables Builder

Creates joined/aggregated tables that are reused across multiple hypotheses,
avoiding redundant computation and ensuring consistency.

Primary Data Products:
    1. cluster_frame: One row per centroid (main workhorse for design insights)
       - Joins offline A-series, B-series, routing C-series, M-series aggregates
       
    2. query_frame: One row per query with routing/miss summaries
       - For query-level analysis and stratification
       
    3. miss_attribution_frame: Misses attributed to centroid properties
       - For hypothesis tests about what causes misses

These tables are cached to disk to avoid recomputation across hypothesis runs.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from hypothesis.configs import RuntimeConfig, load_config, ensure_output_dirs
from hypothesis.data import MetricsLoader


class ClusterFrameBuilder:
    """
    Builds the cluster_frame: one row per centroid with all metrics joined.
    
    This is the main workhorse table for hypothesis testing about cluster
    properties and their relationship to routing behavior.
    
    Columns (organized by metric series):
    
    A-series (Offline Structure):
        - centroid_id: Unique identifier
        - n_tokens (A1): Number of token embeddings
        - n_docs (A2): Number of unique documents
        - top_1_doc_share, top_5_doc_share, top_10_doc_share (A3): Concentration
        - gini_coefficient (A3): Document concentration Gini
        - dispersion (A5): Within-centroid dispersion (quantization tightness)
        - tokens_per_doc: Average tokens per doc
        
    B-series (Hubness/Geometry):
        - isolation (B5): Inter-centroid isolation
        - nearest_neighbor_sim (B5): Similarity to nearest centroid
        - mean_neighbor_sim (B5): Mean k-NN similarity
        - sel_freq (B1): Selection frequency (hub occurrence)
        - traffic_share (B2): Fraction of total traffic
        - hub_type (B4): Classification (good_hub/bad_hub/normal)
        - bad_hubness (B4): Traffic × (1 - yield)
        - good_hubness (B4): Traffic × yield
        
    Routing/Yield (A4, C-series aggregated):
        - computed: Total token-token similarities computed
        - influential: Similarities that won MaxSim
        - yield (A4): influential / computed
        
    M-series Aggregated (per-centroid summaries):
        - m6_miss_count: Number of misses attributed to this centroid
        - m6_oracle_win_count: Number of oracle wins from this centroid
        - m6_miss_rate: Miss rate for this centroid
        - m1_total_sims: Total computation attributed to this centroid
        - m2_redundant_sims: Redundant computation (M1 - influential)
        - redundancy_rate: m2_redundant_sims / m1_total_sims
        
    Derived:
        - size_bin: Quartile bin by n_tokens (Q1, Q2, Q3, Q4)
        - dispersion_bin: Quartile bin by dispersion
        - hubness_bin: Quartile bin by sel_freq
        - log_n_tokens: log10(n_tokens + 1)
        - is_anti_hub: sel_freq == 0
    """
    
    def __init__(
        self, 
        config: RuntimeConfig,
        loader: Optional[MetricsLoader] = None
    ):
        """
        Initialize cluster frame builder.
        
        Args:
            config: Runtime configuration
            loader: Optional pre-configured MetricsLoader
        """
        self.config = config
        self.loader = loader or MetricsLoader(
            index_path=config.paths.index_path,
            run_dir=config.paths.run_dir,
            chunk_size=config.processing.m4_chunk_size,
            verbose=config.verbose
        )
        
        self._cache_path = Path(config.processing.cache_dir) / "cluster_frame.parquet"
    
    def build(self, force_rebuild: bool = False) -> pd.DataFrame:
        """
        Build or load cached cluster_frame.
        
        Args:
            force_rebuild: If True, rebuild even if cache exists
            
        Returns:
            DataFrame with one row per centroid
        """
        # Check cache
        if not force_rebuild and self._cache_path.exists() and self.config.processing.cache_standardized_tables:
            print(f"Loading cached cluster_frame from {self._cache_path}")
            return pd.read_parquet(self._cache_path)
        
        print("Building cluster_frame...")
        
        # Load component data
        offline = self.loader.load_offline_properties()
        
        # Try to load online aggregates (may not exist for all runs)
        try:
            online = self.loader.load_online_centroid_aggregates()
            has_online = True
        except FileNotFoundError:
            warnings.warn("Online centroid aggregates not found, building from raw metrics")
            online = self._build_online_from_raw()
            has_online = online is not None
        
        # Load M6 global for miss attribution
        try:
            m6_global = self.loader.load_m6_global()
            has_m6 = True
        except FileNotFoundError:
            warnings.warn("M6 global not found, miss metrics will be missing")
            m6_global = None
            has_m6 = False
        
        # Start with offline as base
        df = offline.copy()
        
        # Join online metrics
        if has_online:
            # Ensure consistent naming
            online = online.rename(columns={'centroid_id': 'centroid_id'})
            df = df.merge(online, on='centroid_id', how='left')
        else:
            # Add placeholder columns
            for col in ['sel_freq', 'computed', 'influential', 'yield', 
                       'traffic_share', 'bad_hubness', 'good_hubness', 'hub_type']:
                df[col] = np.nan
        
        # Join M6 metrics
        if has_m6:
            m6_renamed = m6_global.rename(columns={
                'oracle_centroid_id': 'centroid_id',
                'miss_count': 'm6_miss_count',
                'oracle_win_count': 'm6_oracle_win_count',
                'miss_rate': 'm6_miss_rate'
            })
            df = df.merge(m6_renamed, on='centroid_id', how='left')
        else:
            for col in ['m6_miss_count', 'm6_oracle_win_count', 'm6_miss_rate']:
                df[col] = np.nan
        
        # Compute M1/M2 per-centroid aggregates
        try:
            m1 = self.loader.load_m1()
            m1_agg = m1.groupby('centroid_id').agg({
                'num_token_token_sims': 'sum'
            }).reset_index()
            m1_agg.columns = ['centroid_id', 'm1_total_sims']
            df = df.merge(m1_agg, on='centroid_id', how='left')
            
            # Compute redundant sims
            df['m2_redundant_sims'] = df['m1_total_sims'] - df['influential'].fillna(0)
            df['redundancy_rate'] = np.where(
                df['m1_total_sims'] > 0,
                df['m2_redundant_sims'] / df['m1_total_sims'],
                0
            )
        except FileNotFoundError:
            warnings.warn("M1 not found, computation metrics will be missing")
            df['m1_total_sims'] = np.nan
            df['m2_redundant_sims'] = np.nan
            df['redundancy_rate'] = np.nan
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        # Fill NaN in categorical columns
        df['hub_type'] = df['hub_type'].fillna('unknown')
        
        # Cache result
        if self.config.processing.cache_standardized_tables:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_path, index=False)
            print(f"Cached cluster_frame to {self._cache_path}")
        
        print(f"Built cluster_frame: {df.shape}")
        return df
    
    def _build_online_from_raw(self) -> Optional[pd.DataFrame]:
        """Build online centroid aggregates from raw M1/M3 if precomputed not available."""
        try:
            m1 = self.loader.load_m1()
            m3 = self.loader.load_m3()
        except FileNotFoundError:
            return None
        
        # M1: total computation per centroid
        m1_agg = m1.groupby('centroid_id').agg({
            'num_token_token_sims': 'sum'
        }).reset_index()
        m1_agg.columns = ['centroid_id', 'computed']
        
        # Selection frequency from M1 (count of times selected)
        sel_freq = m1.groupby('centroid_id').size().reset_index(name='sel_freq')
        m1_agg = m1_agg.merge(sel_freq, on='centroid_id', how='left')
        
        # For M3, we need to map winners back to centroids via offsets
        # This requires index data - simplify by counting per centroid in M1
        # (M3 influential is already in M1's centroids)
        
        # Use a simplified yield estimate
        m1_agg['influential'] = 0  # Placeholder - would need proper M3 join
        m1_agg['yield'] = 0.0
        
        # Traffic share
        total_traffic = m1_agg['sel_freq'].sum()
        m1_agg['traffic_share'] = m1_agg['sel_freq'] / total_traffic if total_traffic > 0 else 0
        
        # Hub metrics (simplified)
        m1_agg['bad_hubness'] = m1_agg['traffic_share'] * (1 - m1_agg['yield'])
        m1_agg['good_hubness'] = m1_agg['traffic_share'] * m1_agg['yield']
        m1_agg['hub_type'] = 'normal'
        
        return m1_agg
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for analysis convenience."""
        # Log transform for skewed size
        df['log_n_tokens'] = np.log10(df['n_tokens'].clip(lower=1))
        
        # Quartile bins - use try/except to handle edge cases with duplicates
        df['size_bin'] = self._safe_qcut(
            df['n_tokens'], 
            labels=['Q1_small', 'Q2_medium', 'Q3_large', 'Q4_huge']
        )
        
        if 'dispersion' in df.columns and df['dispersion'].notna().any():
            df['dispersion_bin'] = self._safe_qcut(
                df['dispersion'].fillna(df['dispersion'].median()),
                labels=['Q1_tight', 'Q2_moderate', 'Q3_loose', 'Q4_dispersed']
            )
        else:
            df['dispersion_bin'] = 'unknown'
        
        if 'sel_freq' in df.columns and df['sel_freq'].notna().any():
            df['hubness_bin'] = self._safe_qcut(
                df['sel_freq'].fillna(0) + 1,  # +1 to handle zeros
                labels=['Q1_cold', 'Q2_warm', 'Q3_hot', 'Q4_hub']
            )
            df['is_anti_hub'] = df['sel_freq'] == 0
        else:
            df['hubness_bin'] = 'unknown'
            df['is_anti_hub'] = False
        
        # Concentration categories
        if 'top_1_doc_share' in df.columns:
            df['is_single_doc_dominated'] = df['top_1_doc_share'] > 0.5
            df['is_concentrated'] = df['top_5_doc_share'] > 0.8
        
        return df
    
    def _safe_qcut(
        self, 
        series: pd.Series, 
        labels: List[str],
        q: int = 4
    ) -> pd.Series:
        """
        Safe quantile cut that handles duplicate bin edges.
        
        When duplicates='drop' results in fewer bins than labels, 
        this falls back to pd.cut with custom percentile-based bins.
        """
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates='drop')
        except ValueError:
            # Fallback: use percentile-based cut
            try:
                # Try qcut without labels first to see how many bins we get
                bins = pd.qcut(series, q=q, duplicates='drop', retbins=True)[1]
                n_bins = len(bins) - 1
                # Use subset of labels matching actual bins
                return pd.cut(series, bins=bins, labels=labels[:n_bins], include_lowest=True)
            except Exception:
                # Ultimate fallback: return 'unknown' 
                return pd.Series(['unknown'] * len(series), index=series.index)


class QueryFrameBuilder:
    """
    Builds the query_frame: one row per query with routing/miss summaries.
    
    Useful for query-level analysis and stratification.
    
    Columns:
        Query ID:
            - query_id: Unique identifier
            
        Routing Metrics (C-series):
            - unique_centroids (C1): Number of centroids activated
            - activation_entropy (C1): Entropy of centroid distribution
            - load_gini (C2): Load imbalance
            - top_5_centroid_share (C2): Top-5 centroid traffic share
            
        Recall (C3):
            - recall@10, recall@100, recall@1000: Pruning recall
            
        Computation (M1/M2):
            - m1_total_sims: Total token-token sims
            - m2_redundant_sims: Redundant sims
            - redundancy_rate: M2/M1
            
        Misses (M5/M6):
            - total_misses: Number of routing misses
            - miss_rate: Fraction of oracle winners that were misses
            - mean_score_delta: Average severity of misses
            - max_score_delta: Worst miss severity
    """
    
    def __init__(
        self, 
        config: RuntimeConfig,
        loader: Optional[MetricsLoader] = None
    ):
        self.config = config
        self.loader = loader or MetricsLoader(
            index_path=config.paths.index_path,
            run_dir=config.paths.run_dir,
            chunk_size=config.processing.m4_chunk_size,
            verbose=config.verbose
        )
        
        self._cache_path = Path(config.processing.cache_dir) / "query_frame.parquet"
    
    def build(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Build or load cached query_frame."""
        if not force_rebuild and self._cache_path.exists() and self.config.processing.cache_standardized_tables:
            print(f"Loading cached query_frame from {self._cache_path}")
            return pd.read_parquet(self._cache_path)
        
        print("Building query_frame...")
        
        # Start with online per-query metrics as base
        try:
            df = self.loader.load_online_per_query_metrics()
        except FileNotFoundError:
            warnings.warn("Online per-query metrics not found, building from raw")
            df = self._build_from_raw()
        
        # Add M2 redundancy metrics
        try:
            m2 = self.loader.load_m2()
            df = df.merge(m2, on='query_id', how='left')
        except FileNotFoundError:
            pass
        
        # Add M5 miss summaries
        try:
            m5 = self.loader.load_m5()
            m5_agg = m5.groupby('query_id').agg({
                'is_miss': ['sum', 'mean'],
                'score_delta': ['mean', 'max', 'sum']
            })
            m5_agg.columns = ['total_misses', 'miss_rate', 'mean_score_delta', 'max_score_delta', 'total_score_delta']
            m5_agg = m5_agg.reset_index()
            df = df.merge(m5_agg, on='query_id', how='left')
        except FileNotFoundError:
            pass
        
        # Cache
        if self.config.processing.cache_standardized_tables:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_path, index=False)
            print(f"Cached query_frame to {self._cache_path}")
        
        print(f"Built query_frame: {df.shape}")
        return df
    
    def _build_from_raw(self) -> pd.DataFrame:
        """Build basic query frame from raw M1."""
        m1 = self.loader.load_m1()
        
        # Aggregate per query
        df = m1.groupby('query_id').agg({
            'centroid_id': 'nunique',
            'num_token_token_sims': 'sum'
        }).reset_index()
        df.columns = ['query_id', 'unique_centroids', 'm1_total_sims']
        
        return df


class MissAttributionFrameBuilder:
    """
    Builds the miss_attribution_frame: M5 misses joined with centroid properties.
    
    This table enables hypothesis tests about what causes misses by attributing
    each miss to the oracle centroid's properties.
    
    Columns:
        Miss Info:
            - query_id, q_token_id, doc_id
            - oracle_centroid_id
            - is_miss
            - score_delta (severity)
            
        Oracle Centroid Properties (from cluster_frame):
            - oracle_n_tokens, oracle_dispersion, oracle_sel_freq, etc.
    """
    
    def __init__(
        self, 
        config: RuntimeConfig,
        loader: Optional[MetricsLoader] = None,
        cluster_frame: Optional[pd.DataFrame] = None
    ):
        self.config = config
        self.loader = loader or MetricsLoader(
            index_path=config.paths.index_path,
            run_dir=config.paths.run_dir,
            chunk_size=config.processing.m4_chunk_size,
            verbose=config.verbose
        )
        self._cluster_frame = cluster_frame
        self._cache_path = Path(config.processing.cache_dir) / "miss_attribution_frame.parquet"
    
    def build(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Build or load cached miss_attribution_frame."""
        if not force_rebuild and self._cache_path.exists() and self.config.processing.cache_standardized_tables:
            print(f"Loading cached miss_attribution_frame from {self._cache_path}")
            return pd.read_parquet(self._cache_path)
        
        print("Building miss_attribution_frame...")
        
        # Load M5
        m5 = self.loader.load_m5()
        
        # Get cluster_frame
        if self._cluster_frame is None:
            builder = ClusterFrameBuilder(self.config, self.loader)
            self._cluster_frame = builder.build()
        
        # Select columns to join
        centroid_cols = [
            'centroid_id', 'n_tokens', 'n_docs', 'dispersion', 
            'gini_coefficient', 'sel_freq', 'yield', 'hub_type',
            'size_bin', 'dispersion_bin', 'hubness_bin'
        ]
        
        # Filter to available columns
        available_cols = [c for c in centroid_cols if c in self._cluster_frame.columns]
        centroid_props = self._cluster_frame[available_cols].copy()
        
        # Rename for join
        centroid_props = centroid_props.rename(columns={
            col: f'oracle_{col}' if col != 'centroid_id' else col
            for col in centroid_props.columns
        })
        centroid_props = centroid_props.rename(columns={'centroid_id': 'oracle_centroid_id'})
        
        # Join
        df = m5.merge(centroid_props, on='oracle_centroid_id', how='left')
        
        # Cache
        if self.config.processing.cache_standardized_tables:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_path, index=False)
            print(f"Cached miss_attribution_frame to {self._cache_path}")
        
        print(f"Built miss_attribution_frame: {df.shape}")
        return df


def build_all_standardized_tables(
    config: RuntimeConfig,
    force_rebuild: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Build all standardized tables and return as dictionary.
    
    Args:
        config: Runtime configuration
        force_rebuild: If True, rebuild even if caches exist
        
    Returns:
        Dict with keys: cluster_frame, query_frame, miss_attribution_frame (if M5 available)
    """
    ensure_output_dirs(config)
    
    loader = MetricsLoader(
        index_path=config.paths.index_path,
        run_dir=config.paths.run_dir,
        chunk_size=config.processing.m4_chunk_size,
        verbose=config.verbose
    )
    
    # Build cluster_frame first (others may depend on it)
    cluster_builder = ClusterFrameBuilder(config, loader)
    cluster_frame = cluster_builder.build(force_rebuild=force_rebuild)
    
    # Build query_frame
    query_builder = QueryFrameBuilder(config, loader)
    query_frame = query_builder.build(force_rebuild=force_rebuild)
    
    result = {
        'cluster_frame': cluster_frame,
        'query_frame': query_frame,
    }
    
    # Build miss_attribution_frame (uses cluster_frame) - optional, requires M5
    try:
        miss_builder = MissAttributionFrameBuilder(config, loader, cluster_frame)
        miss_attribution_frame = miss_builder.build(force_rebuild=force_rebuild)
        result['miss_attribution_frame'] = miss_attribution_frame
    except FileNotFoundError:
        print("Note: miss_attribution_frame not built (M5 not available)")
        result['miss_attribution_frame'] = None
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build standardized tables for hypothesis testing")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev",
                       help="Configuration to use")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild even if cache exists")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Override run directory")
    args = parser.parse_args()
    
    config = load_config(args.config, override_run_dir=args.run_dir)
    tables = build_all_standardized_tables(config, force_rebuild=args.force_rebuild)
    
    print("\n" + "="*60)
    print("STANDARDIZED TABLES SUMMARY")
    print("="*60)
    for name, df in tables.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
