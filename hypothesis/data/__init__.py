"""
Data loading utilities with chunking and parallel fetching.

Handles memory-efficient loading of WARP metrics files, especially 
the large M4 oracle winners file which can exceed available RAM.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterator
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


class ChunkedParquetReader:
    """
    Memory-efficient Parquet reader using query-based chunking.
    
    Uses Parquet predicate pushdown to read only the required row groups
    for each batch of query IDs, avoiding loading entire files into memory.
    """
    
    def __init__(
        self,
        path: str,
        chunk_size: int = 500,
        query_column: str = "query_id",
        verbose: bool = True
    ):
        """
        Initialize chunked reader.
        
        Args:
            path: Path to Parquet file
            chunk_size: Number of queries per chunk
            query_column: Name of query ID column for filtering
            verbose: Show progress bars
        """
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.query_column = query_column
        self.verbose = verbose
        
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        self._query_ids: Optional[List[int]] = None
        self._total_rows: Optional[int] = None
    
    def get_query_ids(self) -> List[int]:
        """Get sorted list of unique query IDs (reads only query_id column)."""
        if self._query_ids is None:
            query_col = pd.read_parquet(
                self.path, 
                columns=[self.query_column]
            )[self.query_column]
            self._query_ids = sorted(query_col.unique().tolist())
            self._total_rows = len(query_col)
        return self._query_ids
    
    @property
    def num_queries(self) -> int:
        return len(self.get_query_ids())
    
    @property  
    def num_chunks(self) -> int:
        return (self.num_queries + self.chunk_size - 1) // self.chunk_size
    
    def iter_chunks(
        self, 
        query_subset: Optional[Set[int]] = None,
        columns: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over chunks of data.
        
        Args:
            query_subset: Optional set of query IDs to include
            columns: Optional list of columns to load
            
        Yields:
            DataFrame chunks with ~chunk_size queries each
        """
        query_ids = self.get_query_ids()
        
        if query_subset is not None:
            query_ids = [q for q in query_ids if q in query_subset]
        
        # Create chunks
        chunks = [
            query_ids[i:i + self.chunk_size]
            for i in range(0, len(query_ids), self.chunk_size)
        ]
        
        iterator = tqdm(chunks, desc=f"Reading {self.path.name}") if self.verbose else chunks
        
        for chunk_query_ids in iterator:
            # Use predicate pushdown for efficient filtering
            df = pd.read_parquet(
                self.path,
                columns=columns,
                filters=[(self.query_column, 'in', chunk_query_ids)]
            )
            yield df
    
    def read_full(
        self, 
        query_subset: Optional[Set[int]] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Read entire file (or subset) into memory.
        
        Warning: May cause OOM for large files. Use iter_chunks for large data.
        """
        if query_subset is not None:
            return pd.read_parquet(
                self.path,
                columns=columns,
                filters=[(self.query_column, 'in', list(query_subset))]
            )
        return pd.read_parquet(self.path, columns=columns)


class MetricsLoader:
    """
    Unified loader for all WARP metrics files.
    
    Provides convenient access to:
    - Offline cluster properties (A1-A5, B5)
    - Online cluster properties (A4, A6, B1-B4, C1-C6)
    - Raw metrics (M1, M3, M4, R0)
    - Derived metrics (M2, M5, M6)
    """
    
    def __init__(
        self,
        index_path: str,
        run_dir: str,
        chunk_size: int = 500,
        verbose: bool = True
    ):
        """
        Initialize metrics loader.
        
        Args:
            index_path: Path to WARP index directory
            run_dir: Path to measurement run directory
            chunk_size: Default chunk size for chunked reads
            verbose: Show progress info
        """
        self.index_path = Path(index_path)
        self.run_dir = Path(run_dir)
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        # Subdirectories
        self.tier_a_dir = self.run_dir / "tier_a"
        self.tier_b_dir = self.run_dir / "tier_b"
        self.online_props_dir = self.run_dir / "cluster_properties_online"
        
        # Cached data
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # Offline Cluster Properties (A-series static, B5)
    # =========================================================================
    
    def load_offline_properties(self) -> pd.DataFrame:
        """
        Load offline cluster properties (one row per centroid).
        
        Columns:
            - centroid_id: Centroid identifier
            - n_tokens (A1): Number of embeddings per centroid
            - n_docs (A2): Number of unique documents per centroid
            - top_1_doc_share, top_5_doc_share, top_10_doc_share (A3): Concentration
            - gini_coefficient (A3): Document concentration Gini
            - dispersion (A5): Within-centroid dispersion
            - isolation (B5): Inter-centroid isolation
            - nearest_neighbor_sim, mean_neighbor_sim (B5): Geometry
            - tokens_per_doc: Average tokens per doc in cluster
        """
        cache_key = "offline_properties"
        if cache_key not in self._cache:
            path = self.index_path / "cluster_properties_offline.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Offline properties not found: {path}")
            self._cache[cache_key] = pd.read_parquet(path)
            self._log(f"Loaded offline properties: {self._cache[cache_key].shape}")
        return self._cache[cache_key]
    
    # =========================================================================
    # Online Cluster Properties (A4, A6, B1-B4, C-series)
    # =========================================================================
    
    def load_online_centroid_aggregates(self) -> pd.DataFrame:
        """
        Load online per-centroid aggregates.
        
        Columns:
            - centroid_id: Centroid identifier
            - sel_freq (A6/B1): Selection frequency
            - computed: Total token-token sims computed in centroid
            - influential: Token-token sims that won MaxSim
            - yield (A4): influential / computed
            - traffic_share (B2): Fraction of total traffic
            - bad_hubness (B4): traffic_share * (1 - yield)
            - good_hubness (B4): traffic_share * yield
            - hub_type (B4): Classification (good_hub/bad_hub/normal)
        """
        cache_key = "online_centroid_aggregates"
        if cache_key not in self._cache:
            path = self.online_props_dir / "centroid_aggregates.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Online centroid aggregates not found: {path}")
            self._cache[cache_key] = pd.read_parquet(path)
            self._log(f"Loaded online centroid aggregates: {self._cache[cache_key].shape}")
        return self._cache[cache_key]
    
    def load_online_per_query_metrics(self) -> pd.DataFrame:
        """
        Load online per-query metrics.
        
        Columns:
            - query_id: Query identifier
            - unique_centroids: Number of unique centroids activated
            - activation_entropy (C1): Entropy of centroid activation
            - normalized_activation_entropy (C1): Normalized to [0,1]
            - load_cv, load_gini (C2): Load imbalance metrics
            - top_5_centroid_share (C2): Top-5 centroid traffic share
            - recall@10, recall@100, recall@1000 (C3): Pruning recall
        """
        cache_key = "online_per_query"
        if cache_key not in self._cache:
            path = self.online_props_dir / "per_query_metrics.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Online per-query metrics not found: {path}")
            self._cache[cache_key] = pd.read_parquet(path)
            self._log(f"Loaded online per-query metrics: {self._cache[cache_key].shape}")
        return self._cache[cache_key]
    
    # =========================================================================
    # Raw Metrics (M1, M3, M4, R0)
    # =========================================================================
    
    def load_m1(self, query_subset: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Load M1 (computation per centroid).
        
        Schema: (query_id, q_token_id, centroid_id, num_token_token_sims)
        """
        path = self.tier_a_dir / "M1_compute_per_centroid.parquet"
        if query_subset:
            return pd.read_parquet(path, filters=[('query_id', 'in', list(query_subset))])
        return pd.read_parquet(path)
    
    def load_r0(self, query_subset: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Load R0 (selected centroids).
        
        Schema: (query_id, q_token_id, centroid_id, rank)
        """
        path = self.tier_b_dir / "R0_selected_centroids.parquet"
        if query_subset:
            return pd.read_parquet(path, filters=[('query_id', 'in', list(query_subset))])
        return pd.read_parquet(path)
    
    def load_m3(self, query_subset: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Load M3 (observed winners).
        
        Schema: (query_id, q_token_id, doc_id, winner_embedding_pos, winner_score)
        """
        path = self.tier_b_dir / "M3_observed_winners.parquet"
        if query_subset:
            return pd.read_parquet(path, filters=[('query_id', 'in', list(query_subset))])
        return pd.read_parquet(path)
    
    def get_m4_reader(self) -> ChunkedParquetReader:
        """
        Get chunked reader for M4 (oracle winners) - large file.
        
        Schema: (query_id, q_token_id, doc_id, oracle_embedding_pos, oracle_score)
        """
        path = self.tier_b_dir / "M4_oracle_winners.parquet"
        return ChunkedParquetReader(path, chunk_size=self.chunk_size, verbose=self.verbose)
    
    def get_m5_reader(self) -> ChunkedParquetReader:
        """
        Get chunked reader for M5 (routing misses) - very large file.
        
        Schema: (query_id, q_token_id, doc_id, oracle_embedding_pos, oracle_score,
                 oracle_centroid_id, is_miss, observed_embedding_pos, observed_score, score_delta)
        """
        path = self.tier_b_dir / "M5_routing_misses.parquet"
        return ChunkedParquetReader(path, chunk_size=self.chunk_size, verbose=self.verbose)
    
    # =========================================================================
    # Derived Metrics (M2, M5, M6)
    # =========================================================================
    
    def load_m2(self) -> pd.DataFrame:
        """
        Load M2 (redundant computation per query).
        
        Schema: (query_id, m1_total_sims, m3_influential_pairs, m2_redundant_sims, redundancy_rate)
        """
        path = self.tier_a_dir / "M2_redundant_computation.parquet"
        return pd.read_parquet(path)
    
    def load_m5(self, query_subset: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Load M5 (routing misses).
        
        Schema: (query_id, q_token_id, doc_id, oracle_embedding_pos, oracle_score,
                 oracle_centroid_id, is_miss, observed_embedding_pos, observed_score, score_delta)
        """
        path = self.tier_b_dir / "M5_routing_misses.parquet"
        if query_subset:
            return pd.read_parquet(path, filters=[('query_id', 'in', list(query_subset))])
        return pd.read_parquet(path)
    
    def load_m6_global(self) -> pd.DataFrame:
        """
        Load M6 global (missed centroids aggregated across all queries).
        
        Schema: (oracle_centroid_id, miss_count, oracle_win_count, miss_rate)
        """
        path = self.tier_a_dir / "M6_missed_centroids_global.parquet"
        return pd.read_parquet(path)
    
    def load_m6_per_query(self, query_subset: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Load M6 per query (missed centroids per query).
        
        Schema: (query_id, oracle_centroid_id, miss_count, oracle_win_count, miss_rate)
        """
        path = self.tier_b_dir / "M6_per_query.parquet"
        if query_subset:
            return pd.read_parquet(path, filters=[('query_id', 'in', list(query_subset))])
        return pd.read_parquet(path)
    
    # =========================================================================
    # Parallel Loading
    # =========================================================================
    
    def load_multiple_parallel(
        self, 
        datasets: List[str],
        query_subset: Optional[Set[int]] = None,
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets in parallel.
        
        Args:
            datasets: List of dataset names to load (e.g., ["m1", "m3", "r0"])
            query_subset: Optional query ID filter
            max_workers: Number of parallel threads
            
        Returns:
            Dict mapping dataset name to DataFrame
        """
        loader_map = {
            "offline": lambda: self.load_offline_properties(),
            "online_centroids": lambda: self.load_online_centroid_aggregates(),
            "online_queries": lambda: self.load_online_per_query_metrics(),
            "m1": lambda: self.load_m1(query_subset),
            "m2": lambda: self.load_m2(),
            "m3": lambda: self.load_m3(query_subset),
            "m5": lambda: self.load_m5(query_subset),
            "m6_global": lambda: self.load_m6_global(),
            "m6_per_query": lambda: self.load_m6_per_query(query_subset),
            "r0": lambda: self.load_r0(query_subset),
        }
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(loader_map[name]): name
                for name in datasets
                if name in loader_map
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    warnings.warn(f"Failed to load {name}: {e}")
        
        return results
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
