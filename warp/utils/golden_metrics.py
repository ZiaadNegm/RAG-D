"""
Golden Document Metrics Computation (M4R/M6R)

This module computes oracle accessibility metrics restricted to "golden" documents -
documents that are known to be relevant according to qrels (ground truth).

Key Metrics:
- M4R: For each (query, q_token, golden_doc) tuple, computes:
  - Oracle optimality: Is the oracle-winning embedding accessible via routing?
  - Full accessibility: Is ANY embedding of the doc accessible via routing?
  
- Routing Status: Three-way classification per (query, golden_doc):
  - FULLY_OPTIMAL: All query tokens have their oracle winners accessible
  - PARTIAL: At least one embedding is accessible, but some oracle winners are blocked
  - MSE_ONLY: NO embeddings are accessible - completely dependent on MSE fallback

- M6R: Centroids that contain golden document embeddings but were not routed to

Usage:
    from warp.utils.golden_metrics import GoldenMetricsComputer
    
    computer = GoldenMetricsComputer(
        index_path="/path/to/warp/index",
        metrics_dir="/path/to/metrics/output"
    )
    
    # Compute all metrics
    m4r_df = computer.compute_m4r(qrels)
    routing_status_df = computer.compute_routing_status(m4r_df)
    m6r_df = computer.compute_m6r(qrels)
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

# Import from WARP utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from warp.engine.utils.reverse_index import ReverseIndex


class RoutingStatus(Enum):
    """Three-way classification of golden document accessibility."""
    FULLY_OPTIMAL = "fully_optimal"  # All oracle winners accessible
    PARTIAL = "partial"               # Some accessibility, some oracle misses
    MSE_ONLY = "mse_only"            # No embeddings accessible at all


@dataclass
class GoldenMetricsConfig:
    """Configuration for golden metrics computation."""
    nprobe: int = 32  # Default WARP nprobe setting
    batch_size: int = 1000  # Queries to process before writing checkpoint
    verbose: bool = True


class GoldenMetricsComputer:
    """
    Computes oracle and accessibility metrics for golden documents.
    
    This restricts the full M4/M5/M6 metrics to only golden documents,
    making computation tractable (thousands vs millions of docs).
    """
    
    def __init__(
        self,
        index_path: str,
        metrics_dir: str,
        config: Optional[GoldenMetricsConfig] = None
    ):
        """
        Initialize the golden metrics computer.
        
        Args:
            index_path: Path to WARP index directory.
            metrics_dir: Path to metrics output directory (containing R0.parquet, M1.parquet, etc.)
            config: Optional configuration settings.
        """
        self.index_path = Path(index_path)
        self.metrics_dir = Path(metrics_dir)
        self.config = config or GoldenMetricsConfig()
        
        # Lazy load index components
        self._reverse_index: Optional[ReverseIndex] = None
        self._offsets_compacted: Optional[torch.Tensor] = None
        self._sizes_compacted: Optional[torch.Tensor] = None
        self._centroids: Optional[torch.Tensor] = None
        self._residuals_compacted: Optional[torch.Tensor] = None
        self._codes_compacted: Optional[torch.Tensor] = None
        
        # Lazy load metrics data
        self._r0_df: Optional[pd.DataFrame] = None  # Routing decisions
        self._m1_df: Optional[pd.DataFrame] = None  # Oracle computations
        
        # Cache
        self._num_centroids: Optional[int] = None
        
    @property
    def reverse_index(self) -> ReverseIndex:
        """Lazy load reverse index."""
        if self._reverse_index is None:
            if self.config.verbose:
                print(f"Loading reverse index from {self.index_path}...")
            self._reverse_index = ReverseIndex.load_or_build(
                str(self.index_path), 
                verbose=self.config.verbose
            )
        return self._reverse_index
    
    @property
    def offsets_compacted(self) -> torch.Tensor:
        """Lazy load centroid offsets."""
        if self._offsets_compacted is None:
            sizes_path = self.index_path / "sizes.compacted.pt"
            sizes = torch.load(sizes_path)
            self._sizes_compacted = sizes
            self._offsets_compacted = torch.zeros(len(sizes) + 1, dtype=torch.int64)
            torch.cumsum(sizes, dim=0, out=self._offsets_compacted[1:])
        return self._offsets_compacted
    
    @property
    def num_centroids(self) -> int:
        """Get number of centroids."""
        if self._num_centroids is None:
            _ = self.offsets_compacted  # Ensure loaded
            self._num_centroids = len(self._sizes_compacted)
        return self._num_centroids
    
    @property
    def r0_df(self) -> pd.DataFrame:
        """Lazy load R0 (routing decisions)."""
        if self._r0_df is None:
            # Try multiple paths - tier_b or root
            r0_paths = [
                self.metrics_dir / "tier_b" / "R0_selected_centroids.parquet",
                self.metrics_dir / "R0.parquet",
                self.metrics_dir / "R0_selected_centroids.parquet"
            ]
            
            r0_path = None
            for path in r0_paths:
                if path.exists():
                    r0_path = path
                    break
            
            if r0_path is None:
                raise FileNotFoundError(
                    f"R0 file not found. Tried: {[str(p) for p in r0_paths]}"
                )
            
            if self.config.verbose:
                print(f"Loading R0 (routing decisions) from {r0_path}...")
            self._r0_df = pd.read_parquet(r0_path)
        return self._r0_df
    
    @property
    def m4_df(self) -> pd.DataFrame:
        """
        Lazy load M4 (oracle winners) - WARNING: This is a huge file!
        Use get_m4_path() and filter with pyarrow instead.
        """
        raise RuntimeError(
            "M4 file is too large to load fully (~2.6B rows). "
            "Use get_m4_filtered() or filter_m4_for_golden_docs() instead."
        )
    
    def get_m4_path(self) -> Path:
        """Get the path to M4 file."""
        m4_paths = [
            self.metrics_dir / "tier_b" / "M4_oracle_winners.parquet",
            self.metrics_dir / "M1.parquet",
            self.metrics_dir / "M4_oracle_winners.parquet"
        ]
        
        for path in m4_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"M4/M1 file not found. Tried: {[str(p) for p in m4_paths]}"
        )
    
    def filter_m4_for_golden_docs(
        self,
        qrel_pairs: Set[Tuple[int, int]],
        golden_query_ids: Set[int],
        golden_doc_ids: Set[int]
    ) -> pd.DataFrame:
        """
        Filter M4 to only golden document entries using exact pair matching.
        
        This reads row groups and filters in memory with two stages:
        1. Fast pre-filter by query_id and doc_id sets
        2. Exact match by (query_id, doc_id) tuple in qrels
        
        Args:
            qrel_pairs: Set of (query_id, doc_id) tuples from qrels
            golden_query_ids: Set of query IDs for pre-filtering
            golden_doc_ids: Set of doc IDs for pre-filtering
        """
        import pyarrow.parquet as pq
        
        m4_path = self.get_m4_path()
        
        if self.config.verbose:
            print(f"Filtering M4 for {len(qrel_pairs):,} (query,doc) pairs...")
            print(f"  Source: {m4_path}")
        
        # Read parquet file metadata to get row groups
        pf = pq.ParquetFile(m4_path)
        num_row_groups = pf.metadata.num_row_groups
        
        if self.config.verbose:
            print(f"  Row groups: {num_row_groups}")
        
        # Process row groups in chunks
        filtered_chunks = []
        total_rows_scanned = 0
        
        for rg_idx in tqdm(range(num_row_groups), desc="Scanning M4 row groups", disable=not self.config.verbose):
            table = pf.read_row_group(rg_idx)
            df_chunk = table.to_pandas()
            total_rows_scanned += len(df_chunk)
            
            # Stage 1: Fast pre-filter (may include false positives)
            pre_mask = df_chunk['query_id'].isin(golden_query_ids) & df_chunk['doc_id'].isin(golden_doc_ids)
            candidates = df_chunk[pre_mask]
            
            if len(candidates) > 0:
                # Stage 2: Exact (query_id, doc_id) pair matching
                pairs = list(zip(candidates['query_id'], candidates['doc_id']))
                exact_mask = [p in qrel_pairs for p in pairs]
                filtered = candidates.loc[exact_mask]
                
                if len(filtered) > 0:
                    filtered_chunks.append(filtered)
        
        if filtered_chunks:
            df = pd.concat(filtered_chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=['query_id', 'q_token_id', 'doc_id', 'oracle_embedding_pos', 'oracle_score'])
        
        if self.config.verbose:
            print(f"  Scanned {total_rows_scanned:,} rows")
            print(f"  Filtered M4: {len(df):,} rows")
        
        return df
    
    # Keep m1_df as alias for backwards compatibility
    @property
    def m1_df(self) -> pd.DataFrame:
        """Alias for m4_df (backwards compatibility)."""
        return self.m4_df
    
    def get_routed_centroids(self, query_id: int, q_token_id: int) -> Set[int]:
        """
        Get the set of centroid IDs that were routed to for a (query, token).
        
        Args:
            query_id: Query identifier.
            q_token_id: Query token position.
            
        Returns:
            Set of centroid IDs in the routing set.
        """
        mask = (self.r0_df['query_id'] == query_id) & (self.r0_df['q_token_id'] == q_token_id)
        return set(self.r0_df.loc[mask, 'centroid_id'].values)
    
    def get_all_routed_centroids_for_query(self, query_id: int) -> Dict[int, Set[int]]:
        """
        Get routed centroids for all tokens in a query.
        
        Returns:
            Dict mapping q_token_id -> set of routed centroid IDs.
        """
        query_r0 = self.r0_df[self.r0_df['query_id'] == query_id]
        result = {}
        for q_token_id, group in query_r0.groupby('q_token_id'):
            result[q_token_id] = set(group['centroid_id'].values)
        return result
    
    def get_doc_centroids(self, doc_id: int) -> Set[int]:
        """
        Get all centroids that contain embeddings from a document.
        
        Args:
            doc_id: Document identifier.
            
        Returns:
            Set of centroid IDs containing this document's embeddings.
        """
        positions = self.reverse_index.get_embedding_positions(doc_id)
        centroids = set()
        for pos in positions:
            centroid = self.reverse_index.get_centroid_for_position(
                pos.item(), 
                self.offsets_compacted
            )
            centroids.add(centroid)
        return centroids
    
    def compute_m4r(
        self,
        qrels: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute M4R: Oracle accessibility metrics for golden documents.
        
        For each (query, q_token, golden_doc) tuple, computes:
        - oracle_embedding_pos: Position of oracle-winning embedding
        - oracle_score: MaxSim score for this embedding
        - oracle_centroid_id: Centroid containing the oracle winner
        - oracle_is_accessible: Whether oracle centroid was routed to
        - num_doc_embeddings: Total embeddings in this document
        - num_doc_centroids: Number of unique centroids for this doc
        - num_accessible_centroids: How many of doc's centroids were routed
        - any_embedding_accessible: Whether ANY embedding is accessible
        
        Args:
            qrels: DataFrame with columns ['query_id', 'doc_id', 'relevance']
            output_path: Optional path to save results.
            
        Returns:
            DataFrame with M4R metrics.
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Computing M4R: Golden Document Oracle Accessibility")
            print(f"{'='*60}")
            print(f"Golden doc pairs: {len(qrels):,}")
        
        # Get unique queries and docs
        golden_queries = set(qrels['query_id'].unique())
        all_golden_docs = set(qrels['doc_id'].unique())
        
        # Build set of exact (query_id, doc_id) pairs from qrels
        qrel_pairs = set(zip(qrels['query_id'], qrels['doc_id']))
        
        if self.config.verbose:
            print(f"Unique queries with golden docs: {len(golden_queries):,}")
            print(f"Unique golden docs: {len(all_golden_docs):,}")
            print(f"Qrel (query,doc) pairs: {len(qrel_pairs):,}")
        
        # Step 1: Load filtered M4 (only golden query-doc pairs)
        if self.config.verbose:
            print("\nStep 1: Loading filtered M4 oracle data...")
        
        m4_golden = self.filter_m4_for_golden_docs(qrel_pairs, golden_queries, all_golden_docs)
        
        # Step 2: Pre-compute doc -> centroids mapping
        if self.config.verbose:
            print("\nStep 2: Pre-computing document centroid mappings...")
        
        doc_to_centroids: Dict[int, Set[int]] = {}
        doc_to_num_embeddings: Dict[int, int] = {}
        
        # Only process docs that exist in both qrels and M4
        docs_in_m4 = set(m4_golden['doc_id'].unique())
        docs_to_process = all_golden_docs & docs_in_m4
        
        if self.config.verbose:
            print(f"  Docs in both qrels and M4: {len(docs_to_process):,}")
        
        for doc_id in tqdm(docs_to_process, desc="Mapping docs to centroids", disable=not self.config.verbose):
            doc_to_centroids[doc_id] = self.get_doc_centroids(doc_id)
            doc_to_num_embeddings[doc_id] = self.reverse_index.get_num_embeddings(doc_id)
        
        # Step 3: Build lookup structures
        if self.config.verbose:
            print("\nStep 3: Building lookup structures...")
        
        # Build query -> golden_docs mapping
        query_to_golden = qrels.groupby('query_id')['doc_id'].apply(set).to_dict()
        
        # Build routing lookup (query, token) -> set of centroids
        r0_grouped = self.r0_df.groupby(['query_id', 'q_token_id'])['centroid_id'].apply(set).to_dict()
        
        # Build query -> all routed centroids (union across all tokens) - OPTIMIZED: single groupby
        # Filter R0 to only queries with golden docs, then group
        r0_filtered = self.r0_df[self.r0_df['query_id'].isin(golden_queries)]
        query_to_all_routed: Dict[int, Set[int]] = r0_filtered.groupby('query_id')['centroid_id'].apply(lambda x: set(x.unique())).to_dict()
        # Ensure all golden queries have an entry (even if empty)
        for query_id in golden_queries:
            if query_id not in query_to_all_routed:
                query_to_all_routed[query_id] = set()
        
        # Step 4: Process M4 rows (vectorized operations)
        if self.config.verbose:
            print("\nStep 4: Computing M4R metrics (vectorized)...")
        
        # Pre-compute oracle centroids for all embedding positions in M4
        emb_positions = m4_golden['oracle_embedding_pos'].values
        
        if self.config.verbose:
            print(f"  Computing centroids for {len(emb_positions):,} embedding positions...")
        
        # Vectorized centroid lookup using numpy searchsorted
        oracle_centroids = np.searchsorted(
            self.offsets_compacted, 
            emb_positions, 
            side='right'
        ) - 1
        oracle_centroids = np.clip(oracle_centroids, 0, len(self.offsets_compacted) - 1)
        
        m4_golden = m4_golden.copy()
        m4_golden['oracle_centroid_id'] = oracle_centroids
        
        # Add doc-level info using merge/map (vectorized)
        if self.config.verbose:
            print("  Adding document-level info...")
        
        # Create doc info lookup
        doc_info = pd.DataFrame([
            {
                'doc_id': doc_id,
                'num_doc_embeddings': doc_to_num_embeddings[doc_id],
                'num_doc_centroids': len(doc_to_centroids[doc_id]),
                'doc_centroid_set': frozenset(doc_to_centroids[doc_id])
            }
            for doc_id in doc_to_centroids
        ])
        
        m4_golden = m4_golden.merge(doc_info, on='doc_id', how='left')
        
        # Compute any_embedding_accessible per (query_id, doc_id) using vectorized set operations
        if self.config.verbose:
            print("  Computing accessibility metrics...")
        
        # Pre-compute (query_id, doc_id) -> any_accessible using our lookups
        qd_pairs = m4_golden[['query_id', 'doc_id']].drop_duplicates()
        accessibility_results = []
        
        for _, row in tqdm(qd_pairs.iterrows(), 
                          total=len(qd_pairs), 
                          desc="Computing accessibility",
                          disable=not self.config.verbose):
            query_id = row['query_id']
            doc_id = row['doc_id']
            
            if doc_id in doc_to_centroids:
                doc_centroids = doc_to_centroids[doc_id]
                all_routed = query_to_all_routed.get(query_id, set())
                accessible = doc_centroids & all_routed
                num_accessible = len(accessible)
                any_accessible = num_accessible > 0
            else:
                num_accessible = 0
                any_accessible = False
            
            accessibility_results.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'num_accessible_centroids': num_accessible,
                'any_embedding_accessible': any_accessible
            })
        
        accessibility_df = pd.DataFrame(accessibility_results)
        m4_golden = m4_golden.merge(accessibility_df, on=['query_id', 'doc_id'], how='left')
        
        # Compute oracle_is_accessible per row using merge with R0
        if self.config.verbose:
            print("  Computing oracle accessibility per token (via merge)...")
        
        # Create a dataframe of (query_id, q_token_id, centroid_id) from R0
        # Each row indicates that centroid_id is routed for that (query, token)
        r0_lookup = self.r0_df[['query_id', 'q_token_id', 'centroid_id']].drop_duplicates()
        r0_lookup['_routed'] = True
        
        # Join M4R with R0 on (query_id, q_token_id, oracle_centroid_id == centroid_id)
        m4_golden = m4_golden.merge(
            r0_lookup,
            left_on=['query_id', 'q_token_id', 'oracle_centroid_id'],
            right_on=['query_id', 'q_token_id', 'centroid_id'],
            how='left'
        )
        m4_golden['oracle_is_accessible'] = m4_golden['_routed'].fillna(False).astype(bool)
        m4_golden.drop(columns=['_routed', 'centroid_id'], inplace=True, errors='ignore')
        
        # Build final M4R dataframe with proper columns
        m4r_columns = [
            'query_id', 'q_token_id', 'doc_id',
            'oracle_embedding_pos', 'oracle_score', 'oracle_centroid_id',
            'oracle_is_accessible', 'num_doc_embeddings', 'num_doc_centroids',
            'num_accessible_centroids', 'any_embedding_accessible'
        ]
        
        df = m4_golden[m4r_columns].copy()
        
        # Drop the frozenset column that was used internally
        if 'doc_centroid_set' in m4_golden.columns:
            m4_golden.drop(columns=['doc_centroid_set'], inplace=True)
        
        if self.config.verbose:
            print(f"\nM4R computed: {len(df):,} rows")
            if len(df) > 0:
                oracle_hit_rate = df['oracle_is_accessible'].mean() * 100
                any_accessible_rate = df['any_embedding_accessible'].mean() * 100
                print(f"  Oracle hit rate: {oracle_hit_rate:.2f}%")
                print(f"  Any-accessible rate: {any_accessible_rate:.2f}%")
        
        if output_path:
            df.to_parquet(output_path, index=False)
            if self.config.verbose:
                print(f"  Saved to: {output_path}")
        
        return df
    
    def compute_routing_status(
        self,
        m4r_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute routing status: Three-way classification per (query, golden_doc).
        
        Classifications:
        - FULLY_OPTIMAL: All query tokens have oracle winners accessible
        - PARTIAL: At least one embedding accessible, but some oracle misses  
        - MSE_ONLY: No embeddings accessible at all
        
        Args:
            m4r_df: DataFrame from compute_m4r()
            output_path: Optional path to save results.
            
        Returns:
            DataFrame with routing status per (query, doc) pair.
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Computing Routing Status Classification")
            print(f"{'='*60}")
        
        # Aggregate per (query, doc)
        grouped = m4r_df.groupby(['query_id', 'doc_id']).agg({
            'oracle_is_accessible': ['sum', 'count'],
            'any_embedding_accessible': 'first',  # Same for all tokens of a doc
            'num_doc_embeddings': 'first',
            'num_doc_centroids': 'first',
            'num_accessible_centroids': 'first'
        })
        
        # Flatten column names
        grouped.columns = [
            'oracle_hits', 'num_tokens',
            'any_embedding_accessible',
            'num_doc_embeddings', 'num_doc_centroids', 'num_accessible_centroids'
        ]
        grouped = grouped.reset_index()
        
        # Classify
        def classify(row):
            if not row['any_embedding_accessible']:
                return RoutingStatus.MSE_ONLY.value
            elif row['oracle_hits'] == row['num_tokens']:
                return RoutingStatus.FULLY_OPTIMAL.value
            else:
                return RoutingStatus.PARTIAL.value
        
        grouped['routing_status'] = grouped.apply(classify, axis=1)
        
        # Compute derived metrics
        grouped['oracle_hit_rate'] = grouped['oracle_hits'] / grouped['num_tokens']
        grouped['centroid_coverage'] = (
            grouped['num_accessible_centroids'] / grouped['num_doc_centroids']
        ).clip(upper=1.0)
        
        if self.config.verbose:
            status_counts = grouped['routing_status'].value_counts()
            total = len(grouped)
            print(f"\nRouting Status Distribution ({total:,} query-doc pairs):")
            for status in [RoutingStatus.FULLY_OPTIMAL.value, RoutingStatus.PARTIAL.value, RoutingStatus.MSE_ONLY.value]:
                count = status_counts.get(status, 0)
                pct = count / total * 100
                print(f"  {status:15s}: {count:6,} ({pct:5.2f}%)")
        
        if output_path:
            grouped.to_parquet(output_path, index=False)
            if self.config.verbose:
                print(f"\n  Saved to: {output_path}")
        
        return grouped
    
    def compute_m6r(
        self,
        qrels: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute M6R: Missed centroids for golden documents.
        
        For each (query, golden_doc), lists centroids that:
        - Contain embeddings from the golden document
        - Were NOT routed to for this query
        
        Args:
            qrels: DataFrame with columns ['query_id', 'doc_id', 'relevance']
            output_path: Optional path to save results.
            
        Returns:
            DataFrame with missed centroid information.
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Computing M6R: Golden Document Missed Centroids")
            print(f"{'='*60}")
        
        # Build query -> golden_docs mapping
        query_to_golden = qrels.groupby('query_id')['doc_id'].apply(set).to_dict()
        
        # Pre-compute doc -> centroids
        all_golden_docs = set(qrels['doc_id'].unique())
        doc_to_centroids: Dict[int, Set[int]] = {}
        
        if self.config.verbose:
            print("Pre-computing document centroid mappings...")
        
        for doc_id in tqdm(all_golden_docs, desc="Mapping docs to centroids", disable=not self.config.verbose):
            doc_to_centroids[doc_id] = self.get_doc_centroids(doc_id)
        
        results = []
        golden_queries = qrels['query_id'].unique()
        
        if self.config.verbose:
            print("\nProcessing queries...")
        
        for query_id in tqdm(golden_queries, desc="Finding missed centroids", disable=not self.config.verbose):
            golden_docs = query_to_golden[query_id]
            
            # Get union of all routed centroids for this query
            routed_by_token = self.get_all_routed_centroids_for_query(query_id)
            if not routed_by_token:
                continue
            all_routed = set().union(*routed_by_token.values())
            
            for doc_id in golden_docs:
                doc_centroids = doc_to_centroids.get(doc_id, set())
                if not doc_centroids:
                    continue
                
                # Find missed centroids
                missed_centroids = doc_centroids - all_routed
                
                for centroid_id in missed_centroids:
                    results.append({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'missed_centroid_id': centroid_id,
                        'num_doc_centroids': len(doc_centroids),
                        'num_missed': len(missed_centroids)
                    })
        
        df = pd.DataFrame(results)
        
        if self.config.verbose:
            print(f"\nM6R computed: {len(df):,} rows (missed centroid records)")
            if len(df) > 0:
                unique_pairs = df.groupby(['query_id', 'doc_id']).size()
                print(f"  Query-doc pairs with missed centroids: {len(unique_pairs):,}")
                print(f"  Avg missed centroids per pair: {unique_pairs.mean():.2f}")
        
        if output_path:
            df.to_parquet(output_path, index=False)
            if self.config.verbose:
                print(f"  Saved to: {output_path}")
        
        return df


def load_qrels(qrels_path: str, collection_map_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load qrels from a TSV or JSON file.
    
    TSV format: query_id, corpus_id, score (tab-separated, no header).
    JSON format: {query_id: {doc_id: relevance, ...}, ...}
    
    If collection_map_path is provided, doc_ids will be translated from external
    (qrels) format to internal (index) format.
    
    Args:
        qrels_path: Path to qrels file (TSV or JSON).
        collection_map_path: Optional path to collection_map.json that maps
                            internal_id -> external_id
        
    Returns:
        DataFrame with columns ['query_id', 'doc_id', 'relevance'].
        If collection_map is provided, doc_ids are internal index IDs.
    """
    import json
    
    # Load collection map if provided (maps internal -> external)
    external_to_internal = {}
    if collection_map_path:
        with open(collection_map_path, 'r') as f:
            collection_map = json.load(f)
        # Build reverse map: external -> internal
        external_to_internal = {v: int(k) for k, v in collection_map.items()}
    
    if qrels_path.endswith('.json'):
        # JSON format
        with open(qrels_path, 'r') as f:
            qrels_dict = json.load(f)
        
        rows = []
        for query_id, docs in qrels_dict.items():
            for doc_id, relevance in docs.items():
                # Convert doc_id using collection map if available
                if external_to_internal:
                    internal_doc_id = external_to_internal.get(doc_id)
                    if internal_doc_id is None:
                        continue  # Skip docs not in collection
                else:
                    internal_doc_id = int(doc_id) if doc_id.isdigit() else doc_id
                    
                rows.append({
                    'query_id': int(query_id) if query_id.isdigit() else query_id,
                    'doc_id': internal_doc_id,
                    'relevance': relevance
                })
        return pd.DataFrame(rows)
    else:
        # TSV format
        df = pd.read_csv(
            qrels_path, 
            sep='\t', 
            names=['query_id', 'doc_id', 'relevance'],
            dtype={'query_id': str, 'doc_id': str, 'relevance': int}
        )
        # Convert query_id to int if numeric
        if df['query_id'].str.isnumeric().all():
            df['query_id'] = df['query_id'].astype(int)
        
        # Apply collection map if available
        if external_to_internal:
            df['doc_id'] = df['doc_id'].map(external_to_internal)
            df = df.dropna(subset=['doc_id'])  # Drop docs not in collection
            df['doc_id'] = df['doc_id'].astype(int)
        else:
            df['doc_id'] = df['doc_id'].astype(int)
        
        return df


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute golden document metrics")
    parser.add_argument("--index-path", required=True, help="Path to WARP index")
    parser.add_argument("--metrics-dir", required=True, help="Path to metrics directory")
    parser.add_argument("--qrels-path", required=True, help="Path to qrels TSV file")
    parser.add_argument("--output-dir", help="Output directory (default: metrics_dir)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.metrics_dir
    os.makedirs(output_dir, exist_ok=True)
    
    config = GoldenMetricsConfig(verbose=not args.quiet)
    computer = GoldenMetricsComputer(
        index_path=args.index_path,
        metrics_dir=args.metrics_dir,
        config=config
    )
    
    # Load qrels
    print(f"Loading qrels from {args.qrels_path}...")
    qrels = load_qrels(args.qrels_path)
    print(f"Loaded {len(qrels):,} qrel entries")
    
    # Compute M4R
    m4r_path = os.path.join(output_dir, "M4R.parquet")
    m4r_df = computer.compute_m4r(qrels, output_path=m4r_path)
    
    # Compute routing status
    routing_status_path = os.path.join(output_dir, "routing_status.parquet")
    routing_status_df = computer.compute_routing_status(m4r_df, output_path=routing_status_path)
    
    # Compute M6R
    m6r_path = os.path.join(output_dir, "M6R.parquet")
    m6r_df = computer.compute_m6r(qrels, output_path=m6r_path)
    
    print(f"\n{'='*60}")
    print("Golden metrics computation complete!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  M4R:           {m4r_path}")
    print(f"  Routing Status: {routing_status_path}")
    print(f"  M6R:           {m6r_path}")


if __name__ == "__main__":
    main()
