"""
Derived metrics computation for WARP measurements.

Computes M2, M5, M6 from raw measurement Parquet files:
- M2: Redundant computation (M1 - M3)
- M5: Routing misses (oracle centroid not selected)
- M6: Missed centroid aggregation

These are post-processing utilities that operate on Parquet files produced
by the M1, M3, M4, and R0 measurement infrastructure. No runtime pipeline
changes are required.

Usage:
    from warp.utils.derived_metrics import DerivedMetricsComputer
    
    computer = DerivedMetricsComputer(
        run_dir="/mnt/warp_measurements/runs/my_run",
        index_path="/mnt/datasets/index/beir-quora.split=test.nbits=4"
    )
    
    # Compute all derived metrics
    results = computer.compute_all()
    
    # Or compute individually
    m2_df = computer.compute_m2()
    m5_df = computer.compute_m5()
    m6_global, m6_per_query = computer.compute_m6()
    
    # Print summary
    computer.print_summary()

See docs/M2_M5_M6_INTEGRATION_PLAN.md for detailed specification.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq


class DerivedMetricsComputer:
    """
    Computes derived metrics (M2, M5, M6) from raw measurement files.
    
    All derived metrics are computed offline by joining/aggregating the
    raw Parquet files (M1, M3, M4, R0) produced during WARP search.
    
    Attributes:
        run_dir: Path to measurement run directory
        index_path: Path to WARP index (needed for centroid lookup)
        tier_a_dir: Path to Tier A output directory
        tier_b_dir: Path to Tier B output directory
    """
    
    def __init__(
        self,
        run_dir: str,
        index_path: Optional[str] = None,
        chunk_size: int = 1_000_000
    ):
        """
        Initialize the derived metrics computer.
        
        Args:
            run_dir: Path to measurement run directory (contains tier_a/, tier_b/)
            index_path: Path to WARP index (needed for M5 centroid lookup).
                       Can be auto-detected from metadata.json if not provided.
            chunk_size: Number of rows to process at once for large files.
                       Currently unused but reserved for future MS MARCO scale.
        """
        self.run_dir = Path(run_dir)
        self.chunk_size = chunk_size
        
        # Paths
        self.tier_a_dir = self.run_dir / "tier_a"
        self.tier_b_dir = self.run_dir / "tier_b"
        
        # Try to get index_path from metadata if not provided
        if index_path is None:
            index_path = self._get_index_path_from_metadata()
        self.index_path = Path(index_path) if index_path else None
        
        # Lazy-loaded index data
        self._offsets_compacted: Optional[torch.Tensor] = None
        
        # Validate run directory exists
        if not self.run_dir.exists():
            raise ValueError(f"Run directory not found: {self.run_dir}")
    
    def _get_index_path_from_metadata(self) -> Optional[str]:
        """Try to extract index_path from metadata.json."""
        metadata_path = self.run_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get("index", {}).get("path")
        return None
    
    @property
    def offsets_compacted(self) -> torch.Tensor:
        """
        Load offsets_compacted lazily (needed for M5 centroid lookup).
        
        The offsets array maps embedding positions to centroids:
        - offsets[c] = first embedding position in centroid c
        - offsets[c+1] = first embedding position in centroid c+1
        - centroid_id = searchsorted(offsets, embedding_pos, side='right') - 1
        
        Returns:
            Tensor of shape (num_centroids + 1,) with cumulative embedding counts
        
        Raises:
            ValueError: If index_path is not set
        """
        if self._offsets_compacted is None:
            if self.index_path is None:
                raise ValueError("index_path required for M5/M6 computation")
            sizes = torch.load(self.index_path / "sizes.compacted.pt")
            self._offsets_compacted = torch.zeros(len(sizes) + 1, dtype=torch.long)
            torch.cumsum(sizes, dim=0, out=self._offsets_compacted[1:])
        return self._offsets_compacted
    
    def _check_source_files(self, required: list) -> None:
        """
        Check that required source files exist.
        
        Args:
            required: List of file names to check (e.g., ["M1", "M3"])
        
        Raises:
            FileNotFoundError: If any required file is missing
        """
        file_paths = {
            "M1": self.tier_a_dir / "M1_compute_per_centroid.parquet",
            "M3": self.tier_b_dir / "M3_observed_winners.parquet",
            "M4": self.tier_b_dir / "M4_oracle_winners.parquet",
            "R0": self.tier_b_dir / "R0_selected_centroids.parquet",
        }
        
        missing = []
        for name in required:
            if name in file_paths and not file_paths[name].exists():
                missing.append(f"{name}: {file_paths[name]}")
        
        if missing:
            raise FileNotFoundError(
                f"Required source files not found:\n" + "\n".join(missing)
            )
    
    def compute_m2(self, save: bool = True) -> pd.DataFrame:
        """
        Compute M2: Redundant token-level computation.
        
        M2 = M1 (total computed sims) - M3 (influential pairs)
        
        This quantifies the "wasted" computation: token-token similarities
        that were computed but could not influence the final document scores
        due to MaxSim reduction.
        
        Note: M3 only covers top-k documents, so M2 may undercount influential
        pairs if documents outside top-k had evidence. For a more accurate
        measure, use M4's scope (all scored docs) as the denominator.
        
        Args:
            save: Whether to save results to Parquet file
        
        Returns:
            DataFrame with columns:
            - query_id: Query identifier
            - m1_total_sims: Total token-token similarities computed
            - m3_influential_pairs: Number of (token, doc) pairs with evidence
            - m2_redundant_sims: M1 - M3 (redundant computation)
            - redundancy_rate: M2 / M1 (fraction of wasted work)
        """
        self._check_source_files(["M1", "M3"])
        
        # Load M1: total token-token sims per (query, token, centroid)
        m1_path = self.tier_a_dir / "M1_compute_per_centroid.parquet"
        m1 = pd.read_parquet(m1_path)
        
        # Aggregate M1 per query: sum all token-token sims
        m1_per_query = m1.groupby("query_id")["num_token_token_sims"].sum().reset_index()
        m1_per_query.columns = ["query_id", "m1_total_sims"]
        
        # Load M3: one row per (query, token, doc) that had computed evidence
        m3_path = self.tier_b_dir / "M3_observed_winners.parquet"
        m3 = pd.read_parquet(m3_path)
        
        # M3 per query: count unique (token, doc) pairs
        # Each row in M3 represents one influential interaction
        m3_per_query = m3.groupby("query_id").size().reset_index(name="m3_influential_pairs")
        
        # Join and compute M2
        result = m1_per_query.merge(m3_per_query, on="query_id", how="outer").fillna(0)
        result["m2_redundant_sims"] = result["m1_total_sims"] - result["m3_influential_pairs"]
        result["redundancy_rate"] = result["m2_redundant_sims"] / result["m1_total_sims"]
        
        # Convert types for clean storage
        result["query_id"] = result["query_id"].astype("int32")
        result["m1_total_sims"] = result["m1_total_sims"].astype("int64")
        result["m3_influential_pairs"] = result["m3_influential_pairs"].astype("int64")
        result["m2_redundant_sims"] = result["m2_redundant_sims"].astype("int64")
        result["redundancy_rate"] = result["redundancy_rate"].astype("float32")
        
        # Validation: M2 should never be negative
        if (result["m2_redundant_sims"] < 0).any():
            warnings.warn(
                f"Found {(result['m2_redundant_sims'] < 0).sum()} queries with negative M2 "
                "(M3 > M1). This indicates a possible bug."
            )
        
        if save:
            output_path = self.tier_a_dir / "M2_redundant_computation.parquet"
            result.to_parquet(output_path, index=False)
            print(f"Saved M2 to {output_path}")
        
        return result
    
    def compute_m5(
        self,
        save: bool = True,
        include_m3_join: bool = True,
        top_k_only: bool = False
    ) -> pd.DataFrame:
        """
        Compute M5: Routing misses.
        
        A miss occurs when the oracle winner's centroid was NOT in the
        selected centroids for that query token. This identifies specific
        instances where routing constraints caused WARP to miss better evidence.
        
        Args:
            save: Whether to save results to Parquet file
            include_m3_join: If True, join with M3 to get observed_score for
                            score_delta computation. Only works for docs in top-k.
            top_k_only: If True, only compute misses for docs that appear in M3
                       (top-k). This produces results comparable to the miss
                       rate reported in M4 E2E tests.
        
        Returns:
            DataFrame with columns:
            - query_id, q_token_id, doc_id: Identifiers
            - oracle_embedding_pos: Oracle winner's global embedding position
            - oracle_score: Oracle MaxSim score (over ALL doc embeddings)
            - oracle_centroid_id: Centroid containing oracle winner
            - is_miss: True if oracle centroid was NOT selected
            - observed_embedding_pos: Observed winner's position (if in M3)
            - observed_score: Observed MaxSim score (if in M3)
            - score_delta: oracle_score - observed_score
        """
        self._check_source_files(["M4", "R0"])
        
        # Load M4 (oracle winners)
        m4 = pd.read_parquet(self.tier_b_dir / "M4_oracle_winners.parquet")
        
        # Load R0 (selected centroids)
        r0 = pd.read_parquet(self.tier_b_dir / "R0_selected_centroids.parquet")
        
        # Optionally filter to top-k docs only
        if top_k_only:
            self._check_source_files(["M3"])
            m3 = pd.read_parquet(self.tier_b_dir / "M3_observed_winners.parquet")
            top_k_docs = set(zip(m3["query_id"], m3["q_token_id"], m3["doc_id"]))
            m4_keys = set(zip(m4["query_id"], m4["q_token_id"], m4["doc_id"]))
            m4 = m4[m4.apply(lambda r: (r["query_id"], r["q_token_id"], r["doc_id"]) in top_k_docs, axis=1)]
            print(f"Filtered M4 to {len(m4):,} rows (top-k docs only)")
        
        # Step 1: Compute oracle_centroid_id via vectorized binary search
        # This is the key operation: map embedding_pos → centroid_id
        oracle_positions = torch.tensor(m4["oracle_embedding_pos"].values, dtype=torch.long)
        oracle_centroids = torch.searchsorted(
            self.offsets_compacted, oracle_positions, side="right"
        ) - 1
        m4["oracle_centroid_id"] = oracle_centroids.numpy().astype("int32")
        
        # Step 2: Build selected centroid sets per (query, token)
        r0_sets = r0.groupby(["query_id", "q_token_id"])["centroid_id"].apply(set).to_dict()
        
        # Step 3: Compute is_miss for each M4 row
        # Vectorized implementation for better performance
        def compute_is_miss_vectorized(df: pd.DataFrame, r0_sets: dict) -> pd.Series:
            """Check if oracle centroid is in selected centroids for each row."""
            is_miss = []
            for qid, tid, oc in zip(df["query_id"], df["q_token_id"], df["oracle_centroid_id"]):
                selected = r0_sets.get((qid, tid), set())
                is_miss.append(oc not in selected)
            return pd.Series(is_miss, index=df.index)
        
        m4["is_miss"] = compute_is_miss_vectorized(m4, r0_sets)
        
        # Step 4: Optionally join with M3 for score_delta
        if include_m3_join:
            self._check_source_files(["M3"])
            m3 = pd.read_parquet(self.tier_b_dir / "M3_observed_winners.parquet")
            
            m4 = m4.merge(
                m3[["query_id", "q_token_id", "doc_id", "winner_embedding_pos", "winner_score"]],
                on=["query_id", "q_token_id", "doc_id"],
                how="left"
            )
            m4.rename(columns={
                "winner_embedding_pos": "observed_embedding_pos",
                "winner_score": "observed_score"
            }, inplace=True)
            
            # score_delta = oracle - observed (positive means oracle was better)
            # For docs not in top-k, observed_score is NaN, use oracle_score
            m4["score_delta"] = m4["oracle_score"] - m4["observed_score"].fillna(m4["oracle_score"])
        
        # Select output columns
        output_cols = [
            "query_id", "q_token_id", "doc_id",
            "oracle_embedding_pos", "oracle_score", "oracle_centroid_id", "is_miss"
        ]
        if include_m3_join:
            output_cols.extend(["observed_embedding_pos", "observed_score", "score_delta"])
        
        result = m4[output_cols].copy()
        
        if save:
            output_path = self.tier_b_dir / "M5_routing_misses.parquet"
            result.to_parquet(output_path, index=False)
            print(f"Saved M5 to {output_path}")
        
        return result
    
    def compute_m6(
        self,
        m5_df: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute M6: Missed centroid aggregation.
        
        Aggregates M5 misses by centroid to identify "problem centroids"
        that frequently contain oracle winners but are rarely selected.
        
        Args:
            m5_df: Pre-computed M5 DataFrame. If not provided, will load from
                  M5_routing_misses.parquet or compute fresh.
            save: Whether to save results to Parquet files
        
        Returns:
            Tuple of (m6_global, m6_per_query):
            
            m6_global (Tier A): Global aggregation by centroid
            - oracle_centroid_id: Centroid identifier
            - miss_count: Total misses for this centroid
            - oracle_win_count: Total oracle wins for this centroid
            - miss_rate: miss_count / oracle_win_count
            
            m6_per_query (Tier B): Per-query breakdown
            - query_id, oracle_centroid_id: Identifiers
            - miss_count, oracle_win_count, miss_rate: Per-query stats
        """
        if m5_df is None:
            m5_path = self.tier_b_dir / "M5_routing_misses.parquet"
            if m5_path.exists():
                m5_df = pd.read_parquet(m5_path)
            else:
                m5_df = self.compute_m5(save=False)
        
        # M6 Tier A: Global aggregation by centroid
        m6_global = m5_df.groupby("oracle_centroid_id").agg(
            miss_count=("is_miss", "sum"),
            oracle_win_count=("is_miss", "count")
        ).reset_index()
        m6_global["miss_rate"] = m6_global["miss_count"] / m6_global["oracle_win_count"]
        
        # Type conversion for clean storage
        m6_global = m6_global.astype({
            "oracle_centroid_id": "int32",
            "miss_count": "int32",
            "oracle_win_count": "int32",
            "miss_rate": "float32"
        })
        
        # M6 Tier B: Per-query aggregation
        m6_per_query = m5_df.groupby(["query_id", "oracle_centroid_id"]).agg(
            miss_count=("is_miss", "sum"),
            oracle_win_count=("is_miss", "count")
        ).reset_index()
        m6_per_query["miss_rate"] = m6_per_query["miss_count"] / m6_per_query["oracle_win_count"]
        
        # Type conversion
        m6_per_query["query_id"] = m6_per_query["query_id"].astype("int32")
        m6_per_query["oracle_centroid_id"] = m6_per_query["oracle_centroid_id"].astype("int32")
        m6_per_query["miss_count"] = m6_per_query["miss_count"].astype("int32")
        m6_per_query["oracle_win_count"] = m6_per_query["oracle_win_count"].astype("int32")
        m6_per_query["miss_rate"] = m6_per_query["miss_rate"].astype("float32")
        
        if save:
            global_path = self.tier_a_dir / "M6_missed_centroids_global.parquet"
            per_query_path = self.tier_b_dir / "M6_per_query.parquet"
            
            m6_global.to_parquet(global_path, index=False)
            m6_per_query.to_parquet(per_query_path, index=False)
            
            print(f"Saved M6 global to {global_path}")
            print(f"Saved M6 per-query to {per_query_path}")
        
        return m6_global, m6_per_query
    
    def compute_all(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Compute all derived metrics (M2, M5, M6).
        
        Args:
            verbose: Whether to print progress messages
        
        Returns:
            Dictionary with all computed DataFrames:
            - "M2": Redundant computation per query
            - "M5": Routing misses per (query, token, doc)
            - "M6_global": Missed centroids aggregated globally
            - "M6_per_query": Missed centroids per query
        """
        if verbose:
            print("Computing M2 (redundant computation)...")
        m2 = self.compute_m2(save=True)
        
        if verbose:
            print("Computing M5 (routing misses)...")
        m5 = self.compute_m5(save=True)
        
        if verbose:
            print("Computing M6 (missed centroid aggregation)...")
        m6_global, m6_per_query = self.compute_m6(m5_df=m5, save=True)
        
        return {
            "M2": m2,
            "M5": m5,
            "M6_global": m6_global,
            "M6_per_query": m6_per_query
        }
    
    def print_summary(self) -> None:
        """
        Print a summary of all available measurements (raw + derived).
        
        Loads existing Parquet files and prints key statistics.
        """
        print("=" * 60)
        print("MEASUREMENT SUMMARY")
        print(f"Run: {self.run_dir.name}")
        print("=" * 60)
        
        # Raw metrics
        print("\n--- Raw Metrics ---")
        
        m1_path = self.tier_a_dir / "M1_compute_per_centroid.parquet"
        if m1_path.exists():
            m1 = pd.read_parquet(m1_path)
            print(f"\nM1 (Token-Level Computation):")
            print(f"  Records: {len(m1):,}")
            print(f"  Queries: {m1['query_id'].nunique()}")
            print(f"  Total sims: {m1['num_token_token_sims'].sum():,}")
        
        m3_path = self.tier_b_dir / "M3_observed_winners.parquet"
        if m3_path.exists():
            m3 = pd.read_parquet(m3_path)
            print(f"\nM3 (Observed Winners):")
            print(f"  Records: {len(m3):,}")
            print(f"  Unique docs: {m3['doc_id'].nunique():,}")
        
        m4_path = self.tier_b_dir / "M4_oracle_winners.parquet"
        if m4_path.exists():
            m4 = pd.read_parquet(m4_path)
            print(f"\nM4 (Oracle Winners):")
            print(f"  Records: {len(m4):,}")
            print(f"  Unique docs: {m4['doc_id'].nunique():,}")
        
        # Derived metrics
        print("\n--- Derived Metrics ---")
        
        m2_path = self.tier_a_dir / "M2_redundant_computation.parquet"
        if m2_path.exists():
            m2 = pd.read_parquet(m2_path)
            print(f"\nM2 (Redundant Computation):")
            print(f"  Queries: {len(m2):,}")
            print(f"  Mean M1 (total sims): {m2['m1_total_sims'].mean():,.0f}")
            print(f"  Mean M3 (influential): {m2['m3_influential_pairs'].mean():,.0f}")
            print(f"  Mean redundancy rate: {m2['redundancy_rate'].mean():.1%}")
        else:
            print("\nM2: Not computed yet")
        
        m5_path = self.tier_b_dir / "M5_routing_misses.parquet"
        if m5_path.exists():
            m5 = pd.read_parquet(m5_path)
            print(f"\nM5 (Routing Misses):")
            print(f"  Total rows: {len(m5):,}")
            print(f"  Total misses: {m5['is_miss'].sum():,}")
            print(f"  Miss rate: {m5['is_miss'].mean():.2%}")
            if "score_delta" in m5.columns:
                misses = m5[m5["is_miss"]]
                if len(misses) > 0:
                    print(f"  Mean score_delta on miss: {misses['score_delta'].mean():.4f}")
        else:
            print("\nM5: Not computed yet")
        
        m6_global_path = self.tier_a_dir / "M6_missed_centroids_global.parquet"
        if m6_global_path.exists():
            m6 = pd.read_parquet(m6_global_path)
            print(f"\nM6 (Missed Centroids - Global):")
            print(f"  Centroids with oracle wins: {len(m6):,}")
            print(f"  Centroids with any miss: {(m6['miss_count'] > 0).sum():,}")
            print(f"  Mean miss rate: {m6['miss_rate'].mean():.2%}")
            if len(m6) > 0:
                top = m6.loc[m6['miss_count'].idxmax()]
                print(f"  Top problem centroid: {int(top['oracle_centroid_id'])} "
                      f"({int(top['miss_count']):,} misses)")
        else:
            print("\nM6: Not computed yet")


def compute_derived_metrics_for_run(
    run_dir: str,
    index_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to compute all derived metrics for a run.
    
    Args:
        run_dir: Path to measurement run directory
        index_path: Path to WARP index (auto-detected if not provided)
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary with all computed DataFrames
    
    Example:
        >>> results = compute_derived_metrics_for_run(
        ...     "/mnt/warp_measurements/runs/my_run",
        ...     "/mnt/datasets/index/beir-quora.split=test.nbits=4"
        ... )
        >>> print(f"Miss rate: {results['M5']['is_miss'].mean():.2%}")
    """
    computer = DerivedMetricsComputer(run_dir=run_dir, index_path=index_path)
    results = computer.compute_all(verbose=verbose)
    if verbose:
        computer.print_summary()
    return results
