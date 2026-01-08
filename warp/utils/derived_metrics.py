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
from typing import Optional, Tuple, Dict, Any, List
import warnings

import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from warp.utils.chunked_m4 import ChunkedM4Processor, merge_partitioned_parquet, DEFAULT_CHUNK_SIZE


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
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verbose: bool = True
    ):
        """
        Initialize the derived metrics computer.
        
        Args:
            run_dir: Path to measurement run directory (contains tier_a/, tier_b/)
            index_path: Path to WARP index (needed for M5 centroid lookup).
                       Can be auto-detected from metadata.json if not provided.
            chunk_size: Number of queries to process per chunk for M4.
                       Default 500 queries. Set higher if more RAM is available.
            verbose: Whether to print progress messages.
        """
        self.run_dir = Path(run_dir)
        self.chunk_size = chunk_size
        self.verbose = verbose
        
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
        top_k_only: bool = False,
        use_chunked: bool = True
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
            use_chunked: If True, process M4 in chunks to avoid OOM errors.
                        Recommended for large runs (10K+ queries).
        
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
        
        m4_path = self.tier_b_dir / "M4_oracle_winners.parquet"
        
        # Check if M4 is large enough to warrant chunked processing
        m4_size_mb = m4_path.stat().st_size / (1024 * 1024)
        
        if use_chunked and m4_size_mb > 1000:  # > 1GB triggers chunked processing
            if self.verbose:
                print(f"M4 file is {m4_size_mb:.0f} MB - using chunked processing (chunk_size={self.chunk_size})")
            return self._compute_m5_chunked(
                save=save,
                include_m3_join=include_m3_join,
                top_k_only=top_k_only
            )
        else:
            # Small file - use original single-load approach
            return self._compute_m5_single(
                save=save,
                include_m3_join=include_m3_join,
                top_k_only=top_k_only
            )
    
    def _compute_m5_single(
        self,
        save: bool = True,
        include_m3_join: bool = True,
        top_k_only: bool = False
    ) -> pd.DataFrame:
        """
        Original single-load M5 computation (for small M4 files).
        """
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
        
        # Process the data
        result = self._process_m5_chunk(m4, r0, include_m3_join)
        
        if save:
            output_path = self.tier_b_dir / "M5_routing_misses.parquet"
            result.to_parquet(output_path, index=False)
            if self.verbose:
                print(f"Saved M5 to {output_path}")
        
        return result
    
    def _compute_m5_chunked(
        self,
        save: bool = True,
        include_m3_join: bool = True,
        top_k_only: bool = False
    ) -> pd.DataFrame:
        """
        Chunked M5 computation for large M4 files.
        
        Processes M4 in query batches and writes partitioned output files,
        then merges them at the end.
        """
        m4_path = self.tier_b_dir / "M4_oracle_winners.parquet"
        
        # Create partition output directory
        partition_dir = self.tier_b_dir / "M5_partitions"
        partition_dir.mkdir(exist_ok=True)
        
        # Load R0 once (small file, fits in memory)
        r0 = pd.read_parquet(self.tier_b_dir / "R0_selected_centroids.parquet")
        if self.verbose:
            print(f"Loaded R0: {len(r0):,} rows")
        
        # Load M3 once if needed for join
        m3_lookup = None
        if include_m3_join:
            self._check_source_files(["M3"])
            m3 = pd.read_parquet(self.tier_b_dir / "M3_observed_winners.parquet")
            m3_lookup = m3[["query_id", "q_token_id", "doc_id", "winner_embedding_pos", "winner_score"]]
            if self.verbose:
                print(f"Loaded M3: {len(m3):,} rows")
        
        # Load top-k doc set if filtering
        top_k_docs = None
        if top_k_only:
            self._check_source_files(["M3"])
            if m3_lookup is None:
                m3 = pd.read_parquet(self.tier_b_dir / "M3_observed_winners.parquet")
            else:
                m3 = m3_lookup
            top_k_docs = set(zip(m3["query_id"], m3["q_token_id"], m3["doc_id"]))
        
        # Initialize chunked processor
        processor = ChunkedM4Processor(
            m4_path,
            chunk_size=self.chunk_size,
            verbose=self.verbose
        )
        
        if self.verbose:
            info = processor.get_file_info()
            mem_est = processor.estimate_chunk_memory()
            print(f"Processing {info['num_queries']:,} queries in {info['num_chunks']} chunks")
            print(f"Estimated memory per chunk: {mem_est['estimated_chunk_memory_gb']:.1f} GB")
        
        # Process chunks and write partitioned output
        partition_idx = 0
        total_rows = 0
        total_misses = 0
        
        for chunk_qids, m4_chunk in processor.iter_chunks_with_ids(show_progress=self.verbose):
            # Filter to top-k if requested
            if top_k_only and top_k_docs:
                m4_chunk = m4_chunk[
                    m4_chunk.apply(
                        lambda r: (r["query_id"], r["q_token_id"], r["doc_id"]) in top_k_docs,
                        axis=1
                    )
                ]
            
            if len(m4_chunk) == 0:
                continue
            
            # Filter R0 to only this chunk's queries for efficiency
            r0_chunk = r0[r0["query_id"].isin(chunk_qids)]
            
            # Filter M3 lookup if needed
            m3_chunk = None
            if m3_lookup is not None:
                m3_chunk = m3_lookup[m3_lookup["query_id"].isin(chunk_qids)]
            
            # Process chunk
            chunk_result = self._process_m5_chunk(m4_chunk, r0_chunk, include_m3_join, m3_chunk)
            
            # Write partition
            partition_path = partition_dir / f"partition_{partition_idx:04d}.parquet"
            chunk_result.to_parquet(partition_path, index=False)
            
            total_rows += len(chunk_result)
            total_misses += chunk_result["is_miss"].sum()
            partition_idx += 1
        
        if self.verbose:
            print(f"\nWritten {partition_idx} partitions, {total_rows:,} total rows")
            print(f"Total misses: {total_misses:,} ({total_misses/max(total_rows,1):.2%})")
        
        # Merge partitions into final output
        output_path = self.tier_b_dir / "M5_routing_misses.parquet"
        
        if save:
            if self.verbose:
                print("Merging partitions...")
            merge_partitioned_parquet(
                partition_dir=partition_dir,
                output_path=output_path,
                delete_partitions=True,
                verbose=self.verbose
            )
            # Clean up partition directory
            partition_dir.rmdir()
        
        # Return merged result (or read back from disk if large)
        if save and output_path.exists():
            # For very large outputs, don't load into memory
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            if output_size_mb > 5000:  # > 5GB
                if self.verbose:
                    print(f"M5 output is {output_size_mb:.0f} MB - returning None to avoid OOM")
                    print(f"Load with: pd.read_parquet('{output_path}')")
                return None
            return pd.read_parquet(output_path)
        else:
            # Read all partitions and return concatenated
            partition_files = sorted(partition_dir.glob("partition_*.parquet"))
            return pd.concat([pd.read_parquet(pf) for pf in partition_files], ignore_index=True)
    
    def _process_m5_chunk(
        self,
        m4_chunk: pd.DataFrame,
        r0_chunk: pd.DataFrame,
        include_m3_join: bool,
        m3_chunk: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Process a single chunk of M4 data to compute M5.
        
        This is the core computation shared by single and chunked processing.
        """
        # Step 1: Compute oracle_centroid_id via vectorized binary search
        oracle_positions = torch.tensor(m4_chunk["oracle_embedding_pos"].values, dtype=torch.long)
        oracle_centroids = torch.searchsorted(
            self.offsets_compacted, oracle_positions, side="right"
        ) - 1
        m4_chunk = m4_chunk.copy()
        m4_chunk["oracle_centroid_id"] = oracle_centroids.numpy().astype("int32")
        
        # Step 2: Build selected centroid sets per (query, token)
        r0_sets = r0_chunk.groupby(["query_id", "q_token_id"])["centroid_id"].apply(set).to_dict()
        
        # Step 3: Compute is_miss for each M4 row
        def compute_is_miss_vectorized(df: pd.DataFrame, r0_sets: dict) -> pd.Series:
            """Check if oracle centroid is in selected centroids for each row."""
            is_miss = []
            for qid, tid, oc in zip(df["query_id"], df["q_token_id"], df["oracle_centroid_id"]):
                selected = r0_sets.get((qid, tid), set())
                is_miss.append(oc not in selected)
            return pd.Series(is_miss, index=df.index)
        
        m4_chunk["is_miss"] = compute_is_miss_vectorized(m4_chunk, r0_sets)
        
        # Step 4: Optionally join with M3 for score_delta
        if include_m3_join:
            if m3_chunk is None:
                # Load M3 if not provided (fallback for single mode)
                m3_chunk = pd.read_parquet(self.tier_b_dir / "M3_observed_winners.parquet")
                m3_chunk = m3_chunk[["query_id", "q_token_id", "doc_id", "winner_embedding_pos", "winner_score"]]
            
            m4_chunk = m4_chunk.merge(
                m3_chunk,
                on=["query_id", "q_token_id", "doc_id"],
                how="left"
            )
            m4_chunk.rename(columns={
                "winner_embedding_pos": "observed_embedding_pos",
                "winner_score": "observed_score"
            }, inplace=True)
            
            # score_delta = oracle - observed (positive means oracle was better)
            m4_chunk["score_delta"] = m4_chunk["oracle_score"] - m4_chunk["observed_score"].fillna(m4_chunk["oracle_score"])
        
        # Select output columns
        output_cols = [
            "query_id", "q_token_id", "doc_id",
            "oracle_embedding_pos", "oracle_score", "oracle_centroid_id", "is_miss"
        ]
        if include_m3_join:
            output_cols.extend(["observed_embedding_pos", "observed_score", "score_delta"])
        
        return m4_chunk[output_cols].copy()
    
    def compute_m6(
        self,
        m5_df: Optional[pd.DataFrame] = None,
        save: bool = True,
        use_chunked: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute M6: Missed centroid aggregation.
        
        Aggregates M5 misses by centroid to identify "problem centroids"
        that frequently contain oracle winners but are rarely selected.
        
        Args:
            m5_df: Pre-computed M5 DataFrame. If not provided, will load from
                  M5_routing_misses.parquet or compute fresh.
            save: Whether to save results to Parquet files
            use_chunked: If True and M5 file is large, process in chunks.
        
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
        m5_path = self.tier_b_dir / "M5_routing_misses.parquet"
        
        if m5_df is None:
            if m5_path.exists():
                # Check M5 file size
                m5_size_mb = m5_path.stat().st_size / (1024 * 1024)
                
                if use_chunked and m5_size_mb > 5000:  # > 5GB triggers chunked processing
                    if self.verbose:
                        print(f"M5 file is {m5_size_mb:.0f} MB - using chunked M6 computation")
                    return self._compute_m6_chunked(save=save)
                else:
                    m5_df = pd.read_parquet(m5_path)
            else:
                m5_df = self.compute_m5(save=False)
        
        # Handle None return from chunked M5 processing
        if m5_df is None:
            if m5_path.exists():
                return self._compute_m6_chunked(save=save)
            else:
                raise ValueError("M5 data not available and could not be computed")
        
        return self._compute_m6_from_df(m5_df, save=save)
    
    def _compute_m6_from_df(
        self,
        m5_df: pd.DataFrame,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute M6 from an in-memory M5 DataFrame.
        """
        # M6 Tier A: Global aggregation by centroid
        m6_global = m5_df.groupby("oracle_centroid_id").agg(
            miss_count=("is_miss", "sum"),
            oracle_win_count=("is_miss", "count")
        ).reset_index()
        m6_global["miss_rate"] = m6_global["miss_count"] / m6_global["oracle_win_count"]
        
        # Type conversion for clean storage
        m6_global = m6_global.astype({
            "oracle_centroid_id": "int32",
            "miss_count": "int64",
            "oracle_win_count": "int64",
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
        m6_per_query["miss_count"] = m6_per_query["miss_count"].astype("int64")
        m6_per_query["oracle_win_count"] = m6_per_query["oracle_win_count"].astype("int64")
        m6_per_query["miss_rate"] = m6_per_query["miss_rate"].astype("float32")
        
        if save:
            global_path = self.tier_a_dir / "M6_missed_centroids_global.parquet"
            per_query_path = self.tier_b_dir / "M6_per_query.parquet"
            
            m6_global.to_parquet(global_path, index=False)
            m6_per_query.to_parquet(per_query_path, index=False)
            
            if self.verbose:
                print(f"Saved M6 global to {global_path}")
                print(f"Saved M6 per-query to {per_query_path}")
        
        return m6_global, m6_per_query
    
    def _compute_m6_chunked(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute M6 from a large M5 file using chunked processing.
        
        Accumulates aggregates across chunks and combines at the end.
        """
        m5_path = self.tier_b_dir / "M5_routing_misses.parquet"
        
        if self.verbose:
            print("Computing M6 with chunked M5 processing...")
        
        # Get unique query IDs for chunking
        query_ids = pd.read_parquet(m5_path, columns=['query_id'])['query_id'].unique()
        query_ids = sorted(query_ids.tolist())
        
        # Accumulators for global aggregation
        global_accumulator = {}  # centroid_id -> (miss_count, oracle_win_count)
        
        # Per-query results (write directly to partitioned output)
        per_query_partition_dir = self.tier_b_dir / "M6_per_query_partitions"
        per_query_partition_dir.mkdir(exist_ok=True)
        
        # Process in chunks
        num_chunks = (len(query_ids) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in tqdm(range(num_chunks), desc="Processing M6 chunks", disable=not self.verbose):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(query_ids))
            chunk_qids = query_ids[start_idx:end_idx]
            
            # Load M5 chunk using predicate pushdown
            m5_chunk = pd.read_parquet(
                m5_path,
                columns=['query_id', 'oracle_centroid_id', 'is_miss'],
                filters=[('query_id', 'in', chunk_qids)]
            )
            
            # Global aggregation (accumulate)
            chunk_global = m5_chunk.groupby('oracle_centroid_id').agg(
                miss_count=('is_miss', 'sum'),
                oracle_win_count=('is_miss', 'count')
            )
            
            for centroid_id, row in chunk_global.iterrows():
                if centroid_id in global_accumulator:
                    prev_miss, prev_total = global_accumulator[centroid_id]
                    global_accumulator[centroid_id] = (
                        prev_miss + row['miss_count'],
                        prev_total + row['oracle_win_count']
                    )
                else:
                    global_accumulator[centroid_id] = (row['miss_count'], row['oracle_win_count'])
            
            # Per-query aggregation (write partition)
            chunk_per_query = m5_chunk.groupby(['query_id', 'oracle_centroid_id']).agg(
                miss_count=('is_miss', 'sum'),
                oracle_win_count=('is_miss', 'count')
            ).reset_index()
            chunk_per_query['miss_rate'] = chunk_per_query['miss_count'] / chunk_per_query['oracle_win_count']
            
            partition_path = per_query_partition_dir / f"partition_{chunk_idx:04d}.parquet"
            chunk_per_query.to_parquet(partition_path, index=False)
        
        # Build global M6 from accumulator
        m6_global = pd.DataFrame([
            {
                'oracle_centroid_id': cid,
                'miss_count': counts[0],
                'oracle_win_count': counts[1],
                'miss_rate': counts[0] / counts[1] if counts[1] > 0 else 0.0
            }
            for cid, counts in global_accumulator.items()
        ])
        
        m6_global = m6_global.astype({
            'oracle_centroid_id': 'int32',
            'miss_count': 'int64',
            'oracle_win_count': 'int64',
            'miss_rate': 'float32'
        })
        
        # Merge per-query partitions
        per_query_path = self.tier_b_dir / "M6_per_query.parquet"
        
        if save:
            # Save global
            global_path = self.tier_a_dir / "M6_missed_centroids_global.parquet"
            m6_global.to_parquet(global_path, index=False)
            if self.verbose:
                print(f"Saved M6 global to {global_path}")
            
            # Merge per-query partitions
            merge_partitioned_parquet(
                partition_dir=per_query_partition_dir,
                output_path=per_query_path,
                delete_partitions=True,
                verbose=self.verbose
            )
            # Directory already removed by merge_partitioned_parquet
        
        # Load merged per-query (or concatenate if not saved)
        if save and per_query_path.exists():
            m6_per_query = pd.read_parquet(per_query_path)
        else:
            partition_files = sorted(per_query_partition_dir.glob("partition_*.parquet"))
            m6_per_query = pd.concat([pd.read_parquet(pf) for pf in partition_files], ignore_index=True)
        
        # Ensure correct types
        m6_per_query = m6_per_query.astype({
            'query_id': 'int32',
            'oracle_centroid_id': 'int32',
            'miss_count': 'int64',
            'oracle_win_count': 'int64',
            'miss_rate': 'float32'
        })
        
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
