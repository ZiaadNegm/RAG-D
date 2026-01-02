import time
import threading
import json
import os
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Union

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ExecutionTrackerIteration:
    """Context manager for tracking a single query iteration."""
    
    def __init__(self, tracker):
        self._tracker = tracker

    def __enter__(self):
        self._tracker.next_iteration()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._tracker.end_iteration()


class MeasurementTier(Enum):
    """
    Measurement tiers as defined in MEASUREMENT_WISHES.MD.
    
    TIER_A: Small files collected for ALL queries (counters, routing info)
    TIER_B: Large files for forensic analysis (winner identities, scores)
    """
    TIER_A = "tier_a"
    TIER_B = "tier_b"


class MeasurementCollector:
    """
    Collects SQ2 measurements and writes them to Parquet files.
    
    This class handles:
    1. Directory structure setup (one-time, per run_id)
    2. Buffering measurement data in memory
    3. Flushing buffers to Parquet files when threshold is reached
    4. Writing metadata.json with run configuration
    
    Directory structure created:
        /mnt/warp_measurements/
        └── runs/
            └── {run_id}/
                ├── metadata.json
                ├── tier_a/
                │   ├── R0_selected_centroids.parquet
                │   ├── M1_compute_per_centroid.parquet
                │   ├── M3_docs_touched.parquet
                │   └── M6_missed_centroids_global.parquet
                └── tier_b/
                    ├── winners.parquet
                    └── M6_per_query.parquet
    
    Usage:
        collector = MeasurementCollector(
            run_id="2024-12-30_quora_nprobe8",
            output_dir="/mnt/warp_measurements"
        )
        
        # During search loop
        collector.record_m1(query_id=0, q_token_id=1, centroid_id=42, num_sims=1500)
        
        # At end of run
        collector.save_metadata({...})
        collector.flush_all()
    
    See MEASUREMENT_WISHES.MD for full specification.
    """
    
    # Buffer flush threshold: flush to disk when buffer reaches this size.
    # Set conservatively to avoid OOM during heavy WARP pipeline operations.
    # With ~30 bytes per M1 row, 10K rows ≈ 300KB memory per buffer.
    BUFFER_FLUSH_THRESHOLD = 10_000
    
    # Default output directory (can be overridden)
    DEFAULT_OUTPUT_DIR = "/mnt/warp_measurements"
    
    def __init__(
        self,
        run_id: str,
        output_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        create_dirs: bool = True
    ):
        """
        Initialize the measurement collector.
        
        Args:
            run_id: Unique identifier for this run (e.g., "2024-12-30_quora_nprobe8")
            output_dir: Base directory for measurements. Defaults to /mnt/warp_measurements
            index_path: Path to the WARP index being used (for metadata)
            create_dirs: If True, create directory structure immediately
        """
        self.run_id = run_id
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.index_path = index_path
        self._lock = threading.Lock()
        
        # Paths
        self.run_dir = self.output_dir / "runs" / run_id
        self.tier_a_dir = self.run_dir / "tier_a"
        self.tier_b_dir = self.run_dir / "tier_b"
        self.metadata_path = self.run_dir / "metadata.json"
        
        # Buffers for each metric table
        # M1: Total token-level computation per centroid
        self._m1_buffer: List[Dict[str, Any]] = []
        
        # R0: Selected centroids per query token (for future use)
        self._r0_buffer: List[Dict[str, Any]] = []
        
        # M3 Tier A: Docs touched per query token (for future use)
        self._m3a_buffer: List[Dict[str, Any]] = []
        
        # M3 Tier B: Per-token winners for top-k documents
        self._m3b_buffer: List[Dict[str, Any]] = []
        
        # M4: Oracle winners (all embeddings per doc, ignoring routing)
        self._m4_buffer: List[Dict[str, Any]] = []
        
        # Track flush counts for debugging
        self._flush_counts = {"m1": 0, "r0": 0, "m3a": 0, "m3b": 0, "m4": 0}
        
        # Setup directories
        if create_dirs:
            self._setup_directories()
    
    def _setup_directories(self) -> None:
        """
        Create the directory structure for this run.
        
        Creates:
            - {output_dir}/runs/{run_id}/
            - {output_dir}/runs/{run_id}/tier_a/
            - {output_dir}/runs/{run_id}/tier_b/
        
        This is idempotent - safe to call multiple times.
        """
        self.tier_a_dir.mkdir(parents=True, exist_ok=True)
        self.tier_b_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # M1: Total token-level computation per centroid
    # -------------------------------------------------------------------------
    
    def record_m1(
        self,
        query_id: int,
        q_token_id: int,
        centroid_id: int,
        num_token_token_sims: int
    ) -> None:
        """
        Record M1 metric: token-level computation for a specific centroid.
        
        M1 counts all evaluated token-token similarities s(q_i, d_j).
        One unit = one computed similarity between a query token and a doc token.
        
        Args:
            query_id: Query identifier (will be converted to int if string)
            q_token_id: Query token position (0-31 typically)
            centroid_id: The centroid whose embeddings were scored
            num_token_token_sims: Number of similarities computed for this centroid
                                  (equals the number of embeddings in the centroid)
        
        Schema matches MEASUREMENT_WISHES.MD M1_compute_per_centroid.parquet:
            - query_id: int32
            - q_token_id: int8
            - centroid_id: int32
            - num_token_token_sims: int32
        """
        # Convert query_id to int if it's a string (BEIR dataset uses string IDs)
        if isinstance(query_id, str):
            query_id = int(query_id)
        
        with self._lock:
            self._m1_buffer.append({
                "query_id": query_id,
                "q_token_id": q_token_id,
                "centroid_id": centroid_id,
                "num_token_token_sims": num_token_token_sims
            })
            
            if len(self._m1_buffer) >= self.BUFFER_FLUSH_THRESHOLD:
                self._flush_m1_buffer()
    
    def _flush_m1_buffer(self) -> None:
        """
        Write M1 buffer to Parquet file and clear buffer.
        
        Uses PyArrow for efficient columnar storage with append support.
        """
        if not self._m1_buffer:
            return
        
        # Define schema for type consistency
        schema = pa.schema([
            ("query_id", pa.int32()),
            ("q_token_id", pa.int8()),
            ("centroid_id", pa.int32()),
            ("num_token_token_sims", pa.int32())
        ])
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(self._m1_buffer, schema=schema)
        
        # Write to parquet (append if file exists)
        parquet_path = self.tier_a_dir / "M1_compute_per_centroid.parquet"
        
        if parquet_path.exists():
            # Read existing and concatenate
            existing = pq.read_table(parquet_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, parquet_path, compression='snappy')
        
        self._flush_counts["m1"] += 1
        self._m1_buffer = []
    
    # -------------------------------------------------------------------------
    # M3 Tier B: Per-token winners for top-k documents
    # -------------------------------------------------------------------------
    
    def record_m3b_winner(
        self,
        query_id: int,
        q_token_id: int,
        doc_id: int,
        winner_embedding_pos: int,
        winner_score: float
    ) -> None:
        """
        Record M3 Tier B metric: observed winner for a (query_token, doc) pair.
        
        M3 Tier B tracks which embedding position won the MaxSim reduction
        for each (query_token, doc) pair in the top-k results.
        
        Args:
            query_id: Query identifier
            q_token_id: Query token position (0-31)
            doc_id: Document identifier
            winner_embedding_pos: Global position in codes_compacted of the winning embedding
            winner_score: The MaxSim score for this (token, doc) pair
        
        Schema matches MEASUREMENT_WISHES.MD M3_observed_winners.parquet:
            - query_id: int32
            - q_token_id: int8
            - doc_id: int32
            - winner_embedding_pos: int64
            - winner_score: float32
        """
        # Convert query_id to int if it's a string (BEIR dataset uses string IDs)
        if isinstance(query_id, str):
            query_id = int(query_id)
        
        with self._lock:
            self._m3b_buffer.append({
                "query_id": query_id,
                "q_token_id": q_token_id,
                "doc_id": doc_id,
                "winner_embedding_pos": winner_embedding_pos,
                "winner_score": winner_score
            })
            
            if len(self._m3b_buffer) >= self.BUFFER_FLUSH_THRESHOLD:
                self._flush_m3b_buffer()
    
    def _flush_m3b_buffer(self) -> None:
        """
        Write M3 Tier B buffer to Parquet file and clear buffer.
        """
        if not self._m3b_buffer:
            return
        
        # Define schema for type consistency
        schema = pa.schema([
            ("query_id", pa.int32()),
            ("q_token_id", pa.int8()),
            ("doc_id", pa.int32()),
            ("winner_embedding_pos", pa.int64()),
            ("winner_score", pa.float32())
        ])
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(self._m3b_buffer, schema=schema)
        
        # Write to parquet in tier_b directory (append if file exists)
        parquet_path = self.tier_b_dir / "M3_observed_winners.parquet"
        
        if parquet_path.exists():
            # Read existing and concatenate
            existing = pq.read_table(parquet_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, parquet_path, compression='snappy')
        
        self._flush_counts["m3b"] += 1
        self._m3b_buffer = []
    
    # -------------------------------------------------------------------------
    # R0: Selected centroids per query token (for M5 miss detection)
    # -------------------------------------------------------------------------
    
    def record_r0_centroid(
        self,
        query_id: int,
        q_token_id: int,
        centroid_id: int,
        rank: int,
        centroid_score: float = 0.0
    ) -> None:
        """
        Record R0 metric: selected centroid for a query token.
        
        R0 records which centroids were selected for each query token,
        needed for M5 miss detection (comparing oracle winner's centroid
        against selected centroids) and routing fidelity metrics (C4/C5).
        
        Args:
            query_id: Query identifier
            q_token_id: Query token position (0-31)
            centroid_id: The selected centroid ID
            rank: Rank among nprobe selections (0 = best scoring)
            centroid_score: The centroid's routing score (query-centroid similarity)
        
        Schema for R0_selected_centroids.parquet:
            - query_id: int32
            - q_token_id: int8
            - centroid_id: int32
            - rank: int8
            - centroid_score: float32
        """
        if isinstance(query_id, str):
            query_id = int(query_id)
        
        with self._lock:
            self._r0_buffer.append({
                "query_id": query_id,
                "q_token_id": q_token_id,
                "centroid_id": centroid_id,
                "rank": rank,
                "centroid_score": centroid_score
            })
            
            if len(self._r0_buffer) >= self.BUFFER_FLUSH_THRESHOLD:
                self._flush_r0_buffer()
    
    def _flush_r0_buffer(self) -> None:
        """Write R0 buffer to Parquet file and clear buffer."""
        if not self._r0_buffer:
            return
        
        schema = pa.schema([
            ("query_id", pa.int32()),
            ("q_token_id", pa.int8()),
            ("centroid_id", pa.int32()),
            ("rank", pa.int8()),
            ("centroid_score", pa.float32())
        ])
        
        table = pa.Table.from_pylist(self._r0_buffer, schema=schema)
        parquet_path = self.tier_b_dir / "R0_selected_centroids.parquet"
        
        if parquet_path.exists():
            existing = pq.read_table(parquet_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, parquet_path, compression='snappy')
        
        self._flush_counts["r0"] += 1
        self._r0_buffer = []
    
    # -------------------------------------------------------------------------
    # M4: Oracle winners (all embeddings per doc, ignoring routing)
    # -------------------------------------------------------------------------
    
    def record_m4_winner(
        self,
        query_id: int,
        q_token_id: int,
        doc_id: int,
        oracle_embedding_pos: int,
        oracle_score: float
    ) -> None:
        """
        Record M4 metric: oracle winner for a (query_token, doc) pair.
        
        M4 records which embedding would have been the MaxSim winner if
        all document embeddings were considered (ignoring routing/nprobe).
        
        Args:
            query_id: Query identifier
            q_token_id: Query token position (0-31)
            doc_id: Document identifier
            oracle_embedding_pos: Global position of the oracle winning embedding
            oracle_score: The oracle MaxSim score over ALL doc embeddings
        
        Schema for M4_oracle_winners.parquet:
            - query_id: int32
            - q_token_id: int8
            - doc_id: int32
            - oracle_embedding_pos: int64
            - oracle_score: float32
        """
        if isinstance(query_id, str):
            query_id = int(query_id)
        
        with self._lock:
            self._m4_buffer.append({
                "query_id": query_id,
                "q_token_id": q_token_id,
                "doc_id": doc_id,
                "oracle_embedding_pos": oracle_embedding_pos,
                "oracle_score": oracle_score
            })
            
            if len(self._m4_buffer) >= self.BUFFER_FLUSH_THRESHOLD:
                self._flush_m4_buffer()
    
    def record_m4_batch(
        self,
        query_id: int,
        doc_ids: List[int],
        oracle_pos: 'torch.Tensor',
        oracle_scores: 'torch.Tensor',
        num_tokens: int
    ) -> None:
        """
        Record M4 metrics for a batch of documents in a single call.
        
        This is ~50-100x faster than calling record_m4_winner() per token
        because it minimizes Python loop overhead.
        
        Args:
            query_id: Query identifier
            doc_ids: List of document IDs [num_docs]
            oracle_pos: (num_docs, num_tokens) tensor of winning embedding positions
            oracle_scores: (num_docs, num_tokens) tensor of oracle scores
            num_tokens: Number of actual query tokens
        
        Same schema as record_m4_winner - just batched.
        """
        if isinstance(query_id, str):
            query_id = int(query_id)
        
        # Convert tensors to numpy for faster iteration
        pos_np = oracle_pos.numpy()
        scores_np = oracle_scores.numpy()
        
        # Build batch of records
        batch = []
        for d_idx, doc_id in enumerate(doc_ids):
            for t in range(num_tokens):
                pos = int(pos_np[d_idx, t])
                score = float(scores_np[d_idx, t])
                
                # Only record valid winners (pos >= 0)
                if pos >= 0:
                    batch.append({
                        "query_id": query_id,
                        "q_token_id": t,
                        "doc_id": doc_id,
                        "oracle_embedding_pos": pos,
                        "oracle_score": score
                    })
        
        # Add batch to buffer under lock (single lock acquisition)
        with self._lock:
            self._m4_buffer.extend(batch)
            
            if len(self._m4_buffer) >= self.BUFFER_FLUSH_THRESHOLD:
                self._flush_m4_buffer()
    
    def _flush_m4_buffer(self) -> None:
        """Write M4 buffer to Parquet file and clear buffer."""
        if not self._m4_buffer:
            return
        
        schema = pa.schema([
            ("query_id", pa.int32()),
            ("q_token_id", pa.int8()),
            ("doc_id", pa.int32()),
            ("oracle_embedding_pos", pa.int64()),
            ("oracle_score", pa.float32())
        ])
        
        table = pa.Table.from_pylist(self._m4_buffer, schema=schema)
        parquet_path = self.tier_b_dir / "M4_oracle_winners.parquet"
        
        if parquet_path.exists():
            existing = pq.read_table(parquet_path)
            table = pa.concat_tables([existing, table])
        
        pq.write_table(table, parquet_path, compression='snappy')
        
        self._flush_counts["m4"] += 1
        self._m4_buffer = []
    
    # -------------------------------------------------------------------------
    # Metadata and finalization
    # -------------------------------------------------------------------------
    
    def save_metadata(
        self,
        dataset_info: Optional[Dict[str, Any]] = None,
        index_info: Optional[Dict[str, Any]] = None,
        warp_config: Optional[Dict[str, Any]] = None,
        candidate_doc_policy: Optional[Dict[str, Any]] = None,
        git_commit: Optional[str] = None,
        random_seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save run metadata to metadata.json.
        
        Schema follows MEASUREMENT_WISHES.MD specification.
        
        Args:
            dataset_info: Dict with name, split, num_queries
            index_info: Dict with path, num_centroids, num_embeddings, nbits
            warp_config: Dict with n_probe, t_prime, centroid_only
            candidate_doc_policy: Dict with type, k, sample_size
            git_commit: Git commit hash for reproducibility
            random_seed: Random seed used
            extra: Any additional metadata to include
        """
        metadata = {
            "run_id": self.run_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset_info or {},
            "index": index_info or {"path": self.index_path},
            "warp_config": warp_config or {},
            "candidate_doc_policy": candidate_doc_policy or {
                "type": "all_scored",
                "k": None,
                "sample_size": None
            },
            "reproducibility": {
                "git_commit": git_commit,
                "random_seed": random_seed
            },
            "storage": {
                "tier_a_complete": False,  # Updated by flush_all()
                "tier_b_complete": False,
                "tier_b_query_subset": None,
                "flush_counts": self._flush_counts.copy()
            }
        }
        
        if extra:
            metadata.update(extra)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def flush_all(self, tier_a_complete: bool = True, tier_b_complete: bool = False) -> None:
        """
        Flush all buffers and update metadata completion status.
        
        Call this at the end of a measurement run to ensure all data is written.
        
        Args:
            tier_a_complete: Mark Tier A as complete in metadata
            tier_b_complete: Mark Tier B as complete in metadata
        """
        with self._lock:
            self._flush_m1_buffer()
            self._flush_m3b_buffer()
            self._flush_r0_buffer()
            self._flush_m4_buffer()
        
        # Update metadata with completion status
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata["storage"]["tier_a_complete"] = tier_a_complete
            metadata["storage"]["tier_b_complete"] = tier_b_complete
            metadata["storage"]["flush_counts"] = self._flush_counts.copy()
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    
    def get_buffer_sizes(self) -> Dict[str, int]:
        """Return current buffer sizes for monitoring."""
        return {
            "m1": len(self._m1_buffer),
            "r0": len(self._r0_buffer),
            "m3a": len(self._m3a_buffer),
            "m3b": len(self._m3b_buffer),
            "m4": len(self._m4_buffer)
        }
    
    @property
    def is_enabled(self) -> bool:
        """Check if measurement collection is enabled."""
        return True  # Can be extended to support disabling


class NOPMeasurementCollector:
    """
    No-operation measurement collector for when measurements are disabled.
    
    All methods are no-ops, allowing code to call measurement methods
    without checking if collection is enabled.
    """
    
    def record_m1(self, query_id, q_token_id, centroid_id, num_token_token_sims):
        pass
    
    def record_m3b_winner(self, query_id, q_token_id, doc_id, winner_embedding_pos, winner_score):
        pass
    
    def save_metadata(self, **kwargs):
        pass
    
    def flush_all(self, **kwargs):
        pass
    
    def get_buffer_sizes(self):
        return {}
    
    @property
    def is_enabled(self):
        return False

class ExecutionTracker:
    """
    Tracks execution timing and optionally collects SQ2 measurements.
    
    This class serves two purposes:
    1. Performance profiling: Measures time spent in each step of the pipeline
    2. SQ2 measurements: Optionally collects metrics like M1 (token-level computation)
    
    The measurement collector is integrated here (rather than passed separately)
    because ExecutionTracker is already instantiated in many places throughout
    the codebase, reducing the risk of errors from missing collector arguments.
    
    Usage:
        # Without measurements (default, backward compatible)
        tracker = ExecutionTracker("WARP", steps=["Step1", "Step2"])
        
        # With measurements enabled
        tracker = ExecutionTracker(
            "WARP",
            steps=["Step1", "Step2"],
            measurement_run_id="2024-12-30_quora_nprobe8",
            measurement_output_dir="/mnt/warp_measurements"
        )
        
        with tracker.iteration():
            tracker.begin("Step1")
            # ... do work ...
            tracker.record_m1(query_id=0, q_token_id=1, centroid_id=42, num_sims=1500)
            tracker.end("Step1")
        
        # At end of run
        tracker.finalize_measurements()
    """
    
    def __init__(
        self,
        name: str,
        steps: List[str],
        measurement_run_id: Optional[str] = None,
        measurement_output_dir: Optional[str] = None,
        index_path: Optional[str] = None
    ):
        """
        Initialize the execution tracker.
        
        Args:
            name: Name of this tracker (e.g., "WARP", "XTR")
            steps: List of step names that will be tracked (in order)
            measurement_run_id: If provided, enables SQ2 measurement collection.
                              Use a descriptive ID like "2024-12-30_quora_nprobe8"
            measurement_output_dir: Base directory for measurements.
                                   Defaults to /mnt/warp_measurements
            index_path: Path to the WARP index (recorded in metadata)
        """
        self._name = name
        self._steps = steps
        self._num_iterations = 0
        self._time = None
        self._time_per_step = {}
        for step in steps:
            self._time_per_step[step] = 0
        self._iter_begin = None
        self._iter_time = 0
        self._iterating = False
        self._current_steps = []
        
        # Current iteration's query_id (set via record("query_id", ...))
        self._current_query_id: Optional[int] = None
        
        # Legacy in-memory metrics storage (for backward compatibility)
        # TODO: Deprecate in favor of MeasurementCollector
        self._legacy_metrics_current: Dict[str, Any] = {}
        self._legacy_metrics_all: List[Dict[str, Any]] = []
        self._legacy_lock = threading.Lock()
        
        # SQ2 Measurement collector (optional)
        if measurement_run_id is not None:
            self._measurement_collector = MeasurementCollector(
                run_id=measurement_run_id,
                output_dir=measurement_output_dir,
                index_path=index_path,
                create_dirs=True
            )
        else:
            self._measurement_collector = NOPMeasurementCollector()
        
        # M3 Tier B tracking flag (disabled by default due to memory overhead)
        self._m3_tracking_enabled = False
        
        # M4 Oracle tracking flag (disabled by default due to compute overhead)
        self._m4_tracking_enabled = False

    @property
    def measurements_enabled(self) -> bool:
        """Check if measurement collection is enabled for this tracker."""
        return self._measurement_collector.is_enabled
    
    def enable_m3_tracking(self, enabled: bool = True) -> None:
        """
        Enable or disable M3 Tier B winner tracking.
        
        M3 Tier B tracks per-token winners for top-k documents. This has
        additional memory overhead (~5.6 MB per query during processing)
        and is disabled by default.
        
        Args:
            enabled: Whether to enable M3 Tier B tracking
        """
        self._m3_tracking_enabled = enabled
    
    def enable_m4_tracking(self, enabled: bool = True) -> None:
        """
        Enable or disable M4 Oracle winner tracking.
        
        M4 Oracle computes what the true MaxSim winner would be for each
        (query_token, doc) pair if all doc embeddings were considered
        (ignoring routing). This has significant compute overhead (~30-50ms
        per query with C++ implementation) and is disabled by default.
        
        Args:
            enabled: Whether to enable M4 Oracle tracking
        """
        self._m4_tracking_enabled = enabled
    
    @property
    def m4_tracking_enabled(self) -> bool:
        """Check if M4 Oracle tracking is enabled."""
        return self._m4_tracking_enabled
    
    @property
    def measurement_collector(self):
        """Access the measurement collector directly (for advanced usage)."""
        return self._measurement_collector
    
    @property
    def current_query_id(self) -> Optional[int]:
        """Get the current query ID (set via record('query_id', ...))."""
        return self._current_query_id

    def next_iteration(self):
        """Start a new iteration (query)."""
        self._num_iterations += 1
        self._iterating = True
        self._current_steps = []
        self._iter_begin = time.time()
        self._current_query_id = None
        
        # Legacy metrics
        with self._legacy_lock:
            self._legacy_metrics_current = {}

    def end_iteration(self):
        """End the current iteration."""
        tok = time.time()
        if self._steps != self._current_steps:
            print(f"Warning: Expected steps {self._steps}, got {self._current_steps}")
        assert self._steps == self._current_steps, f"Step mismatch: {self._steps} != {self._current_steps}"
        self._iterating = False
        self._iter_time += tok - self._iter_begin
        
        # Legacy metrics
        with self._legacy_lock:
            self._legacy_metrics_all.append(self._legacy_metrics_current.copy())

    def iteration(self):
        """Context manager for iteration tracking."""
        return ExecutionTrackerIteration(self)

    def begin(self, name: str):
        """Begin timing a step."""
        assert self._time is None and self._iterating, f"Cannot begin '{name}' - not in iteration or previous step not ended"
        self._current_steps.append(name)
        self._time = time.time()

    def end(self, name: str):
        """End timing a step."""
        tok = time.time()
        assert self._current_steps[-1] == name, f"Step mismatch: ending '{name}' but current is '{self._current_steps[-1]}'"
        self._time_per_step[name] += tok - self._time
        self._time = None

    # -------------------------------------------------------------------------
    # Legacy record method (backward compatible)
    # -------------------------------------------------------------------------
    
    def record(self, name: str, value: Any):
        """
        Record a metric value (legacy interface).
        
        This stores metrics in memory for backward compatibility.
        For SQ2 measurements, use the dedicated record_m1(), etc. methods.
        
        Special handling for 'query_id': stores it for use by measurement methods.
        """
        # Capture query_id for measurement methods
        if name == "query_id":
            self._current_query_id = value
        
        # Legacy in-memory storage
        with self._legacy_lock:
            self._legacy_metrics_current[name] = value
    
    # -------------------------------------------------------------------------
    # SQ2 Measurement methods
    # -------------------------------------------------------------------------
    
    def record_m1(
        self,
        q_token_id: int,
        centroid_id: int,
        num_token_token_sims: int,
        query_id: Optional[int] = None
    ):
        """
        Record M1 metric: token-level computation for a centroid.
        
        Args:
            q_token_id: Query token position (0-31)
            centroid_id: The centroid that was scored
            num_token_token_sims: Number of embeddings scored in this centroid
            query_id: Query ID (optional, uses current iteration's query_id if not provided)
        """
        qid = query_id if query_id is not None else self._current_query_id
        if qid is None:
            # Silently skip if no query_id (measurements may not be fully set up)
            return
        
        self._measurement_collector.record_m1(
            query_id=qid,
            q_token_id=q_token_id,
            centroid_id=centroid_id,
            num_token_token_sims=num_token_token_sims
        )
    
    def record_m3_winner(
        self,
        q_token_id: int,
        doc_id: int,
        winner_embedding_pos: int,
        winner_score: float,
        query_id: Optional[int] = None
    ):
        """
        Record M3 Tier B metric: observed winner for a (query_token, doc) pair.
        
        Args:
            q_token_id: Query token position (0-31)
            doc_id: Document identifier
            winner_embedding_pos: Global position of the winning embedding
            winner_score: The MaxSim score for this (token, doc) pair
            query_id: Query ID (optional, uses current iteration's query_id if not provided)
        """
        qid = query_id if query_id is not None else self._current_query_id
        if qid is None:
            return
        
        self._measurement_collector.record_m3b_winner(
            query_id=qid,
            q_token_id=q_token_id,
            doc_id=doc_id,
            winner_embedding_pos=winner_embedding_pos,
            winner_score=winner_score
        )
    
    def record_r0_centroid(
        self,
        q_token_id: int,
        centroid_id: int,
        rank: int,
        centroid_score: float = 0.0,
        query_id: Optional[int] = None
    ):
        """
        Record R0 metric: selected centroid for a query token.
        
        Args:
            q_token_id: Query token position (0-31)
            centroid_id: The selected centroid ID
            rank: Rank among nprobe selections (0 = best scoring)
            centroid_score: The centroid's routing score (query-centroid similarity)
            query_id: Query ID (optional, uses current iteration's query_id if not provided)
        """
        qid = query_id if query_id is not None else self._current_query_id
        if qid is None:
            return
        
        self._measurement_collector.record_r0_centroid(
            query_id=qid,
            q_token_id=q_token_id,
            centroid_id=centroid_id,
            rank=rank,
            centroid_score=centroid_score
        )
    
    def record_m4_winner(
        self,
        q_token_id: int,
        doc_id: int,
        oracle_embedding_pos: int,
        oracle_score: float,
        query_id: Optional[int] = None
    ):
        """
        Record M4 metric: oracle winner for a (query_token, doc) pair.
        
        Args:
            q_token_id: Query token position (0-31)
            doc_id: Document identifier
            oracle_embedding_pos: Global position of the oracle winning embedding
            oracle_score: The oracle MaxSim score over ALL doc embeddings
            query_id: Query ID (optional, uses current iteration's query_id if not provided)
        """
        qid = query_id if query_id is not None else self._current_query_id
        if qid is None:
            return
        
        self._measurement_collector.record_m4_winner(
            query_id=qid,
            q_token_id=q_token_id,
            doc_id=doc_id,
            oracle_embedding_pos=oracle_embedding_pos,
            oracle_score=oracle_score
        )
    
    def record_m4_batch(
        self,
        query_id: int,
        doc_ids: List[int],
        oracle_pos: 'torch.Tensor',
        oracle_scores: 'torch.Tensor',
        num_tokens: int
    ):
        """
        Record M4 metrics for a batch of documents (optimized).
        
        This is ~50-100x faster than calling record_m4_winner() per token.
        """
        if query_id is None:
            query_id = self._current_query_id
        if query_id is None:
            return
        
        self._measurement_collector.record_m4_batch(
            query_id=query_id,
            doc_ids=doc_ids,
            oracle_pos=oracle_pos,
            oracle_scores=oracle_scores,
            num_tokens=num_tokens
        )
    
    def finalize_measurements(
        self,
        dataset_info: Optional[Dict[str, Any]] = None,
        index_info: Optional[Dict[str, Any]] = None,
        warp_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Finalize measurements: flush all buffers and save metadata.
        
        Call this at the end of a measurement run.
        
        Args:
            dataset_info: Dict with name, split, num_queries
            index_info: Dict with path, num_centroids, num_embeddings, nbits  
            warp_config: Dict with n_probe, t_prime, centroid_only
            **kwargs: Additional arguments passed to save_metadata
        """
        if not self.measurements_enabled:
            return
        
        self._measurement_collector.save_metadata(
            dataset_info=dataset_info,
            index_info=index_info,
            warp_config=warp_config,
            **kwargs
        )
        self._measurement_collector.flush_all(tier_a_complete=True)

    # -------------------------------------------------------------------------
    # Summary and serialization
    # -------------------------------------------------------------------------

    def summary(self, steps=None):
        """Get timing summary."""
        if steps is None:
            steps = self._steps
        iteration_time = self._iter_time / self._num_iterations
        breakdown = [
            (step, self._time_per_step[step] / self._num_iterations) for step in steps
        ]
        return iteration_time, breakdown

    def as_dict(self):
        """Serialize tracker state to dictionary."""
        return {
            "name": self._name,
            "steps": self._steps,
            "time_per_step": self._time_per_step,
            "num_iterations": self._num_iterations,
            "iteration_time": self._iter_time,
            "special_metrics": self._legacy_metrics_all,  # Legacy name for compatibility
        }

    @staticmethod
    def from_dict(data):
        """Deserialize tracker from dictionary."""
        tracker = ExecutionTracker(data["name"], data["steps"])
        tracker._time_per_step = data["time_per_step"]
        tracker._num_iterations = data["num_iterations"]
        tracker._iter_time = data["iteration_time"]
        if "special_metrics" in data:
            tracker._legacy_metrics_all = data["special_metrics"]
        return tracker

    def __getitem__(self, key):
        """Get average time for a step."""
        assert key in self._steps
        return self._time_per_step[key] / self._num_iterations

    def display(self, steps=None, bound=None):
        """Display timing breakdown as a bar chart."""
        iteration_time, breakdown = self.summary(steps)
        df = pd.DataFrame(
            {
                "Task": [x[0] for x in breakdown],
                "Duration": [x[1] * 1000 for x in breakdown],
            }
        )
        df["Start"] = df["Duration"].cumsum().shift(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 2))

        for i, task in enumerate(df["Task"]):
            start = df["Start"][i]
            duration = df["Duration"][i]
            ax.barh("Tasks", duration, left=start, height=0.5, label=task)

        plt.xlabel("Latency (ms)")
        accumulated = round(sum([x[1] for x in breakdown]) * 1000, 1)
        actual = round(iteration_time * 1000, 1)
        plt.title(
            f"{self._name} (iterations={self._num_iterations}, accumulated={accumulated}ms, actual={actual}ms)"
        )
        ax.set_yticks([])
        ax.set_ylabel("")

        if bound is not None:
            ax.set_xlim([0, bound])

        plt.legend()
        plt.show()


class NOPTracker:
    """
    No-operation tracker for when timing/measurements are disabled.
    
    All methods are no-ops. Useful for production runs where tracking
    overhead is not desired.
    """
    
    def __init__(self):
        self._measurement_collector = NOPMeasurementCollector()

    @property
    def measurements_enabled(self) -> bool:
        return False
    
    @property
    def m4_tracking_enabled(self) -> bool:
        return False
    
    @property  
    def measurement_collector(self):
        return self._measurement_collector

    def next_iteration(self):
        pass

    def begin(self, name):
        pass

    def end(self, name):
        pass

    def end_iteration(self):
        pass

    def iteration(self):
        return self  # Returns self as context manager
    
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        pass

    def summary(self):
        raise AssertionError("NOPTracker does not support summary()")

    def record(self, name, value):
        pass

    def record_m1(self, q_token_id, centroid_id, num_token_token_sims, query_id=None):
        pass
    
    def record_m3_winner(self, q_token_id, doc_id, winner_embedding_pos, winner_score, query_id=None):
        pass
    
    def record_r0_centroid(self, q_token_id, centroid_id, rank, centroid_score=0.0, query_id=None):
        pass
    
    def record_m4_winner(self, q_token_id, doc_id, oracle_embedding_pos, oracle_score, query_id=None):
        pass
    
    def finalize_measurements(self, **kwargs):
        pass

    def display(self):
        raise AssertionError("NOPTracker does not support display()")


def aggregate_trackers(tracker_dicts):
    """
    Aggregate multiple serialized tracker instances into a single tracker.
    
    Args:
        tracker_dicts: List of dictionaries from ExecutionTracker.as_dict()
    
    Returns:
        ExecutionTracker: A merged tracker with combined timing and metrics
    """
    if not tracker_dicts:
        raise ValueError("Cannot aggregate empty list of trackers")
    
    # Use the first tracker as a base
    merged = ExecutionTracker.from_dict(tracker_dicts[0])
    
    # Merge remaining trackers
    for t in tracker_dicts[1:]:
        for step in merged._steps:
            if step in t["time_per_step"]:
                merged._time_per_step[step] += t["time_per_step"][step]
        merged._num_iterations += t["num_iterations"]
        merged._iter_time += t["iteration_time"]
        if "special_metrics" in t:
            merged._legacy_metrics_all.extend(t["special_metrics"])
    
    return merged
