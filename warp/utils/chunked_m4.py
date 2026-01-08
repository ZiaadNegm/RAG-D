"""
Chunked M4 Processing Utility

Provides memory-efficient processing of large M4 (oracle winners) Parquet files
by reading in batches using Parquet predicate pushdown.

Problem: M4 files can be very large (22GB+ for 10K queries, 2.6B rows).
Loading the entire file into memory causes OOM errors on machines with <128GB RAM.

Solution: Read M4 in query batches using pd.read_parquet(filters=[...]) which
leverages Parquet's predicate pushdown to only load required row groups.

Usage:
    from warp.utils.chunked_m4 import ChunkedM4Processor
    
    processor = ChunkedM4Processor(m4_path, chunk_size=500)
    
    # Iterate over chunks
    for chunk_df in processor.iter_chunks():
        # Process chunk_df (contains ~500 queries worth of M4 data)
        process_chunk(chunk_df)
    
    # Or get query IDs first
    query_ids = processor.get_query_ids()
    
See docs/CHUNKED_M4_PROCESSING.md for background on the OOM issue.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Union
import warnings

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# Default chunk size: 500 queries per batch
# With ~264K rows/query, this yields ~132M rows/chunk ≈ 2.7GB RAM
DEFAULT_CHUNK_SIZE = 500


class ChunkedM4Processor:
    """
    Memory-efficient M4 Parquet processor using query-based chunking.
    
    Uses Parquet predicate pushdown to read only the required row groups
    for each batch of query IDs, avoiding loading the entire file into memory.
    
    Attributes:
        m4_path: Path to M4_oracle_winners.parquet file
        chunk_size: Number of queries to process per chunk
        query_ids: Array of unique query IDs in the file
    """
    
    def __init__(
        self,
        m4_path: Union[str, Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verbose: bool = True
    ):
        """
        Initialize the chunked M4 processor.
        
        Args:
            m4_path: Path to M4_oracle_winners.parquet file
            chunk_size: Number of queries to include per chunk (default: 500)
            verbose: Whether to show progress bar during iteration
        """
        self.m4_path = Path(m4_path)
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        # Validate file exists
        if not self.m4_path.exists():
            raise FileNotFoundError(f"M4 file not found: {self.m4_path}")
        
        # Lazy-loaded query IDs
        self._query_ids: Optional[List[int]] = None
        self._total_rows: Optional[int] = None
    
    def get_query_ids(self) -> List[int]:
        """
        Get list of unique query IDs in the M4 file.
        
        This reads only the query_id column, not the full file.
        
        Returns:
            Sorted list of unique query IDs
        """
        if self._query_ids is None:
            # Read only query_id column (very fast)
            query_col = pd.read_parquet(
                self.m4_path, 
                columns=['query_id']
            )['query_id']
            
            self._query_ids = sorted(query_col.unique().tolist())
            self._total_rows = len(query_col)
            
            if self.verbose:
                print(f"M4 file: {len(self._query_ids):,} queries, {self._total_rows:,} total rows")
        
        return self._query_ids
    
    @property
    def num_queries(self) -> int:
        """Total number of unique queries in M4 file."""
        return len(self.get_query_ids())
    
    @property
    def num_chunks(self) -> int:
        """Number of chunks based on chunk_size."""
        return (self.num_queries + self.chunk_size - 1) // self.chunk_size
    
    @property
    def total_rows(self) -> int:
        """Total number of rows in M4 file."""
        if self._total_rows is None:
            _ = self.get_query_ids()  # Trigger lazy load
        return self._total_rows or 0
    
    def iter_chunks(
        self,
        columns: Optional[List[str]] = None,
        show_progress: Optional[bool] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over M4 data in query-based chunks.
        
        Uses Parquet predicate pushdown to efficiently read only the
        rows for each batch of query IDs.
        
        Args:
            columns: Specific columns to read (default: all columns)
            show_progress: Override verbose setting for progress bar
        
        Yields:
            DataFrame containing M4 data for chunk_size queries at a time
        
        Example:
            processor = ChunkedM4Processor(m4_path, chunk_size=500)
            for chunk in processor.iter_chunks():
                # chunk contains ~500 queries worth of M4 data
                process(chunk)
        """
        query_ids = self.get_query_ids()
        show_progress = show_progress if show_progress is not None else self.verbose
        
        # Create chunk ranges
        chunk_ranges = [
            (i, min(i + self.chunk_size, len(query_ids)))
            for i in range(0, len(query_ids), self.chunk_size)
        ]
        
        # Progress bar over chunks
        iterator = chunk_ranges
        if show_progress:
            iterator = tqdm(
                chunk_ranges,
                desc="Processing M4 chunks",
                unit="chunk",
                total=len(chunk_ranges)
            )
        
        for start_idx, end_idx in iterator:
            chunk_qids = query_ids[start_idx:end_idx]
            
            # Use Parquet predicate pushdown to read only matching rows
            # This is the key optimization: only loads row groups containing these query_ids
            chunk_df = pd.read_parquet(
                self.m4_path,
                columns=columns,
                filters=[('query_id', 'in', chunk_qids)]
            )
            
            yield chunk_df
    
    def iter_chunks_with_ids(
        self,
        columns: Optional[List[str]] = None,
        show_progress: Optional[bool] = None
    ) -> Iterator[tuple]:
        """
        Iterate over M4 chunks, also returning the query IDs for each chunk.
        
        Yields:
            Tuple of (query_ids_list, chunk_dataframe)
        """
        query_ids = self.get_query_ids()
        show_progress = show_progress if show_progress is not None else self.verbose
        
        chunk_ranges = [
            (i, min(i + self.chunk_size, len(query_ids)))
            for i in range(0, len(query_ids), self.chunk_size)
        ]
        
        iterator = chunk_ranges
        if show_progress:
            iterator = tqdm(
                chunk_ranges,
                desc="Processing M4 chunks",
                unit="chunk",
                total=len(chunk_ranges)
            )
        
        for start_idx, end_idx in iterator:
            chunk_qids = query_ids[start_idx:end_idx]
            
            chunk_df = pd.read_parquet(
                self.m4_path,
                columns=columns,
                filters=[('query_id', 'in', chunk_qids)]
            )
            
            yield chunk_qids, chunk_df
    
    def get_file_info(self) -> dict:
        """
        Get metadata about the M4 Parquet file.
        
        Returns:
            Dictionary with file statistics
        """
        # Use pyarrow for detailed metadata
        parquet_file = pq.ParquetFile(self.m4_path)
        metadata = parquet_file.metadata
        
        return {
            'path': str(self.m4_path),
            'file_size_mb': self.m4_path.stat().st_size / (1024 * 1024),
            'num_row_groups': metadata.num_row_groups,
            'num_rows': metadata.num_rows,
            'num_columns': metadata.num_columns,
            'schema': [field.name for field in parquet_file.schema_arrow],
            'num_queries': self.num_queries,
            'chunk_size': self.chunk_size,
            'num_chunks': self.num_chunks,
            'avg_rows_per_query': metadata.num_rows / max(self.num_queries, 1),
        }
    
    def estimate_chunk_memory(self) -> dict:
        """
        Estimate memory usage per chunk.
        
        Returns:
            Dictionary with memory estimates
        """
        info = self.get_file_info()
        
        # Rough estimate: each row is ~21 bytes on disk, but 2-3x more in DataFrame
        avg_rows_per_chunk = info['avg_rows_per_query'] * self.chunk_size
        bytes_per_row_memory = 50  # Conservative estimate with pandas overhead
        
        chunk_memory_mb = (avg_rows_per_chunk * bytes_per_row_memory) / (1024 * 1024)
        
        return {
            'avg_rows_per_chunk': int(avg_rows_per_chunk),
            'estimated_chunk_memory_mb': chunk_memory_mb,
            'estimated_chunk_memory_gb': chunk_memory_mb / 1024,
            'recommended_available_ram_gb': chunk_memory_mb / 1024 * 3,  # 3x for safety
        }


def merge_partitioned_parquet(
    partition_dir: Path,
    output_path: Path,
    delete_partitions: bool = True,
    verbose: bool = True,
    streaming: bool = True
) -> None:
    """
    Merge partitioned Parquet files into a single file.
    
    Args:
        partition_dir: Directory containing partition_*.parquet files
        output_path: Path for merged output file
        delete_partitions: Whether to delete partition files after merge
        verbose: Whether to print progress
        streaming: Use streaming merge (low memory) vs concat (faster but high memory)
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    partition_files = sorted(partition_dir.glob("partition_*.parquet"))
    
    if not partition_files:
        raise FileNotFoundError(f"No partition files found in {partition_dir}")
    
    if verbose:
        print(f"Merging {len(partition_files)} partition files...")
    
    if streaming:
        # Streaming merge: write row-by-row without loading all into memory
        # Get schema from first file
        schema = pq.read_schema(partition_files[0])
        
        with pq.ParquetWriter(output_path, schema) as writer:
            total_rows = 0
            for pf in tqdm(partition_files, desc="Streaming merge", disable=not verbose):
                table = pq.read_table(pf)
                writer.write_table(table)
                total_rows += len(table)
                del table  # Free memory immediately
        
        if verbose:
            print(f"Written {total_rows:,} rows via streaming merge")
    else:
        # Original concat approach (high memory)
        dfs = []
        for pf in tqdm(partition_files, desc="Reading partitions", disable=not verbose):
            dfs.append(pd.read_parquet(pf))
        
        merged = pd.concat(dfs, ignore_index=True)
        
        if verbose:
            print(f"Writing merged file ({len(merged):,} rows)...")
        
        merged.to_parquet(output_path, index=False)
    
    if delete_partitions:
        if verbose:
            print("Cleaning up partition files...")
        for pf in partition_files:
            pf.unlink()
        # Also remove the partition directory
        try:
            partition_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue
    
    if verbose:
        print(f"Merged output saved to {output_path}")
