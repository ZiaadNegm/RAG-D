"""
Configuration management for hypothesis testing.

Provides three runtime tiers:
- smoke: Fast sanity checks (5-50 queries, minutes)
- dev: Signal exploration (500-2000 queries, hours)  
- prod: Final statistical analysis (10k+ queries, days)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class DataPaths:
    """Paths to data sources and outputs."""
    # Input paths
    index_path: str = "/mnt/datasets/index/beir-quora.split=test.nbits=4"
    run_dir: str = "/mnt/tmp/online_dev_v2/runs/m4_e2e_20260102_123805"
    
    # Output paths (on high-capacity disk)
    output_root: str = "/mnt/hypothesis_outputs"
    
    @property
    def offline_properties_path(self) -> str:
        return f"{self.index_path}/cluster_properties_offline.parquet"
    
    @property
    def tier_a_dir(self) -> str:
        return f"{self.run_dir}/tier_a"
    
    @property
    def tier_b_dir(self) -> str:
        return f"{self.run_dir}/tier_b"
    
    @property
    def online_properties_dir(self) -> str:
        return f"{self.run_dir}/cluster_properties_online"


@dataclass  
class ProcessingConfig:
    """Configuration for data processing."""
    # Chunking to prevent OOM
    m4_chunk_size: int = 500  # Queries per M4 chunk
    max_memory_gb: float = 16.0  # Target memory ceiling
    
    # Parallelization
    num_workers: int = 8
    parallel_reads: bool = True
    
    # Caching
    cache_standardized_tables: bool = True
    cache_dir: str = "/mnt/hypothesis_outputs/cache"


@dataclass
class RuntimeConfig:
    """Full runtime configuration."""
    name: str  # smoke, dev, prod
    
    # Query scope
    num_queries: Optional[int] = None  # None = all available
    query_sample_seed: int = 42
    
    # Paths
    paths: DataPaths = field(default_factory=DataPaths)
    
    # Processing
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Verbosity
    verbose: bool = True
    progress_bars: bool = True


def load_config(name: str = "dev", override_run_dir: Optional[str] = None) -> RuntimeConfig:
    """
    Load a named configuration.
    
    Args:
        name: Config name ("smoke", "dev", or "prod")
        override_run_dir: Optional override for the measurement run directory
        
    Returns:
        RuntimeConfig with appropriate settings
    """
    configs_dir = Path(__file__).parent
    config_path = configs_dir / f"{name}.yaml"
    
    if not config_path.exists():
        raise ValueError(f"Config '{name}' not found at {config_path}")
    
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    # Build config objects
    paths = DataPaths(
        index_path=raw.get("paths", {}).get("index_path", DataPaths.index_path),
        run_dir=override_run_dir or raw.get("paths", {}).get("run_dir", DataPaths.run_dir),
        output_root=raw.get("paths", {}).get("output_root", DataPaths.output_root),
    )
    
    processing = ProcessingConfig(
        m4_chunk_size=raw.get("processing", {}).get("m4_chunk_size", 500),
        max_memory_gb=raw.get("processing", {}).get("max_memory_gb", 16.0),
        num_workers=raw.get("processing", {}).get("num_workers", 8),
        parallel_reads=raw.get("processing", {}).get("parallel_reads", True),
        cache_standardized_tables=raw.get("processing", {}).get("cache_standardized_tables", True),
        cache_dir=raw.get("processing", {}).get("cache_dir", "/mnt/hypothesis_outputs/cache"),
    )
    
    config = RuntimeConfig(
        name=name,
        num_queries=raw.get("num_queries"),
        query_sample_seed=raw.get("query_sample_seed", 42),
        paths=paths,
        processing=processing,
        verbose=raw.get("verbose", True),
        progress_bars=raw.get("progress_bars", True),
    )
    
    return config


def ensure_output_dirs(config: RuntimeConfig) -> None:
    """Create output directories if they don't exist."""
    dirs = [
        config.paths.output_root,
        config.processing.cache_dir,
        f"{config.paths.output_root}/figures",
        f"{config.paths.output_root}/tables",
        f"{config.paths.output_root}/reports",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
