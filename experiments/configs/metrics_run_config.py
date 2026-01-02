"""
Metrics Run Configuration

Defines test and production configurations for running all SQ2 metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class RawMeasurementsConfig:
    """Configuration for M1, M3, M4, R0 collection."""
    
    # Query scope
    num_queries: int = 100
    
    # WARP search parameters
    nprobe: int = 16
    k: int = 100  # Top-k documents (affects M3 scope)
    nbits: int = 4
    
    # Pipeline settings
    fused_ext: bool = False  # Must be False for M4 tracking
    num_threads: int = 2  # Must be >=2 for parallel pipeline (M4)
    
    # Storage
    output_dir: str = "/mnt/tmp/warp_measurements"
    buffer_flush_threshold: int = 10_000
    
    # Dataset
    collection: str = "beir"
    dataset: str = "quora"
    datasplit: str = "test"


@dataclass
class OfflineMetricsConfig:
    """Configuration for A1, A2, A3, A5, B5 computation."""
    
    # A5 (Dispersion) sampling
    a5_sample_fraction: float = 0.25
    a5_min_samples: int = 100
    a5_max_samples: int = 10_000
    a5_seed: int = 42
    
    # B5 (Isolation) parameters
    b5_k_neighbors: int = 10


@dataclass
class DerivedMetricsConfig:
    """Configuration for M2, M5, M6 computation."""
    
    # M5/M6 scope
    top_k_only: bool = False  # If True, only compute for top-k docs


@dataclass  
class OnlineMetricsConfig:
    """Configuration for A4, A6, B1-B4, C1-C6 computation."""
    
    # Hub classification (B4)
    hub_percentile: float = 95.0
    bad_yield_threshold: float = 0.02  # Lowered from 0.1 (given ~0.6% mean yield)
    good_yield_threshold: float = 0.05  # Lowered from 0.3
    
    # Parallelization
    num_workers: int = 8
    
    # Recall@k values (C3)
    recall_k_values: List[int] = field(default_factory=lambda: [10, 100, 1000])


@dataclass
class MetricsRunConfig:
    """Complete configuration for a metrics run."""
    
    name: str
    description: str
    
    # Index path
    index_path: str = "/mnt/datasets/index/beir-quora.split=test.nbits=4"
    
    # Sub-configurations
    raw: RawMeasurementsConfig = field(default_factory=RawMeasurementsConfig)
    offline: OfflineMetricsConfig = field(default_factory=OfflineMetricsConfig)
    derived: DerivedMetricsConfig = field(default_factory=DerivedMetricsConfig)
    online: OnlineMetricsConfig = field(default_factory=OnlineMetricsConfig)
    
    def get_run_output_dir(self) -> str:
        """Get the output directory for this run."""
        return f"{self.raw.output_dir}/{self.name}"


# =============================================================================
# TEST CONFIGURATION
# =============================================================================
# Purpose: Quick validation run to catch errors before production
# Expected runtime: ~2-5 minutes
# Expected storage: ~100-200 MB

TEST_CONFIG = MetricsRunConfig(
    name="test_run",
    description="Small validation run (50 queries) to catch errors before production",
    index_path="/mnt/datasets/index/beir-quora.split=test.nbits=4",
    raw=RawMeasurementsConfig(
        num_queries=50,           # Small sample for quick validation
        nprobe=16,
        k=100,
        nbits=4,
        fused_ext=False,
        num_threads=2,
        output_dir="/mnt/tmp/warp_measurements",
        buffer_flush_threshold=10_000,
        collection="beir",
        dataset="quora",
        datasplit="test",
    ),
    offline=OfflineMetricsConfig(
        a5_sample_fraction=0.25,
        a5_min_samples=100,
        a5_max_samples=10_000,
        a5_seed=42,
        b5_k_neighbors=10,
    ),
    derived=DerivedMetricsConfig(
        top_k_only=False,
    ),
    online=OnlineMetricsConfig(
        hub_percentile=95.0,
        bad_yield_threshold=0.02,
        good_yield_threshold=0.05,
        num_workers=8,
        recall_k_values=[10, 100, 1000],
    ),
)


# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================
# Purpose: Full-scale analysis for SQ2 investigation
# Expected runtime: ~30-60 minutes (depends on num_queries)
# Expected storage: ~10-15 GB for 10K queries

PRODUCTION_CONFIG = MetricsRunConfig(
    name="production_beir_quora",
    description="Full production run on BEIR-quora test set for SQ2 analysis",
    index_path="/mnt/datasets/index/beir-quora.split=test.nbits=4",
    raw=RawMeasurementsConfig(
        num_queries=10000,        # Full test set
        nprobe=16,
        k=1000,                   # Deeper ranking analysis
        nbits=4,
        fused_ext=False,
        num_threads=4,            # More threads for speed
        output_dir="/mnt/tmp/warp_measurements",
        buffer_flush_threshold=10_000,
        collection="beir",
        dataset="quora",
        datasplit="test",
    ),
    offline=OfflineMetricsConfig(
        a5_sample_fraction=0.25,
        a5_min_samples=100,
        a5_max_samples=10_000,
        a5_seed=42,
        b5_k_neighbors=10,
    ),
    derived=DerivedMetricsConfig(
        top_k_only=False,         # Full analysis
    ),
    online=OnlineMetricsConfig(
        hub_percentile=95.0,
        bad_yield_threshold=0.02,
        good_yield_threshold=0.05,
        num_workers=16,           # More workers for production
        recall_k_values=[10, 100, 1000],
    ),
)


def print_config(config: MetricsRunConfig) -> None:
    """Print configuration summary."""
    print("=" * 70)
    print(f"Configuration: {config.name}")
    print(f"Description: {config.description}")
    print("=" * 70)
    
    print("\n[Raw Measurements]")
    print(f"  num_queries:    {config.raw.num_queries}")
    print(f"  nprobe:         {config.raw.nprobe}")
    print(f"  k:              {config.raw.k}")
    print(f"  nbits:          {config.raw.nbits}")
    print(f"  num_threads:    {config.raw.num_threads}")
    print(f"  output_dir:     {config.raw.output_dir}")
    print(f"  dataset:        {config.raw.collection}/{config.raw.dataset}/{config.raw.datasplit}")
    
    print("\n[Offline Metrics]")
    print(f"  a5_sample_fraction: {config.offline.a5_sample_fraction}")
    print(f"  b5_k_neighbors:     {config.offline.b5_k_neighbors}")
    
    print("\n[Derived Metrics]")
    print(f"  top_k_only:     {config.derived.top_k_only}")
    
    print("\n[Online Metrics]")
    print(f"  hub_percentile:       {config.online.hub_percentile}")
    print(f"  bad_yield_threshold:  {config.online.bad_yield_threshold}")
    print(f"  good_yield_threshold: {config.online.good_yield_threshold}")
    print(f"  num_workers:          {config.online.num_workers}")
    print(f"  recall_k_values:      {config.online.recall_k_values}")
    
    print("\n[Storage]")
    print(f"  index_path:     {config.index_path}")
    print(f"  run_output_dir: {config.get_run_output_dir()}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METRICS RUN CONFIGURATIONS")
    print("=" * 70)
    
    print("\n### TEST CONFIGURATION ###")
    print_config(TEST_CONFIG)
    
    print("\n### PRODUCTION CONFIGURATION ###")
    print_config(PRODUCTION_CONFIG)
