"""
Hypothesis Testing Framework for WARP Metrics Analysis

This package provides a structured approach to testing hypotheses about
WARP's centroid-based routing behavior using collected metrics.

Structure:
    hypothesis/
    ├── __init__.py              # This file
    ├── configs/                 # Runtime configs (smoke/dev/prod)
    │   ├── smoke.yaml          # 5-50 queries, ~minutes
    │   ├── dev.yaml            # 500-2000 queries, ~hours  
    │   └── prod.yaml           # 10k+ queries, ~days
    ├── data/                    # Data loading and standardized tables
    │   ├── loader.py           # Chunked/parallel data loading
    │   └── standardized_tables.py  # cluster_frame, etc.
    ├── stats/                   # Statistical utilities
    │   └── core.py             # Stratification, tests, correlations
    ├── viz/                     # Visualization utilities
    │   └── core.py             # Standard plots for hypothesis testing
    ├── hypotheses/              # Individual hypothesis implementations
    │   └── template.py         # Example hypothesis structure
    └── outputs/                 # Symlink to /mnt/hypothesis_outputs

Key Data Products:
    - cluster_frame: One row per centroid, joins offline (A-series), 
      hubness (B-series), routing (C-series), and M-series aggregates
    - query_frame: One row per query with routing/miss summaries
    - interaction_frame: Query-token-doc level for fine-grained analysis

Usage:
    from hypothesis.data.standardized_tables import ClusterFrameBuilder
    from hypothesis.stats.core import stratify_by_column, correlation_matrix
    from hypothesis.viz.core import plot_dispersion_vs_misses
    
    # Load config
    from hypothesis.configs import load_config
    config = load_config("dev")
    
    # Build cluster_frame
    builder = ClusterFrameBuilder(config)
    cluster_frame = builder.build()
"""

__version__ = "0.1.0"
