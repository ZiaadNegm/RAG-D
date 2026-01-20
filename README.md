# WARP with Routing Measurement Framework

This repository extends the [WARP retrieval engine](https://github.com/jlscheerer/xtr-warp) with instrumentation for measuring and analyzing centroid-based routing behavior in multi-vector retrieval systems.

## Overview

This codebase accompanies a thesis investigating **routing efficiency in ColBERT-style dense retrieval systems**. The original WARP engine has been extended with:

1. **Measurement Infrastructure** — Collects fine-grained metrics about token–centroid interactions during search
2. **Hypothesis Testing Framework** — Statistical analysis tools for testing research hypotheses about routing behavior
3. **Offline and Online Metrics** — Cluster-level properties computed both from the index structure and from query execution traces

## Repository Structure

```
├── warp/                      # Modified WARP engine with measurement hooks
│   ├── utils/
│   │   ├── tracker.py         # MeasurementCollector for M1/M3/M4/R0 metrics
│   │   └── derived_metrics.py # Compute M2/M5/M6 from raw measurements
│   └── engine/
│       └── utils/
│           ├── oracle_scorer.py   # Ground-truth MaxSim computation
│           └── reverse_index.py   # Document ID → embedding lookup
│
├── hypothesis/                # Hypothesis testing framework
│   ├── configs/               # Experiment configurations
│   ├── data/                  # Data loading and frame construction
│   ├── hypotheses/            # Individual hypothesis implementations
│   ├── stats/                 # Statistical utilities
│   └── viz/                   # Visualization utilities
│
├── experiments/               # Experiment scripts
│   ├── run_full_metrics.py    # Main metrics collection pipeline
│   └── thread_scaling_analysis.py
│
├── scripts/                   # Utility scripts
│   ├── compute_derived_metrics.py
│   ├── compute_golden_metrics.py
│   ├── compute_offline_cluster_properties.py
│   └── compute_online_cluster_properties.py
│
└── utility/                   # Execution utilities
```

## Metrics Collected

### Raw Measurements (collected during search)

| Metric | Description |
|--------|-------------|
| **M1** | Total token-level similarity computations per centroid |
| **M3** | Influential interactions (actual top-k winners) |
| **M4** | Oracle interactions (ground-truth winners) |
| **R0** | Selected centroids per query token |

### Derived Measurements

| Metric | Description |
|--------|-------------|
| **M2** | Redundant computation (M1 − M3) |
| **M5** | Routing misses (in M4 but not M3) |
| **M6** | Missed centroid aggregation |

### Cluster Properties

**Offline (computed from index):**
- A1: Cluster size (number of embeddings)
- A2: Document diversity (unique documents per cluster)
- A3: Average intra-cluster distance
- A5: Size rank

**Online (computed from query traces):**
- B1–B4: Traffic and hubness metrics
- C1–C6: Query-level routing statistics

## Installation

```bash
# Create and activate environment
conda env create -f conda_env.yml
conda activate warp

# Or for CPU-only:
conda env create -f conda_env_cpu.yml
```

### Environment Variables

Create a `.env` file with:

```bash
INDEX_ROOT=/path/to/indexes
EXPERIMENT_ROOT=/path/to/experiments
BEIR_COLLECTION_PATH=/path/to/beir/data
```

## Usage

### 1. Build or Download an Index

```bash
# Build from scratch
python build_msmarco_index.py

# Or download pre-built
python download_msmarco_index.py
```

### 2. Collect Measurements

```bash
python experiments/run_full_metrics.py \
    --index-path $INDEX_ROOT/msmarco \
    --dataset beir/quora \
    --output-dir /path/to/output
```

### 3. Run Hypothesis Analysis

```bash
python hypothesis/run_all_production.py
```

## Key Modifications to WARP

This repository modifies the following components of the original WARP engine:

| File | Modification |
|------|--------------|
| `warp/utils/tracker.py` | Added `MeasurementCollector` class for recording M1/M3/M4/R0 |
| `warp/engine/candidate_generation.py` | Integrated measurement hooks for token-centroid interactions |
| `warp/engine/candidate_generation.cpp` | Added winner-tracking variants for M3 computation |
| `warp/engine/ext.cpp` | New Python bindings for winner-tracking functions |

New modules added:
- `warp/utils/derived_metrics.py` — Compute M2, M5, M6
- `warp/engine/utils/oracle_scorer.py` — Ground-truth computation
- `warp/engine/utils/reverse_index.py` — Reverse document lookup

## Hypotheses Tested

The `hypothesis/hypotheses/` directory contains implementations for testing various research hypotheses:

- **H3**: Fallback availability and document diversity
- **H4**: Concentration and redundancy
- **H5**: Dispersion and misses
- **H10**: Hubness and redundancy correlation
- **H15**: Miss severity analysis
- **H17**: Borderline cluster behavior

## Citation

This work builds on:

```bibtex
@article{warp2024,
  title={WARP: An Efficient Engine for Contextualized Multi-Vector Retrieval},
  author={Scheerer, Jonas L.},
  year={2024}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- Original WARP implementation by [Jonas L. Scheerer](https://github.com/jlscheerer/xtr-warp)
- ColBERTv2/PLAID by Stanford FutureData
- XTR by Google DeepMind
