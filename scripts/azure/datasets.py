"""
Dataset registry for Azure ML asset management.

Defines available datasets (BEIR, LoTTE) and their configurations.
"""

from pathlib import Path
from typing import Dict

from .config import (
    AssetConfig,
    AssetKind,
    DatasetConfig,
    BEIR_BASE_PATH,
    INDEX_ROOT,
    LOTTE_BASE_PATH,
)


# =============================================================================
# Dataset Configuration Generators
# =============================================================================

def _beir_dataset_config(dataset_name: str) -> DatasetConfig:
    """Generate config for a BEIR dataset.
    
    Expected local structure:
        ${BEIR_COLLECTION_PATH}/<dataset>/collection.tsv
        ${BEIR_COLLECTION_PATH}/<dataset>/questions.test.tsv
        ...
    """
    return DatasetConfig(
        data=AssetConfig(
            asset_name=f"beir-{dataset_name}-data",
            local_path=BEIR_BASE_PATH / dataset_name,
            base_remote_dir=f"beir/{dataset_name}/data",
            description=f"BEIR {dataset_name} dataset (corpus, queries, qrels)",
            default_version="1",
        ),
        index=AssetConfig(
            asset_name=f"beir-{dataset_name}-index-nbits4",
            local_path=INDEX_ROOT / f"beir-{dataset_name}.split=test.nbits=4",
            base_remote_dir=f"beir/{dataset_name}/index",
            description=f"WARP index for BEIR {dataset_name} (nbits=4, split=test)",
            default_version="1",
        ),
    )


def _lotte_dataset_config(dataset_name: str) -> DatasetConfig:
    """Generate config for a LoTTE dataset.
    
    Expected local structure:
        ${LOTTE_COLLECTION_PATH}/<dataset>/test/collection.tsv
        ${LOTTE_COLLECTION_PATH}/<dataset>/test/questions.search.tsv
        ...
    """
    return DatasetConfig(
        data=AssetConfig(
            asset_name=f"lotte-{dataset_name}-data",
            local_path=LOTTE_BASE_PATH / dataset_name,
            base_remote_dir=f"lotte/{dataset_name}/data",
            description=f"LoTTE {dataset_name} dataset (includes test split)",
            default_version="1",
        ),
        index=AssetConfig(
            asset_name=f"lotte-{dataset_name}-index-nbits4",
            local_path=INDEX_ROOT / f"lotte-{dataset_name}.split=test.nbits=4",
            base_remote_dir=f"lotte/{dataset_name}/index",
            description=f"WARP index for LoTTE {dataset_name} (nbits=4, split=test)",
            default_version="1",
        ),
    )


# =============================================================================
# Dataset Registry
# =============================================================================

# Available dataset names
BEIR_DATASETS = ["quora", "nfcorpus", "scifact", "scidocs", "fiqa", "webis-touche2020"]
LOTTE_DATASETS = ["writing", "recreation", "science", "technology", "lifestyle", "pooled"]

# Build the registry
DATASETS: Dict[str, DatasetConfig] = {}

for ds in BEIR_DATASETS:
    DATASETS[f"beir_{ds}"] = _beir_dataset_config(ds)

for ds in LOTTE_DATASETS:
    DATASETS[f"lotte_{ds}"] = _lotte_dataset_config(ds)


# =============================================================================
# Lookup Functions
# =============================================================================

def get_dataset_config(dataset_key: str) -> DatasetConfig:
    """Get configuration for a dataset by key."""
    if dataset_key not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset: '{dataset_key}'\n"
            f"Available datasets: {available}"
        )
    return DATASETS[dataset_key]


def get_asset_config(dataset_key: str, kind: str) -> AssetConfig:
    """Get configuration for a specific asset (data or index)."""
    dataset_config = get_dataset_config(dataset_key)
    
    # Normalize kind to enum
    try:
        kind_enum = AssetKind(kind)
    except ValueError:
        valid = [k.value for k in AssetKind]
        raise ValueError(f"Invalid kind: '{kind}'. Must be one of: {valid}")
    
    if kind_enum == AssetKind.DATA:
        return dataset_config.data
    elif kind_enum == AssetKind.INDEX:
        return dataset_config.index
    else:
        raise ValueError(f"Unhandled asset kind: {kind_enum}")


def list_datasets() -> None:
    """Print all available datasets and their configurations."""
    print("Available datasets:\n")
    for key, config in sorted(DATASETS.items()):
        print(f"  {key}:")
        print(f"    data:  {config.data.local_path}")
        print(f"    index: {config.index.local_path}")
        print()
