"""
Azure ML utilities for WARP dataset management.

Modules:
    config: Core configuration classes (AzureMLConfig, AssetConfig, etc.)
    datasets: Dataset registry and lookup functions
    storage: Blob upload/download utilities
    ml_client: Azure ML client helper
    cli: Shared CLI argument parsing

Scripts:
    upload_to_datastore.py: Upload and register assets
    download_from_datastore.py: Download assets
"""

from .config import AzureMLConfig, AssetConfig, AssetKind, DatasetConfig
from .datasets import get_asset_config, get_dataset_config, list_datasets, DATASETS
from .ml_client import get_ml_client
from .storage import upload_folder_to_datastore, download_from_datastore, is_folder_non_empty

__all__ = [
    # Config
    "AzureMLConfig",
    "AssetConfig",
    "AssetKind",
    "DatasetConfig",
    # Datasets
    "DATASETS",
    "get_asset_config",
    "get_dataset_config",
    "list_datasets",
    # ML Client
    "get_ml_client",
    # Storage
    "upload_folder_to_datastore",
    "download_from_datastore",
    "is_folder_non_empty",
]
