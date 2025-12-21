#!/usr/bin/env python3
"""
Download dataset assets from Azure ML.

Usage:
    python scripts/azure/download_from_datastore.py -d beir_quora -k data
    python scripts/azure/download_from_datastore.py -d beir_quora -k index
    python scripts/azure/download_from_datastore.py -d beir_quora -k data --force
    python scripts/azure/download_from_datastore.py --list
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.azure.config import AzureMLConfig
from scripts.azure.datasets import get_asset_config, list_datasets
from scripts.azure.ml_client import get_ml_client
from scripts.azure.storage import download_from_datastore, is_folder_non_empty
from scripts.azure.cli import add_common_arguments, validate_common_arguments


def download_asset(
    dataset_key: str,
    kind: str,
    version: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Download a Data asset from Azure ML to the configured local path.
    
    Args:
        dataset_key: Dataset identifier (e.g., 'beir_quora')
        kind: Asset type ('data' or 'index')
        version: Asset version (uses default from config if None)
        force: If True, re-download even if local folder exists
    """
    # Get configuration
    azure_config = AzureMLConfig.from_env()
    asset_config = get_asset_config(dataset_key, kind)
    
    # Determine version
    version = version or asset_config.default_version
    
    # Check if local path already exists
    local_path = asset_config.local_path
    try:
        if is_folder_non_empty(local_path) and not force:
            print(f"Local folder already exists and is non-empty:")
            print(f"  {local_path}")
            print(f"Use --force to re-download and overwrite.")
            return
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Get ML client
    ml_client = get_ml_client(azure_config)
    
    # Get the data asset
    asset_name = asset_config.asset_name
    print(f"Fetching asset info...")
    print(f"  Name: {asset_name}")
    print(f"  Version: {version}")
    
    try:
        data_asset = ml_client.data.get(name=asset_name, version=version)
    except Exception as e:
        print(f"ERROR: Could not find asset '{asset_name}' version '{version}'")
        print(f"  {e}")
        print(f"\nMake sure the asset has been uploaded first.")
        sys.exit(1)
    
    print(f"  Remote path: {data_asset.path}")
    print()
    
    # Download
    try:
        download_from_datastore(
            ml_client=ml_client,
            datastore_uri=data_asset.path,
            local_path=local_path,
        )
        
        # Show what was downloaded
        if local_path.is_dir():
            files = list(local_path.iterdir())
            print(f"  Contains {len(files)} items")
            for f in files[:10]:  # Show first 10
                print(f"    - {f.name}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset assets from Azure ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_common_arguments(parser)
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list_datasets:
        list_datasets()
        return
    
    validate_common_arguments(args, parser)
    
    download_asset(
        dataset_key=args.dataset,
        kind=args.kind,
        version=args.version,
        force=args.force,
    )


if __name__ == "__main__":
    main()
