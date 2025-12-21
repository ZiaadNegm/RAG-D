#!/usr/bin/env python3
"""
Upload and register dataset assets to Azure ML.

Usage:
    python scripts/azure/upload_to_datastore.py -d beir_quora -k data
    python scripts/azure/upload_to_datastore.py -d beir_quora -k index
    python scripts/azure/upload_to_datastore.py -d beir_quora -k data --force
    python scripts/azure/upload_to_datastore.py --list
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

# Add parent to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.azure.config import AzureMLConfig
from scripts.azure.datasets import get_asset_config, list_datasets
from scripts.azure.ml_client import get_ml_client
from scripts.azure.storage import upload_folder_to_datastore
from scripts.azure.cli import add_common_arguments, validate_common_arguments


def upload_asset(
    dataset_key: str,
    kind: str,
    version: Optional[str] = None,
    force: bool = False,
    skip_upload: bool = False,
) -> None:
    """
    Upload a local folder to Azure ML datastore and register as Data asset.
    
    Args:
        dataset_key: Dataset identifier (e.g., 'beir_quora')
        kind: Asset type ('data' or 'index')
        version: Asset version (uses default from config if None)
        force: If True, overwrite existing asset
        skip_upload: If True, only register the asset (assumes data already uploaded)
    """
    # Get configuration
    azure_config = AzureMLConfig.from_env()
    asset_config = get_asset_config(dataset_key, kind)
    
    # Determine version
    version = version or asset_config.default_version
    
    # Validate local path exists (unless skipping upload)
    local_path = asset_config.local_path
    if not skip_upload:
        if not local_path.exists():
            print(f"ERROR: Local path does not exist: {local_path}")
            print(f"Make sure the {kind} has been created/downloaded first.")
            sys.exit(1)
        
        if not local_path.is_dir():
            print(f"ERROR: Local path is not a directory: {local_path}")
            sys.exit(1)
    
    # Get ML client
    ml_client = get_ml_client(azure_config)
    
    # Check if asset already exists
    asset_name = asset_config.asset_name
    try:
        existing = ml_client.data.get(name=asset_name, version=version)
        if not force:
            print(f"Asset already exists: {asset_name} v{version}")
            print(f"  Path: {existing.path}")
            print(f"Use --force to overwrite.")
            return
        print(f"Asset exists, will overwrite (--force): {asset_name} v{version}")
    except ResourceNotFoundError:
        # Asset doesn't exist, good to proceed
        pass
    except Exception as e:
        # Unexpected error (auth, network, etc.) - don't swallow
        print(f"WARNING: Could not check if asset exists: {e}")
        print("Proceeding with upload attempt...")
    
    # Compute remote path and datastore URI
    remote_path = asset_config.remote_path(version)
    datastore_uri = asset_config.datastore_uri(azure_config.datastore_name, version)
    
    print(f"=== Asset Registration ===")
    print(f"  Name: {asset_name}")
    print(f"  Version: {version}")
    print(f"  Local path: {local_path}")
    print(f"  Datastore: {azure_config.datastore_name}")
    print(f"  Remote path: {remote_path}")
    print(f"  Datastore URI: {datastore_uri}")
    print(f"  Description: {asset_config.description}")
    print()
    
    # Step 1: Upload files to blob storage (unless skipped)
    if not skip_upload:
        upload_folder_to_datastore(
            ml_client=ml_client,
            local_path=local_path,
            datastore_name=azure_config.datastore_name,
            remote_path=remote_path,
        )
        print()
    else:
        print("Skipping upload (--skip-upload flag set)")
        print()
    
    # Step 2: Register Data asset pointing to the datastore URI
    # NOTE: We use datastore_uri, NOT local_path - this prevents auto-upload
    print(f"Registering Data asset...")
    
    data_asset = Data(
        name=asset_name,
        version=version,
        description=asset_config.description,
        path=datastore_uri,  # Use datastore URI, not local path!
        type=AssetTypes.URI_FOLDER,
    )
    
    try:
        created = ml_client.data.create_or_update(data_asset)
        print(f"SUCCESS: Created data asset")
        print(f"  Name: {created.name}")
        print(f"  Version: {created.version}")
        print(f"  Path: {created.path}")
    except ResourceExistsError as e:
        print(f"ERROR: Asset already exists and cannot be overwritten: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to create asset: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload and register dataset assets to Azure ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_common_arguments(parser)
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading files; only register the Data asset (assumes data already in datastore)",
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list_datasets:
        list_datasets()
        return
    
    validate_common_arguments(args, parser)
    
    upload_asset(
        dataset_key=args.dataset,
        kind=args.kind,
        version=args.version,
        force=args.force,
        skip_upload=args.skip_upload,
    )


if __name__ == "__main__":
    main()
