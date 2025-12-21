"""
Azure blob storage utilities.

Functions for uploading/downloading files to/from Azure ML datastores.
"""

import subprocess
from pathlib import Path

import azure.ai.ml._artifacts._artifact_utilities as artifact_utils


def upload_folder_to_datastore(
    ml_client,
    local_path: Path,
    datastore_name: str,
    remote_path: str,
) -> None:
    """
    Upload a local folder to the storage account backing a datastore.
    
    Uses Azure CLI `az storage blob upload-batch` for the upload.
    
    Args:
        ml_client: Azure ML client (used to get datastore info)
        local_path: Local folder to upload
        datastore_name: Name of the Azure ML datastore
        remote_path: Destination path within the datastore
    
    Raises:
        RuntimeError: If the upload fails
    """
    # Get datastore details
    datastore = ml_client.datastores.get(datastore_name)
    
    account_name = datastore.account_name
    container_name = datastore.container_name
    
    print(f"Uploading to blob storage...")
    print(f"  Storage account: {account_name}")
    print(f"  Container: {container_name}")
    print(f"  Source: {local_path}")
    print(f"  Destination path: {remote_path}")
    print()
    
    # Use az storage blob upload-batch
    cmd = [
        "az", "storage", "blob", "upload-batch",
        "--account-name", account_name,
        "--destination", container_name,
        "--destination-path", remote_path,
        "--source", str(local_path),
        "--overwrite",
        "--auth-mode", "login",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Upload failed with exit code {result.returncode}")
    
    print()
    print("Upload complete.")


def download_from_datastore(
    ml_client,
    datastore_uri: str,
    local_path: Path,
) -> None:
    """
    Download files from an Azure ML datastore URI to a local path.
    
    Args:
        ml_client: Azure ML client
        datastore_uri: Full Azure ML datastore URI (azureml://datastores/...)
        local_path: Local destination path
    
    Raises:
        Exception: If the download fails
    """
    # Create parent directories if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading to: {local_path}")
    print("This may take a while for large datasets...")
    print()
    
    artifact_utils.download_artifact_from_aml_uri(
        uri=datastore_uri,
        destination=str(local_path),
        datastore_operation=ml_client.datastores,
    )
    
    print(f"Download complete: {local_path}")


def is_folder_non_empty(path: Path) -> bool:
    """
    Check if a folder exists and contains files.
    
    Args:
        path: Path to check
    
    Returns:
        False if path doesn't exist or is empty.
    
    Raises:
        ValueError: If path exists but is a file (not directory).
    """
    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"Path exists but is a file, not a directory: {path}")
    # Check if there's at least one item
    return any(path.iterdir())
