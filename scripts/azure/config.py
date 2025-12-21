"""
Configuration for Azure ML dataset upload/download.

Loads settings from .env and defines core configuration classes.
"""

import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from repo root
REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


# =============================================================================
# Azure ML Configuration
# =============================================================================

@dataclass
class AzureMLConfig:
    """Azure ML workspace configuration."""
    subscription_id: str
    resource_group: str
    workspace_name: str
    datastore_name: str

    @classmethod
    def from_env(cls) -> "AzureMLConfig":
        """Load Azure ML config from environment variables."""
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        datastore_name = os.getenv("AZURE_DATASTORE_NAME", "msmarco_ds")

        missing = []
        if not subscription_id:
            missing.append("AZURE_SUBSCRIPTION_ID")
        if not resource_group:
            missing.append("AZURE_RESOURCE_GROUP")
        if not workspace_name:
            missing.append("AZURE_WORKSPACE_NAME")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set them in your .env file."
            )

        return cls(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            datastore_name=datastore_name,
        )


# =============================================================================
# Local Path Configuration (from .env)
# =============================================================================

BEIR_BASE_PATH = Path(os.getenv("BEIR_COLLECTION_PATH", "/mnt/datasets/BEIR"))
INDEX_ROOT = Path(os.getenv("INDEX_ROOT", "/mnt/datasets/index"))
LOTTE_BASE_PATH = Path(os.getenv("LOTTE_COLLECTION_PATH", "/mnt/datasets/LOTTE"))


# =============================================================================
# Asset Kind Enum
# =============================================================================

class AssetKind(str, Enum):
    """Type of asset (data or index)."""
    DATA = "data"
    INDEX = "index"


# =============================================================================
# Asset Configuration
# =============================================================================

@dataclass
class AssetConfig:
    """Configuration for a single asset (data or index)."""
    asset_name: str
    local_path: Path
    base_remote_dir: str  # Base path within the datastore (e.g., "beir/quora/data")
    description: str
    default_version: str = "1"

    def remote_path(self, version: str) -> str:
        """Get the versioned remote path within the datastore."""
        return f"{self.base_remote_dir}/v{version}"

    def datastore_uri(self, datastore_name: str, version: str) -> str:
        """Get the full Azure ML datastore URI for this asset."""
        return f"azureml://datastores/{datastore_name}/paths/{self.remote_path(version)}"


@dataclass
class DatasetConfig:
    """Configuration for a dataset with its data and index assets."""
    data: AssetConfig
    index: AssetConfig
