"""
Azure ML client utilities.

Provides a singleton MLClient instance for reuse across scripts.
"""

from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from .config import AzureMLConfig


_ml_client: Optional[MLClient] = None


def get_ml_client(config: Optional[AzureMLConfig] = None) -> MLClient:
    """
    Get or create an MLClient instance.
    
    Args:
        config: Azure ML configuration. If None, loads from environment.
        
    Returns:
        Configured MLClient instance.
    """
    global _ml_client
    
    if _ml_client is not None:
        return _ml_client
    
    if config is None:
        config = AzureMLConfig.from_env()
    
    print(f"Connecting to Azure ML workspace...")
    print(f"  Resource Group: {config.resource_group}")
    print(f"  Workspace: {config.workspace_name}")
    
    credential = DefaultAzureCredential()
    _ml_client = MLClient(
        credential=credential,
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name,
    )
    
    print("  Connected successfully!\n")
    return _ml_client


def reset_client() -> None:
    """Reset the cached MLClient (useful for testing)."""
    global _ml_client
    _ml_client = None
