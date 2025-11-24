# download_msmarco_index.py

from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils

# --- Config: adjust if you ever move to another subscription/RG/workspace --- #
SUBSCRIPTION_ID = "431e1cff-c0bf-45fe-b9cb-5a69f7be7b5c"
RESOURCE_GROUP = "rag-d"
WORKSPACE_NAME = "rag-d"

DATA_NAME = "msmarco_passage_index_nbits2"
DATA_VERSION = "1"

LOCAL_DEST = Path("data/msmarco_index/msmarco_passage.nbits2")  # local folder

def main():
    print(">> Setting up MLClient...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    print(f">> Fetching data asset: {DATA_NAME}:{DATA_VERSION}")
    data_asset = ml_client.data.get(name=DATA_NAME, version=DATA_VERSION)
    print("   Asset path (remote):", data_asset.path)

    dest = LOCAL_DEST
    dest.mkdir(parents=True, exist_ok=True)

    print(f">> Downloading to: {dest}")
    artifact_utils.download_artifact_from_aml_uri(
        uri=data_asset.path,
        destination=str(dest),
        datastore_operation=ml_client.datastores,
    )

    print("✅ Download complete.")

if __name__ == "__main__":
    main()

