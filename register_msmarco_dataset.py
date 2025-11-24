from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.get("msmarco_passage_localcache", version="2")

artifact_utils.download_artifact_from_aml_uri(
    uri=data_asset.path,
    destination="./data/msmarco",
    datastore_operation=ml_client.datastores
)