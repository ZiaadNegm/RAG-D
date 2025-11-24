from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

subscription_id = "431e1cff-c0bf-45fe-b9cb-5a69f7be7b5c"
resource_group = "rag-d"
workspace_name = "rag-d"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

data_asset = Data(
    name="msmarco_passage_index_nbits2",
    version="1",
    description="PLAID index for MS MARCO Passage nbits2",
    path="azureml://datastores/msmarco_ds/paths/indexes/msmarco_passage.nbits2",
    type=AssetTypes.URI_FOLDER,
)

created = ml_client.data.create_or_update(data_asset)
print("Created data asset:", created.name, created.version, created.path)

