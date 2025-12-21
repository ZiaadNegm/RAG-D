# Azure ML Data Management Scripts

This module provides a complete solution for uploading and downloading WARP datasets and indexes to/from Azure ML datastores.

## Overview

These scripts manage the lifecycle of retrieval datasets (BEIR, LoTTE) and their corresponding WARP indexes in Azure ML. They handle:

1. **Uploading** local data/indexes to Azure blob storage via a named datastore
2. **Registering** uploaded assets as versioned Azure ML Data assets
3. **Downloading** registered assets back to local disk

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Azure ML Workspace                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Datastore: msmarco_ds                           ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐││
│  │  │                    Azure Blob Storage Container                     │││
│  │  │                                                                     │││
│  │  │   beir/quora/data/v1/         beir/quora/index/v1/                 │││
│  │  │   ├── collection.jsonl        ├── centroids.pt                     │││
│  │  │   └── queries.jsonl           ├── buckets.pt                       │││
│  │  │                               └── ...                               │││
│  │  │                                                                     │││
│  │  │   beir/scifact/data/v1/       lotte/lifestyle/data/v1/             │││
│  │  │   └── ...                     └── ...                               │││
│  │  └─────────────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Data Asset Registry                             ││
│  │                                                                         ││
│  │   beir-quora-data (v1)  ──────►  azureml://datastores/msmarco_ds/...   ││
│  │   beir-quora-index (v1) ──────►  azureml://datastores/msmarco_ds/...   ││
│  │   beir-scifact-data (v1) ─────►  azureml://datastores/msmarco_ds/...   ││
│  │   ...                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Upload / Download
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Local Disk                                      │
│                                                                              │
│   /mnt/datasets/                                                             │
│   ├── BEIR/                          ├── index/                              │
│   │   ├── quora/                     │   ├── BEIR.quora.XTR-base.warp/      │
│   │   │   ├── collection.jsonl       │   │   ├── centroids.pt               │
│   │   │   └── queries.jsonl          │   │   ├── buckets.pt                 │
│   │   ├── scifact/                   │   │   └── ...                        │
│   │   └── ...                        │   └── ...                            │
│   └── LOTTE/                         │                                       │
│       └── ...                        │                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables (`.env`)

All Azure ML and path configuration is loaded from the project root `.env` file:

```bash
# ═══════════════════════════════════════════════════════════════════════════
# AZURE ML CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

AZURE_SUBSCRIPTION_ID="431e1cff-c0bf-45fe-b9cb-5a69f7be7b5c"
# The Azure subscription ID containing your Azure ML workspace.
# Find this in Azure Portal → Subscriptions, or run:
#   az account show --query id -o tsv

AZURE_RESOURCE_GROUP="rag-d"
# The resource group containing your Azure ML workspace.
# This groups related Azure resources together.
# Find in Azure Portal → Resource Groups, or run:
#   az group list --query "[].name" -o tsv

AZURE_WORKSPACE_NAME="rag-d"
# The name of your Azure ML workspace.
# This is where Data assets, Datastores, and experiments live.
# Find in Azure Portal → Machine Learning, or run:
#   az ml workspace list --query "[].name" -o tsv

AZURE_DATASTORE_NAME="msmarco_ds"
# The datastore to use for uploads/downloads.
# A datastore is a named reference to an Azure Storage account/container.
# 
# IMPORTANT: We use a custom datastore instead of "workspaceblobstore" because:
#   - workspaceblobstore is the default but may have restricted permissions
#   - Custom datastores give you explicit control over the storage location
#
# The datastore maps to:
#   Storage Account: <your-storage-account>
#   Container: <your-container-name>
#
# List datastores:
#   az ml datastore list -w <workspace> -g <resource-group> --query "[].name"

# ═══════════════════════════════════════════════════════════════════════════
# LOCAL PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

INDEX_ROOT="/mnt/datasets/index"
# Base directory where WARP indexes are stored locally.
# Each dataset's index lives in a subdirectory named:
#   {COLLECTION}.{DATASET}.{MODEL}.warp
# Example: /mnt/datasets/index/BEIR.quora.XTR-base.warp/

BEIR_COLLECTION_PATH="/mnt/datasets/BEIR"
# Base directory for BEIR dataset collections.
# Each dataset is a subdirectory containing:
#   - collection.jsonl (documents)
#   - queries.jsonl (queries)
# Example: /mnt/datasets/BEIR/quora/

LOTTE_COLLECTION_PATH="/mnt/datasets/LOTTE"
# Base directory for LoTTE dataset collections.
# Structure mirrors BEIR.
# Example: /mnt/datasets/LOTTE/lifestyle/
```

### How Variables Map to Azure Resources

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Azure Resource Hierarchy                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AZURE_SUBSCRIPTION_ID                                                       │
│  └── Subscription: 431e1cff-c0bf-45fe-b9cb-5a69f7be7b5c                     │
│      │                                                                       │
│      └── AZURE_RESOURCE_GROUP                                                │
│          └── Resource Group: rag-d                                           │
│              │                                                               │
│              ├── AZURE_WORKSPACE_NAME                                        │
│              │   └── ML Workspace: rag-d                                     │
│              │       │                                                       │
│              │       ├── Data Assets (registered datasets/indexes)           │
│              │       │   ├── beir-quora-data (v1, v2, ...)                  │
│              │       │   ├── beir-quora-index (v1, v2, ...)                 │
│              │       │   └── ...                                             │
│              │       │                                                       │
│              │       └── AZURE_DATASTORE_NAME                                │
│              │           └── Datastore: msmarco_ds                           │
│              │               │                                               │
│              │               └── References Storage Account                  │
│              │                                                               │
│              └── Storage Account                                             │
│                  └── Blob Container                                          │
│                      └── beir/quora/data/v1/...  (actual files)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
scripts/azure/
├── __init__.py              # Package marker
├── config.py                # Configuration classes and .env loading
├── datasets.py              # Dataset registry (BEIR, LoTTE definitions)
├── storage.py               # Blob upload/download operations
├── ml_client.py             # Azure ML client singleton
├── cli.py                   # Shared CLI argument parsing
├── upload_to_datastore.py   # Main upload script
├── download_from_datastore.py  # Main download script
└── README.md                # This file
```

### Module Details

#### `config.py` - Configuration Layer

Defines the core data structures that drive everything:

```python
@dataclass
class AzureMLConfig:
    """Azure ML workspace connection settings (from .env)"""
    subscription_id: str      # AZURE_SUBSCRIPTION_ID
    resource_group: str       # AZURE_RESOURCE_GROUP
    workspace_name: str       # AZURE_WORKSPACE_NAME
    datastore_name: str       # AZURE_DATASTORE_NAME

@dataclass
class AssetConfig:
    """A single uploadable/downloadable asset (data OR index)"""
    name: str                 # Azure ML asset name: "beir-quora-data"
    local_path: str           # Local disk path: "/mnt/datasets/BEIR/quora"
    remote_prefix: str        # Blob path prefix: "beir/quora/data"
    
    def remote_path(self, version: str) -> str:
        """Full blob path: 'beir/quora/data/v1'"""
        return f"{self.remote_prefix}/v{version}"
    
    def datastore_uri(self, datastore: str, version: str) -> str:
        """Azure ML datastore URI for asset registration:
           'azureml://datastores/msmarco_ds/paths/beir/quora/data/v1'"""
        return f"azureml://datastores/{datastore}/paths/{self.remote_path(version)}"

@dataclass
class DatasetConfig:
    """A complete dataset with both data and index assets"""
    collection: str           # "BEIR" or "LoTTE"
    name: str                 # "quora", "scifact", etc.
    data: AssetConfig         # The collection/queries asset
    index: AssetConfig        # The WARP index asset
```

#### `datasets.py` - Dataset Registry

Central registry of all supported datasets. Each entry maps a key to its configuration:

```python
DATASETS = {
    "beir_quora": DatasetConfig(
        collection="BEIR",
        name="quora",
        data=AssetConfig(
            name="beir-quora-data",
            local_path="/mnt/datasets/BEIR/quora",        # From BEIR_COLLECTION_PATH
            remote_prefix="beir/quora/data"
        ),
        index=AssetConfig(
            name="beir-quora-index", 
            local_path="/mnt/datasets/index/BEIR.quora.XTR-base.warp",  # From INDEX_ROOT
            remote_prefix="beir/quora/index"
        )
    ),
    # ... more datasets
}
```

**Adding a new dataset:**
```python
"beir_newdataset": DatasetConfig(
    collection="BEIR",
    name="newdataset",
    data=AssetConfig("beir-newdataset-data", f"{beir_path}/newdataset", "beir/newdataset/data"),
    index=AssetConfig("beir-newdataset-index", f"{index_root}/BEIR.newdataset.XTR-base.warp", "beir/newdataset/index")
)
```

#### `storage.py` - Blob Operations

Handles the actual file transfer:

**Upload Flow:**
```python
def upload_folder_to_datastore(ml_client, datastore_name, local_path, remote_path):
    """
    Uses Azure CLI (not SDK) to upload directly to the datastore.
    
    Why CLI instead of SDK?
    - SDK's MLClient.data.create_or_update() auto-uploads to workspaceblobstore
    - We need to upload to a CUSTOM datastore (msmarco_ds)
    - CLI gives explicit control: az storage blob upload-batch --destination <container>
    
    Steps:
    1. Get datastore details (storage account, container, credentials)
    2. Build az storage blob upload-batch command
    3. Execute and stream output
    """
```

**Download Flow:**
```python
def download_from_datastore(ml_client, datastore_uri, local_path):
    """
    Uses Azure ML's artifact utilities to download from datastore URI.
    
    Steps:
    1. Parse the azureml:// URI to extract datastore and path
    2. Use artifact download utilities to fetch files
    3. Handle directory structure preservation
    """
```

#### `ml_client.py` - Client Management

Provides a singleton MLClient instance:

```python
def get_ml_client(config: AzureMLConfig) -> MLClient:
    """
    Creates and caches an MLClient for the Azure ML workspace.
    
    Authentication uses DefaultAzureCredential which tries (in order):
    1. Environment variables (AZURE_CLIENT_ID, etc.)
    2. Managed Identity (if running in Azure)
    3. Azure CLI credentials (az login)
    4. VS Code credentials
    5. Interactive browser login
    """
```

#### `cli.py` - Argument Parsing

Shared CLI utilities for consistent argument handling:

```python
def add_common_arguments(parser):
    """Adds --dataset, --kind, --version, --force, --list to any parser"""

def validate_common_arguments(args):
    """Validates that required arguments are present"""
```

---

## Usage

### List Available Datasets

```bash
# Show all registered datasets
python scripts/azure/upload_to_datastore.py --list
python scripts/azure/download_from_datastore.py --list
```

Output:
```
Available datasets:
  beir_quora       BEIR/quora
  beir_scifact     BEIR/scifact
  beir_nfcorpus    BEIR/nfcorpus
  beir_fiqa        BEIR/fiqa
  beir_arguana     BEIR/arguana
  beir_scidocs     BEIR/scidocs
  lotte_lifestyle  LoTTE/lifestyle
  lotte_recreation LoTTE/recreation
  lotte_technology LoTTE/technology
  lotte_writing    LoTTE/writing
  lotte_science    LoTTE/science
  lotte_pooled     LoTTE/pooled
```

### Upload Data

```bash
# Upload BEIR quora collection (queries + documents)
python scripts/azure/upload_to_datastore.py -d beir_quora -k data

# Upload BEIR quora WARP index
python scripts/azure/upload_to_datastore.py -d beir_quora -k index

# Upload with specific version
python scripts/azure/upload_to_datastore.py -d beir_quora -k data -v 2

# Force re-upload (overwrite existing)
python scripts/azure/upload_to_datastore.py -d beir_quora -k data --force

# Register existing blob data without re-uploading
python scripts/azure/upload_to_datastore.py -d beir_quora -k data --skip-upload
```

### Download Data

```bash
# Download BEIR quora collection
python scripts/azure/download_from_datastore.py -d beir_quora -k data

# Download specific version
python scripts/azure/download_from_datastore.py -d beir_quora -k data -v 2

# Force re-download (overwrite local)
python scripts/azure/download_from_datastore.py -d beir_quora -k data --force
```

---

## Data Flow

### Upload Process

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   Local Disk    │     │   Azure Blob     │     │   Azure ML Registry     │
│                 │     │   Storage        │     │                         │
│ /mnt/datasets/  │     │                  │     │                         │
│ BEIR/quora/     │────►│ beir/quora/      │────►│ Data Asset:             │
│ ├── collection  │     │ data/v1/         │     │ beir-quora-data (v1)    │
│ └── queries     │     │ ├── collection   │     │                         │
│                 │     │ └── queries      │     │ path: azureml://        │
└─────────────────┘     └──────────────────┘     │ datastores/msmarco_ds/  │
                                                 │ paths/beir/quora/data/v1│
                              Step 1:            └─────────────────────────┘
                        az storage blob                    │
                        upload-batch                       │
                                                          Step 2:
                                                    ml_client.data.
                                                    create_or_update()
```

**Step-by-step:**

1. **Lookup Configuration**
   ```python
   config = get_dataset_config("beir_quora")
   asset = get_asset_config("beir_quora", "data")
   # asset.local_path = "/mnt/datasets/BEIR/quora"
   # asset.remote_prefix = "beir/quora/data"
   ```

2. **Upload to Blob Storage**
   ```bash
   az storage blob upload-batch \
     --account-name <storage-account> \
     --destination <container>/beir/quora/data/v1 \
     --source /mnt/datasets/BEIR/quora \
     --auth-mode login
   ```

3. **Register Data Asset**
   ```python
   data_asset = Data(
       name="beir-quora-data",
       version="1",
       type=AssetTypes.URI_FOLDER,
       path="azureml://datastores/msmarco_ds/paths/beir/quora/data/v1"
   )
   ml_client.data.create_or_update(data_asset)
   ```

### Download Process

```
┌─────────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Azure ML Registry     │     │   Azure Blob     │     │   Local Disk    │
│                         │     │   Storage        │     │                 │
│ Data Asset:             │     │                  │     │ /mnt/datasets/  │
│ beir-quora-data (v1)    │────►│ beir/quora/      │────►│ BEIR/quora/     │
│                         │     │ data/v1/         │     │ ├── collection  │
│ path: azureml://...     │     │ ├── collection   │     │ └── queries     │
└─────────────────────────┘     │ └── queries      │     │                 │
                                └──────────────────┘     └─────────────────┘
        Step 1:                                                  │
  ml_client.data.get()                                          │
                                        Step 2:                  │
                                  artifact_utils.download()      │
```

---

## Common Issues & Solutions

### Permission Denied on workspaceblobstore

**Problem:** SDK auto-uploads to workspaceblobstore, but you don't have write permissions.

**Solution:** These scripts use `az storage blob upload-batch` to explicitly upload to your custom datastore (`msmarco_ds`), bypassing the SDK's default behavior.

### Authentication Errors

**Problem:** "DefaultAzureCredential failed to retrieve a token"

**Solution:**
```bash
# Login to Azure CLI
az login

# Verify you're in the right subscription
az account show

# If needed, set the correct subscription
az account set --subscription $AZURE_SUBSCRIPTION_ID
```

### Asset Already Exists

**Problem:** "Data asset already exists" error when uploading.

**Solution:** Use `--force` to overwrite, or increment the version with `-v 2`.

### Local Path Not Found

**Problem:** "Local path does not exist" error.

**Solution:** 
1. Ensure the data is downloaded: `./scripts/setup_data.sh`
2. Or for indexes: `./scripts/index_data.sh`
3. Check paths in `.env` match your actual directory structure.

---

## Azure Concepts Reference

| Concept | Description | Example |
|---------|-------------|---------|
| **Subscription** | Billing/access boundary for Azure resources | `431e1cff-c0bf-45fe-b9cb-5a69f7be7b5c` |
| **Resource Group** | Logical container for related resources | `rag-d` |
| **ML Workspace** | Central hub for ML assets, experiments, models | `rag-d` |
| **Datastore** | Named reference to a storage location | `msmarco_ds` → Storage Account/Container |
| **Data Asset** | Versioned, registered dataset in ML Workspace | `beir-quora-data` v1 |
| **Datastore URI** | Addressable path to data in a datastore | `azureml://datastores/msmarco_ds/paths/beir/quora/data/v1` |

---

## Extending the Scripts

### Adding a New Dataset

1. Edit `datasets.py`:
   ```python
   "beir_newdataset": DatasetConfig(
       collection="BEIR",
       name="newdataset",
       data=AssetConfig(
           name="beir-newdataset-data",
           local_path=f"{beir_path}/newdataset",
           remote_prefix="beir/newdataset/data"
       ),
       index=AssetConfig(
           name="beir-newdataset-index",
           local_path=f"{index_root}/BEIR.newdataset.XTR-base.warp",
           remote_prefix="beir/newdataset/index"
       )
   ),
   ```

2. Download the data locally to the expected path.

3. Upload:
   ```bash
   python scripts/azure/upload_to_datastore.py -d beir_newdataset -k data
   python scripts/azure/upload_to_datastore.py -d beir_newdataset -k index
   ```

### Adding a New Asset Type

If you need more than `data` and `index` (e.g., `embeddings`):

1. Add the new kind to `AssetKind` enum in `config.py`
2. Add the asset to `DatasetConfig` 
3. Update `get_asset_config()` in `datasets.py`

---

## Quick Reference

```bash
# Activate environment
conda activate warp

# List datasets
python scripts/azure/upload_to_datastore.py --list

# Upload workflow
python scripts/azure/upload_to_datastore.py -d beir_quora -k data    # Upload collection
python scripts/azure/upload_to_datastore.py -d beir_quora -k index   # Upload index

# Download workflow  
python scripts/azure/download_from_datastore.py -d beir_quora -k data
python scripts/azure/download_from_datastore.py -d beir_quora -k index

# Check Azure CLI auth
az account show
az ml datastore list -w rag-d -g rag-d --query "[].name" -o tsv
```
