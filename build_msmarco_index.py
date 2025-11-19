from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from pathlib import Path

def main():
    # Project root is one level above the ColBERT repo
    project_root = Path(__file__).resolve().parent.parent

    collection = project_root / "data" / "collection.tsv"
    root = project_root / "ColBERT" / "experiments"  # experiment root
    experiment = "msmarco_passage"
    checkpoint = "colbert-ir/colbertv2.0"            # HF ID OR local path

    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig(
            root=str(root),
            nbits=2,               # PLAID compression
            doc_maxlen=180,        # standard MS MARCO setting
        )

        indexer = Indexer(checkpoint=checkpoint, config=config)

        indexer.index(
            name="msmarco_passage.nbits2",  # index name
            collection=str(collection),
        )

if __name__ == "__main__":
    main()