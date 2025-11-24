from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from pathlib import Path


def main():
    collection = Path("/mnt/RAG-D/data/msmarco/collection.tsv")
    index_root = Path("/mnt/RAG-D/data/msmarco_index")     # where final index goes
    index_name = "msmarco_passage.nbits2"

    checkpoint = "colbert-ir/colbertv2.0"   # HF checkpoint (works with ColBERT main)

    run_cfg = RunConfig(
        nranks=1,
        experiment="msmarco_passage",
    )

    colbert_cfg = ColBERTConfig(
        root=str(index_root),   # ColBERT writes the index under here
        nbits=2,
        doc_maxlen=180
    )

    #
    # --- INDEX ---
    #
    with Run().context(run_cfg):
        indexer = Indexer(
            checkpoint=checkpoint,
            config=colbert_cfg
        )

        indexer.index(
            name=index_name,
            collection=str(collection)
        )


if __name__ == "__main__":
    main()
