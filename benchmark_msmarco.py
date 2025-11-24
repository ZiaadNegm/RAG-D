"""
Benchmark ColBERTv2 on MS MARCO Passage Ranking
Reproduces paper results: MRR@10, Recall@100, Recall@1000
"""
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import os

if __name__ == '__main__':
    # Paths
    queries_path = "data/msmarco/dev/queries.tsv"
    collection_path = "data/msmarco/collection.tsv"
    qrels_path = "data/msmarco/dev/qrels"
    
    # Output
    output_ranking = "results/msmarco_colbertv2_k1000.ranking.tsv"
    os.makedirs("results", exist_ok=True)
    
    print("=" * 80)
    print("Benchmarking ColBERTv2 on MS MARCO Passage Ranking")
    print("=" * 80)
    print(f"Queries: {queries_path}")
    print(f"Collection: {collection_path}")
    print(f"Index: experiments/msmarco/indexes/msmarco_passage.nbits2")
    print(f"K (retrieval depth): 1000")
    print("=" * 80)
    
    # Setup searcher
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root=".",
        )
        
        print("\n[1/3] Loading searcher...")
        searcher = Searcher(
            index="msmarco_passage.nbits2",
            collection=collection_path,
            config=config
        )
        
        print("[2/3] Loading queries and running search (this will take time)...")
        queries = Queries(queries_path)
        print(f"      Total queries: {len(queries)}")
        
        # Search with k=1000
        ranking = searcher.search_all(queries, k=1000)
        
        print(f"[3/3] Saving ranking to {output_ranking}...")
        ranking.save(output_ranking)
    
    print("\n" + "=" * 80)
    print("Running evaluation...")
    print("=" * 80)
    
    # Run evaluation
    import subprocess
    result = subprocess.run([
        "python", "-m", "utility.evaluate.msmarco_passages",
        "--ranking", output_ranking,
        "--qrels", qrels_path
    ])
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print(f"Ranking saved to: {output_ranking}")
    print("=" * 80)
