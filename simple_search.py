"""
Simple ColBERT search script for testing queries
"""
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__ == '__main__':
    # Simple queries to test
    queries = [
        "what is pascal's law in simple terms",
        "what is patricia cornwell's latest book"
    ]
    
    # Setup searcher with proper experiments folder structure
    # The index is now linked at: experiments/msmarco/indexes/msmarco_passage.nbits2
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root=".",  # root is current directory, which contains experiments/
        )
        searcher = Searcher(
            index="msmarco_passage.nbits2",
            collection="data/msmarco/collection.tsv",
            config=config
        )
    
    # Search for each query
    print("=" * 80)
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        # Get top 3 results
        results = searcher.search(query, k=3)
        
        # Display results
        for passage_id, passage_rank, passage_score in zip(*results):
            print(f"[Rank {passage_rank}] Score: {passage_score:.2f}")
            print(f"Passage ID: {passage_id}")
            print(f"Text: {searcher.collection[passage_id][:200]}...")
            print()
    
    print("=" * 80)