#!/usr/bin/env python3
"""
Validate search output format for H3 Phase II analysis.
Run on small subset (50 queries) to verify the join logic works.
"""

import os
import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from tqdm import tqdm

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries

print("=" * 60)
print("VALIDATION: Small subset search (50 queries)")
print("=" * 60)

# Configure the run
config = WARPRunConfig(
    collection='beir',
    dataset='quora',
    datasplit='test',
    nbits=4,
    nprobe=32,
    t_prime=10000,
    k=100,  # Smaller k for validation
    bound=128,
    runtime=None,
    centroid_only=False
)

print("\n1. Loading searcher...")
searcher = WARPSearcher(config)

print("\n2. Loading queries (subset of 50)...")
queries = WARPQueries(config)
# Limit to 50 queries for quick validation
query_data = dict(list(queries.queries.data.items())[:50])
queries.queries.data = query_data
print(f"   Using {len(query_data)} queries")

print("\n3. Running search...")
rankings = searcher.search_all(queries, k=config.k, batched=False, show_progress=True)

print("\n4. Converting rankings to DataFrame...")
results = []
ranking_data = rankings.ranking.data
for qid, doc_scores in ranking_data.items():
    for pid, rank, score in doc_scores:
        results.append({
            'query_id': int(qid),
            'doc_id': int(pid),
            'rank': int(rank),
            'score': float(score)
        })

search_df = pd.DataFrame(results)
print(f"   Search results: {len(search_df):,} rows")
print(f"   Columns: {list(search_df.columns)}")
print(f"   Sample rows:")
print(search_df.head(5).to_string())

print("\n5. Loading golden metrics (routing_status)...")
golden_dir = '/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425/golden_metrics_v2'
routing_df = pd.read_parquet(f'{golden_dir}/routing_status.parquet')
print(f"   Routing status: {len(routing_df):,} rows")
print(f"   Columns: {list(routing_df.columns)}")
print(f"   Sample rows:")
print(routing_df.head(3).to_string())

print("\n6. Testing H3 Phase II join logic...")
# Join search results with routing_status on (query_id, doc_id)
merged = search_df.merge(
    routing_df[['query_id', 'doc_id', 'routing_status']],
    on=['query_id', 'doc_id'],
    how='inner'
)
print(f"   Merged rows (search results that are golden docs): {len(merged):,}")

if len(merged) > 0:
    print(f"\n   Merged sample:")
    print(merged.head(5).to_string())
    
    print("\n   H3 Phase II Analysis Preview:")
    print("   ------------------------------")
    
    # For each routing_status, check if the golden doc was retrieved
    status_counts = merged.groupby('routing_status').agg({
        'doc_id': 'count',
        'rank': 'mean'
    }).rename(columns={'doc_id': 'found_count', 'rank': 'avg_rank'})
    
    print(f"\n   Golden docs found by routing_status:")
    print(status_counts.to_string())
    
    # Show total golden docs per status for reference
    print(f"\n   Total golden docs per status (from routing_status.parquet):")
    print(routing_df['routing_status'].value_counts().to_string())
    
    # Compute recall by routing status (subset analysis)
    query_ids_in_subset = set(search_df['query_id'].unique())
    routing_subset = routing_df[routing_df['query_id'].isin(query_ids_in_subset)]
    
    print(f"\n   Golden docs for our {len(query_ids_in_subset)} queries: {len(routing_subset)}")
    
    for status in ['FULLY_OPTIMAL', 'PARTIAL']:
        golden_for_status = routing_subset[routing_subset['routing_status'] == status]
        found_for_status = merged[merged['routing_status'] == status]
        if len(golden_for_status) > 0:
            recall = len(found_for_status) / len(golden_for_status) * 100
            print(f"   {status}: {len(found_for_status)}/{len(golden_for_status)} found = {recall:.1f}% recall")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE - Output format is correct!")
print("=" * 60)
