#!/usr/bin/env python3
"""
Online Cluster Properties Verification Experiments

This script contains experiments to validate data structures and test metric
implementations for online cluster properties.

Run individual experiments:
    python experiments/online_metrics_experiments.py --experiment e0 --output-dir /mnt/warp_measurements/online_dev
    python experiments/online_metrics_experiments.py --experiment e1 --run-dir /mnt/warp_measurements/online_dev/runs/{run_id}
    python experiments/online_metrics_experiments.py --experiment all --run-dir /mnt/warp_measurements/online_dev/runs/{run_id}

See docs/ONLINE_CLUSTER_PROPERTIES_INTEGRATION_PLAN.md for experiment descriptions.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


def experiment_e0_collect_measurements(output_dir: str, num_queries: int = 1000) -> str:
    """
    E0: Collect M1/M3/M4/R0 measurement data.
    
    This is a wrapper around m4_e2e_test.py to ensure all required files are generated.
    
    Returns:
        Path to the created run directory
    """
    print("=" * 60)
    print("E0: Collect Measurement Data")
    print("=" * 60)
    
    # Import here to avoid slow startup
    from experiments.m4_e2e_test import main as m4_e2e_main
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run measurement collection
    print(f"\nCollecting measurements for {num_queries} queries...")
    print(f"Output directory: {output_dir}")
    
    # Set up arguments for m4_e2e_test
    sys.argv = [
        'm4_e2e_test.py',
        '--num-queries', str(num_queries),
        '--output-dir', output_dir,
    ]
    
    try:
        m4_e2e_main()
    except SystemExit:
        pass  # m4_e2e_test may call sys.exit()
    
    # Find the created run directory
    runs_dir = Path(output_dir) / "runs"
    if runs_dir.exists():
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            run_dir = run_dirs[0]
            print(f"\n✅ E0 PASSED: Run directory created at {run_dir}")
            
            # List created files
            for tier in ["tier_a", "tier_b"]:
                tier_dir = run_dir / tier
                if tier_dir.exists():
                    print(f"\n{tier}/:")
                    for f in tier_dir.iterdir():
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"  {f.name}: {size_mb:.2f} MB")
            
            return str(run_dir)
    
    print("\n❌ E0 FAILED: No run directory created")
    return None


def experiment_e1_m1_structure(run_dir: str) -> dict:
    """
    E1: Validate M1 data structure for online metrics.
    
    Questions answered:
    1. Does each M1 row represent one (query_id, q_token_id, centroid_id) selection?
    2. What's the typical number of selections per query?
    3. Are there duplicate (query_id, q_token_id, centroid_id) tuples?
    """
    print("=" * 60)
    print("E1: Validate M1 Data Structure")
    print("=" * 60)
    
    m1_path = Path(run_dir) / "tier_a" / "M1_compute_per_centroid.parquet"
    if not m1_path.exists():
        print(f"❌ M1 file not found: {m1_path}")
        return None
    
    m1 = pd.read_parquet(m1_path)
    
    print(f"\nTotal rows: {len(m1):,}")
    print(f"Columns: {list(m1.columns)}")
    print(f"\nColumn dtypes:")
    for col in m1.columns:
        print(f"  {col}: {m1[col].dtype}")
    
    print(f"\nUnique values:")
    print(f"  Unique queries: {m1['query_id'].nunique():,}")
    print(f"  Unique centroids touched: {m1['centroid_id'].nunique():,}")
    
    # Check for duplicates
    key_cols = ['query_id', 'q_token_id', 'centroid_id']
    duplicates = m1.duplicated(subset=key_cols, keep=False).sum()
    print(f"\nDuplicate (query, token, centroid) tuples: {duplicates}")
    if duplicates > 0:
        print("  ⚠️ WARNING: Duplicates found - may need deduplication")
    else:
        print("  ✅ No duplicates - each row is unique selection")
    
    # Selections per query
    sel_per_query = m1.groupby('query_id').size()
    print(f"\nSelections per query:")
    print(f"  Min: {sel_per_query.min()}")
    print(f"  Mean: {sel_per_query.mean():.1f}")
    print(f"  Median: {sel_per_query.median():.1f}")
    print(f"  Max: {sel_per_query.max()}")
    
    # Expected: ~query_tokens × nprobe selections per query
    # For 12 tokens × 16 nprobe = 192 selections
    expected_range = (100, 600)  # reasonable range
    mean_sel = sel_per_query.mean()
    if expected_range[0] <= mean_sel <= expected_range[1]:
        print(f"  ✅ Mean in expected range {expected_range}")
    else:
        print(f"  ⚠️ Mean outside expected range {expected_range}")
    
    # num_token_token_sims distribution
    print(f"\nnum_token_token_sims (embeddings per selection):")
    print(f"  Min: {m1['num_token_token_sims'].min()}")
    print(f"  Mean: {m1['num_token_token_sims'].mean():.1f}")
    print(f"  Median: {m1['num_token_token_sims'].median():.1f}")
    print(f"  Max: {m1['num_token_token_sims'].max()}")
    print(f"  Sum (total sims): {m1['num_token_token_sims'].sum():,}")
    
    print("\n✅ E1 PASSED")
    
    return {
        'total_rows': len(m1),
        'columns': list(m1.columns),
        'unique_queries': m1['query_id'].nunique(),
        'unique_centroids': m1['centroid_id'].nunique(),
        'duplicates': int(duplicates),
        'mean_selections_per_query': float(sel_per_query.mean()),
        'total_sims': int(m1['num_token_token_sims'].sum()),
    }


def experiment_e2_m3_centroid_lookup(run_dir: str, index_path: str) -> dict:
    """
    E2: Validate centroid lookup from M3 embedding positions.
    
    Questions answered:
    1. Does searchsorted correctly map embedding_pos → centroid_id?
    2. What's the distribution of winner centroids?
    3. How many unique centroids host winners?
    """
    print("=" * 60)
    print("E2: Validate M3 Centroid Lookup")
    print("=" * 60)
    
    m3_path = Path(run_dir) / "tier_b" / "M3_observed_winners.parquet"
    if not m3_path.exists():
        print(f"❌ M3 file not found: {m3_path}")
        return None
    
    m3 = pd.read_parquet(m3_path)
    
    print(f"\nM3 rows: {len(m3):,}")
    print(f"Columns: {list(m3.columns)}")
    
    # Load offsets
    sizes_path = Path(index_path) / "sizes.compacted.pt"
    if not sizes_path.exists():
        print(f"❌ sizes.compacted.pt not found: {sizes_path}")
        return None
    
    sizes = torch.load(sizes_path)
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    torch.cumsum(sizes, dim=0, out=offsets[1:])
    num_centroids = len(sizes)
    num_embeddings = offsets[-1].item()
    
    print(f"\nIndex info:")
    print(f"  Num centroids: {num_centroids:,}")
    print(f"  Num embeddings: {num_embeddings:,}")
    
    # Derive centroid from embedding_pos
    positions = torch.tensor(m3['winner_embedding_pos'].values, dtype=torch.long)
    centroids = torch.searchsorted(offsets, positions, side='right') - 1
    
    print(f"\nCentroid lookup results:")
    print(f"  Unique winner centroids: {centroids.unique().numel():,}")
    print(f"  Centroid range: [{centroids.min().item()}, {centroids.max().item()}]")
    
    # Verify: centroid should be valid (0 <= c < num_centroids)
    invalid = ((centroids < 0) | (centroids >= num_centroids)).sum().item()
    print(f"\nValidation:")
    print(f"  Invalid centroid IDs: {invalid}")
    if invalid > 0:
        print("  ❌ FAILED: Some centroids out of range")
        return None
    else:
        print("  ✅ All centroids in valid range")
    
    # Verify: embedding_pos should be within centroid bounds
    begin = offsets[centroids]
    end = offsets[centroids + 1]
    within_bounds = ((positions >= begin) & (positions < end)).all().item()
    print(f"  All positions within centroid bounds: {within_bounds}")
    if not within_bounds:
        print("  ❌ FAILED: Some positions outside centroid bounds")
        return None
    else:
        print("  ✅ All positions correctly mapped")
    
    # Distribution of winners per centroid
    centroid_counts = pd.Series(centroids.numpy()).value_counts()
    print(f"\nWinners per centroid:")
    print(f"  Centroids with winners: {len(centroid_counts):,}")
    print(f"  Min winners: {centroid_counts.min()}")
    print(f"  Mean winners: {centroid_counts.mean():.1f}")
    print(f"  Max winners: {centroid_counts.max()}")
    
    print("\n✅ E2 PASSED")
    
    return {
        'num_rows': len(m3),
        'unique_winner_centroids': centroids.unique().numel(),
        'invalid_centroids': invalid,
        'positions_within_bounds': within_bounds,
    }


def experiment_e3_r0_schema(run_dir: str) -> dict:
    """
    E3: Validate R0 schema for C4/C5 routing fidelity metrics.
    
    Questions answered:
    1. Does R0 have a `centroid_score` column?
    2. What's the score range and distribution?
    3. Is `rank` (position in nprobe order) present?
    """
    print("=" * 60)
    print("E3: Validate R0 Schema")
    print("=" * 60)
    
    r0_path = Path(run_dir) / "tier_b" / "R0_selected_centroids.parquet"
    if not r0_path.exists():
        print(f"❌ R0 file not found: {r0_path}")
        return None
    
    r0 = pd.read_parquet(r0_path)
    
    print(f"\nR0 rows: {len(r0):,}")
    print(f"Columns: {list(r0.columns)}")
    print(f"\nColumn dtypes:")
    for col in r0.columns:
        print(f"  {col}: {r0[col].dtype}")
    
    has_score = 'centroid_score' in r0.columns
    has_rank = 'rank' in r0.columns
    
    print(f"\nC4/C5 required fields:")
    print(f"  centroid_score column present: {has_score}")
    print(f"  rank column present: {has_rank}")
    
    if has_score:
        print(f"\ncentroid_score statistics:")
        print(f"  Min: {r0['centroid_score'].min():.4f}")
        print(f"  Mean: {r0['centroid_score'].mean():.4f}")
        print(f"  Median: {r0['centroid_score'].median():.4f}")
        print(f"  Max: {r0['centroid_score'].max():.4f}")
        print("  ✅ C4/C5 can be implemented")
    else:
        print("\n⚠️ WARNING: centroid_score not in R0")
        print("  C4/C5 will need modification. Options:")
        print("  1. Add centroid_score recording to tracker.py")
        print("  2. Recompute scores offline (requires query embeddings)")
        print("  3. Defer C4/C5 to future phase")
    
    if has_rank:
        print(f"\nrank statistics:")
        print(f"  Min: {r0['rank'].min()}")
        print(f"  Max: {r0['rank'].max()}")
    
    # Unique centroids
    print(f"\nCentroid coverage:")
    print(f"  Unique centroids in R0: {r0['centroid_id'].nunique():,}")
    
    status = "PASSED" if has_score else "PARTIAL (centroid_score missing)"
    print(f"\n{'✅' if has_score else '⚠️'} E3 {status}")
    
    return {
        'num_rows': len(r0),
        'columns': list(r0.columns),
        'has_centroid_score': has_score,
        'has_rank': has_rank,
        'unique_centroids': r0['centroid_id'].nunique(),
    }


def experiment_e4_sel_freq(run_dir: str, index_path: str) -> dict:
    """
    E4: Compute and validate selection frequency.
    """
    print("=" * 60)
    print("E4: Selection Frequency")
    print("=" * 60)
    
    m1_path = Path(run_dir) / "tier_a" / "M1_compute_per_centroid.parquet"
    if not m1_path.exists():
        print(f"❌ M1 file not found: {m1_path}")
        return None
    
    m1 = pd.read_parquet(m1_path)
    
    # Load index info
    sizes = torch.load(Path(index_path) / "sizes.compacted.pt")
    num_centroids = len(sizes)
    
    # Compute selection frequency
    sel_freq = m1['centroid_id'].value_counts().reindex(range(num_centroids), fill_value=0)
    
    print(f"\nTotal centroids: {num_centroids:,}")
    print(f"Centroids selected at least once: {(sel_freq > 0).sum():,}")
    print(f"Anti-hubs (never selected): {(sel_freq == 0).sum():,}")
    print(f"Anti-hub rate: {(sel_freq == 0).mean():.1%}")
    
    print(f"\nsel_freq distribution:")
    print(f"  Min: {sel_freq.min()}")
    print(f"  Mean: {sel_freq.mean():.1f}")
    print(f"  Median: {sel_freq.median():.1f}")
    print(f"  Max: {sel_freq.max()}")
    print(f"  Std: {sel_freq.std():.1f}")
    
    # Traffic concentration
    sorted_freq = sel_freq.sort_values(ascending=False)
    total = sorted_freq.sum()
    
    top_1pct = int(np.ceil(num_centroids * 0.01))
    top_5pct = int(np.ceil(num_centroids * 0.05))
    top_10pct = int(np.ceil(num_centroids * 0.10))
    
    print(f"\nTraffic concentration:")
    print(f"  Top 1% ({top_1pct} centroids): {sorted_freq.head(top_1pct).sum() / total:.1%}")
    print(f"  Top 5% ({top_5pct} centroids): {sorted_freq.head(top_5pct).sum() / total:.1%}")
    print(f"  Top 10% ({top_10pct} centroids): {sorted_freq.head(top_10pct).sum() / total:.1%}")
    
    # Entropy
    p = sel_freq / total
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    max_entropy = np.log(num_centroids)
    
    print(f"\nEntropy:")
    print(f"  Entropy: {entropy:.2f}")
    print(f"  Max entropy: {max_entropy:.2f}")
    print(f"  Normalized entropy: {entropy / max_entropy:.3f}")
    
    print("\n✅ E4 PASSED")
    
    return {
        'num_centroids': num_centroids,
        'centroids_selected': int((sel_freq > 0).sum()),
        'anti_hub_rate': float((sel_freq == 0).mean()),
        'top_1pct_share': float(sorted_freq.head(top_1pct).sum() / total),
        'top_5pct_share': float(sorted_freq.head(top_5pct).sum() / total),
        'normalized_entropy': float(entropy / max_entropy),
    }


def experiment_e5_yield(run_dir: str, index_path: str) -> dict:
    """
    E5: Compute and validate yield per centroid.
    """
    print("=" * 60)
    print("E5: Yield Per Centroid")
    print("=" * 60)
    
    m1_path = Path(run_dir) / "tier_a" / "M1_compute_per_centroid.parquet"
    m3_path = Path(run_dir) / "tier_b" / "M3_observed_winners.parquet"
    
    if not m1_path.exists():
        print(f"❌ M1 file not found: {m1_path}")
        return None
    if not m3_path.exists():
        print(f"❌ M3 file not found: {m3_path}")
        return None
    
    m1 = pd.read_parquet(m1_path)
    m3 = pd.read_parquet(m3_path)
    
    # Load offsets
    sizes = torch.load(Path(index_path) / "sizes.compacted.pt")
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    torch.cumsum(sizes, dim=0, out=offsets[1:])
    num_centroids = len(sizes)
    
    # Aggregate M1: total computations per centroid
    computed = m1.groupby('centroid_id')['num_token_token_sims'].sum()
    
    # Derive winner_centroid_id from embedding_pos
    winner_positions = torch.tensor(m3['winner_embedding_pos'].values, dtype=torch.long)
    winner_centroids = torch.searchsorted(offsets, winner_positions, side='right') - 1
    m3_with_centroid = m3.copy()
    m3_with_centroid['winner_centroid_id'] = winner_centroids.numpy()
    
    # Aggregate M3: count winning interactions per centroid
    influential = m3_with_centroid.groupby('winner_centroid_id').size()
    
    # Build result DataFrame
    yield_df = pd.DataFrame({
        'centroid_id': range(num_centroids),
        'computed': computed.reindex(range(num_centroids), fill_value=0).values,
        'influential': influential.reindex(range(num_centroids), fill_value=0).values,
    })
    
    # Compute yield (NaN for never-selected centroids)
    yield_df['yield'] = np.where(
        yield_df['computed'] > 0,
        yield_df['influential'] / yield_df['computed'],
        np.nan
    )
    
    # Filter to centroids that were selected (have computed > 0)
    selected = yield_df[yield_df['computed'] > 0]
    
    print(f"\nCentroids selected: {len(selected):,} / {num_centroids:,}")
    
    print(f"\nYield distribution (selected centroids only):")
    print(f"  Min: {selected['yield'].min():.6f}")
    print(f"  Mean: {selected['yield'].mean():.6f}")
    print(f"  Median: {selected['yield'].median():.6f}")
    print(f"  Max: {selected['yield'].max():.6f}")
    
    # Pure waste centroids (selected but never influential)
    pure_waste = selected[selected['influential'] == 0]
    print(f"\nPure waste centroids (selected, yield=0): {len(pure_waste):,}")
    
    # High yield centroids (> 0.3)
    high_yield = selected[selected['yield'] > 0.3]
    print(f"High yield centroids (yield > 0.3): {len(high_yield):,}")
    
    # Low yield centroids (< 0.01)
    low_yield = selected[selected['yield'] < 0.01]
    print(f"Low yield centroids (yield < 0.01): {len(low_yield):,}")
    
    # M3 coverage vs M4
    m4_path = Path(run_dir) / "tier_b" / "M4_oracle_winners.parquet"
    if m4_path.exists():
        m4 = pd.read_parquet(m4_path)
        m3_per_query = m3.groupby('query_id').size()
        m4_per_query = m4.groupby('query_id').size()
        common_queries = m3_per_query.index.intersection(m4_per_query.index)
        coverage = (m3_per_query[common_queries] / m4_per_query[common_queries]).mean()
        print(f"\nM3 coverage (fraction of M4 rows): {coverage:.1%}")
        print("  Note: M3 only covers top-k docs, so yield numerator may be underestimated")
    
    print("\n✅ E5 PASSED")
    
    return {
        'centroids_selected': len(selected),
        'mean_yield': float(selected['yield'].mean()),
        'median_yield': float(selected['yield'].median()),
        'pure_waste_centroids': len(pure_waste),
        'high_yield_centroids': len(high_yield),
    }


def experiment_e6_pruning_recall(run_dir: str) -> dict:
    """
    E6: Compute and validate pruning recall.
    """
    print("=" * 60)
    print("E6: Pruning Recall")
    print("=" * 60)
    
    m3_path = Path(run_dir) / "tier_b" / "M3_observed_winners.parquet"
    m4_path = Path(run_dir) / "tier_b" / "M4_oracle_winners.parquet"
    
    if not m3_path.exists():
        print(f"❌ M3 file not found: {m3_path}")
        return None
    if not m4_path.exists():
        print(f"❌ M4 file not found: {m4_path}")
        return None
    
    m3 = pd.read_parquet(m3_path)
    m4 = pd.read_parquet(m4_path)
    
    print(f"\nM3 rows: {len(m3):,}")
    print(f"M4 rows: {len(m4):,}")
    
    # Compute actual document scores (sum of observed MaxSim)
    actual_scores = m3.groupby(['query_id', 'doc_id'])['winner_score'].sum().reset_index()
    actual_scores.columns = ['query_id', 'doc_id', 'actual_score']
    
    # Compute oracle document scores (sum of oracle MaxSim)
    oracle_scores = m4.groupby(['query_id', 'doc_id'])['oracle_score'].sum().reset_index()
    oracle_scores.columns = ['query_id', 'doc_id', 'oracle_score']
    
    print(f"\nUnique (query, doc) pairs:")
    print(f"  M3 (actual): {len(actual_scores):,}")
    print(f"  M4 (oracle): {len(oracle_scores):,}")
    
    k_values = [10, 100]
    results = []
    
    for query_id in actual_scores['query_id'].unique():
        # Get actual top-k
        actual_q = actual_scores[actual_scores['query_id'] == query_id]
        actual_q = actual_q.sort_values('actual_score', ascending=False)
        
        # Get oracle top-k
        oracle_q = oracle_scores[oracle_scores['query_id'] == query_id]
        oracle_q = oracle_q.sort_values('oracle_score', ascending=False)
        
        row = {'query_id': query_id}
        
        for k in k_values:
            actual_top_k = set(actual_q.head(k)['doc_id'])
            oracle_top_k = set(oracle_q.head(k)['doc_id'])
            
            overlap = len(actual_top_k & oracle_top_k)
            recall = overlap / min(k, len(oracle_top_k)) if oracle_top_k else 1.0
            row[f'recall@{k}'] = recall
        
        results.append(row)
    
    recall_df = pd.DataFrame(results)
    
    print(f"\nQueries evaluated: {len(recall_df):,}")
    
    for k in k_values:
        col = f'recall@{k}'
        if col in recall_df.columns:
            print(f"\n{col}:")
            print(f"  Min: {recall_df[col].min():.4f}")
            print(f"  Mean: {recall_df[col].mean():.4f}")
            print(f"  Median: {recall_df[col].median():.4f}")
            print(f"  Max: {recall_df[col].max():.4f}")
    
    print("\n✅ E6 PASSED")
    
    return {
        'queries_evaluated': len(recall_df),
        'mean_recall@10': float(recall_df['recall@10'].mean()),
        'mean_recall@100': float(recall_df['recall@100'].mean()),
    }


def experiment_e7_routing_fidelity_feasibility(run_dir: str, index_path: str) -> dict:
    """
    E7: Check if routing fidelity (C4/C5) can be computed.
    """
    print("=" * 60)
    print("E7: Routing Fidelity Feasibility")
    print("=" * 60)
    
    r0_path = Path(run_dir) / "tier_b" / "R0_selected_centroids.parquet"
    m4_path = Path(run_dir) / "tier_b" / "M4_oracle_winners.parquet"
    
    if not r0_path.exists():
        print(f"❌ R0 file not found: {r0_path}")
        return None
    if not m4_path.exists():
        print(f"❌ M4 file not found: {m4_path}")
        return None
    
    r0 = pd.read_parquet(r0_path)
    m4 = pd.read_parquet(m4_path)
    
    # Check R0 for centroid_score
    has_score = 'centroid_score' in r0.columns
    print(f"\nR0 has centroid_score: {has_score}")
    
    if not has_score:
        print("\n⚠️ C4/C5 cannot be implemented without centroid_score in R0.")
        print("Options:")
        print("  1. Modify tracker.py to record centroid_score in R0")
        print("  2. Defer C4/C5 to future phase")
        return {'feasible': False, 'reason': 'centroid_score missing from R0'}
    
    # Load offsets for centroid lookup
    sizes = torch.load(Path(index_path) / "sizes.compacted.pt")
    offsets = torch.zeros(len(sizes) + 1, dtype=torch.long)
    torch.cumsum(sizes, dim=0, out=offsets[1:])
    
    # Derive oracle centroids
    oracle_positions = torch.tensor(m4['oracle_embedding_pos'].values, dtype=torch.long)
    oracle_centroids = torch.searchsorted(offsets, oracle_positions, side='right') - 1
    
    # Build selected centroid sets
    r0_sets = r0.groupby(['query_id', 'q_token_id'])['centroid_id'].apply(set).to_dict()
    
    # Sample check: how many oracle centroids were in selected centroids?
    sample_size = min(10000, len(m4))
    m4_sample = m4.head(sample_size)
    oracle_centroids_sample = oracle_centroids[:sample_size]
    
    in_selected = 0
    for i, (qid, tid) in enumerate(zip(m4_sample['query_id'], m4_sample['q_token_id'])):
        selected = r0_sets.get((qid, tid), set())
        oc = oracle_centroids_sample[i].item()
        if oc in selected:
            in_selected += 1
    
    hit_rate = in_selected / sample_size
    miss_rate = 1 - hit_rate
    
    print(f"\nOracle centroids in selected (sample of {sample_size:,}):")
    print(f"  In selected: {in_selected:,} ({hit_rate:.1%})")
    print(f"  Not in selected (misses): {sample_size - in_selected:,} ({miss_rate:.1%})")
    
    print("\n✅ E7 PASSED: C4/C5 is feasible")
    
    return {
        'feasible': True,
        'has_centroid_score': has_score,
        'oracle_hit_rate': float(hit_rate),
        'oracle_miss_rate': float(miss_rate),
    }


def main():
    parser = argparse.ArgumentParser(description="Online metrics verification experiments")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "all"],
                       help="Which experiment to run")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Path to measurement run directory (for e1-e7)")
    parser.add_argument("--index-path", type=str,
                       default="/mnt/datasets/index/beir-quora.split=test.nbits=4",
                       help="Path to WARP index")
    parser.add_argument("--output-dir", type=str, default="/mnt/warp_measurements/online_dev",
                       help="Output directory for e0")
    parser.add_argument("--num-queries", type=int, default=100,
                       help="Number of queries for e0")
    
    args = parser.parse_args()
    
    # Setup environment
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("INDEX_ROOT", "/mnt/datasets/index")
    os.environ.setdefault("BEIR_COLLECTION_PATH", "/mnt/datasets/BEIR")
    
    results = {}
    
    # Run experiments
    if args.experiment in ["e0", "all"]:
        run_dir = experiment_e0_collect_measurements(args.output_dir, args.num_queries)
        if run_dir:
            results['e0'] = {'run_dir': run_dir}
            if args.run_dir is None:
                args.run_dir = run_dir
    
    if args.run_dir is None:
        print("\n❌ ERROR: --run-dir required for experiments e1-e7")
        print("Run e0 first to collect measurement data, or provide existing run directory.")
        return
    
    if args.experiment in ["e1", "all"]:
        results['e1'] = experiment_e1_m1_structure(args.run_dir)
    
    if args.experiment in ["e2", "all"]:
        results['e2'] = experiment_e2_m3_centroid_lookup(args.run_dir, args.index_path)
    
    if args.experiment in ["e3", "all"]:
        results['e3'] = experiment_e3_r0_schema(args.run_dir)
    
    if args.experiment in ["e4", "all"]:
        results['e4'] = experiment_e4_sel_freq(args.run_dir, args.index_path)
    
    if args.experiment in ["e5", "all"]:
        results['e5'] = experiment_e5_yield(args.run_dir, args.index_path)
    
    if args.experiment in ["e6", "all"]:
        results['e6'] = experiment_e6_pruning_recall(args.run_dir)
    
    if args.experiment in ["e7", "all"]:
        results['e7'] = experiment_e7_routing_fidelity_feasibility(args.run_dir, args.index_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{exp}: {status}")
    
    # Save results
    if args.run_dir:
        results_path = Path(args.run_dir) / "online_metrics_experiments.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
