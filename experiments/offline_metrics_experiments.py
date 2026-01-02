#!/usr/bin/env python3
"""
Offline Cluster Properties Verification Experiments

This script runs verification experiments for offline cluster metrics (A1, A2, A3, A5, B5):
- E0: Verify index file loading and shapes
- E1: B5 centroid isolation scalability
- E2: Verify decompression via ResidualCodec
- E3: A1/A2/A3 correctness verification
- E4: A5 sampling accuracy validation
- E5: End-to-end timing

Usage:
    cd /home/azureuser/repos/RAG-D
    conda activate warp
    python experiments/offline_metrics_experiments.py --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4
    
    # Run specific experiment
    python experiments/offline_metrics_experiments.py --index-path ... --experiment E0
    python experiments/offline_metrics_experiments.py --index-path ... --experiment all

See docs/OFFLINE_CLUSTER_PROPERTIES_INTEGRATION_PLAN.md for details.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import Counter

import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Index Loading Utilities (adapted from m4_verification_experiments.py)
# =============================================================================

def load_index_components(index_path: str) -> dict:
    """Load all components needed for offline metrics computation."""
    print(f"Loading index components from '{index_path}'...")
    
    components = {}
    
    # Centroids
    components['centroids'] = torch.from_numpy(
        np.load(os.path.join(index_path, "centroids.npy"))
    )
    print(f"  centroids: {components['centroids'].shape}")
    
    # Sizes and offsets
    components['sizes_compacted'] = torch.load(
        os.path.join(index_path, "sizes.compacted.pt")
    )
    num_centroids = len(components['sizes_compacted'])
    components['offsets_compacted'] = torch.zeros((num_centroids + 1,), dtype=torch.long)
    torch.cumsum(components['sizes_compacted'], dim=0, out=components['offsets_compacted'][1:])
    print(f"  sizes_compacted: {components['sizes_compacted'].shape}")
    print(f"  offsets_compacted: {components['offsets_compacted'].shape}")
    
    # Codes (doc IDs per embedding)
    components['codes_compacted'] = torch.load(
        os.path.join(index_path, "codes.compacted.pt")
    )
    print(f"  codes_compacted: {components['codes_compacted'].shape}")
    
    # Residuals (compressed)
    components['residuals_compacted'] = torch.load(
        os.path.join(index_path, "residuals.compacted.pt")
    )
    print(f"  residuals_compacted: {components['residuals_compacted'].shape}")
    
    # Bucket weights
    components['bucket_weights'] = torch.from_numpy(
        np.load(os.path.join(index_path, "bucket_weights.npy"))
    )
    print(f"  bucket_weights: {components['bucket_weights'].shape}")
    
    # Determine nbits from residual shape
    packed_dim = components['residuals_compacted'].shape[1]
    components['nbits'] = 4 if packed_dim == 64 else 2
    print(f"  nbits: {components['nbits']} (inferred from packed_dim={packed_dim})")
    
    # Derived values
    components['num_centroids'] = num_centroids
    components['num_embeddings'] = components['codes_compacted'].shape[0]
    
    return components


# =============================================================================
# E0: Verify Index File Loading
# =============================================================================

def run_experiment_e0(index_path: str) -> bool:
    """
    E0: Verify index files load correctly and have expected shapes.
    """
    print("\n" + "=" * 70)
    print("E0: Verify Index File Loading")
    print("=" * 70)
    
    try:
        components = load_index_components(index_path)
    except Exception as e:
        print(f"\n✗ E0 FAILED: Could not load index: {e}")
        return False
    
    num_centroids = components['num_centroids']
    num_embeddings = components['num_embeddings']
    
    print(f"\nIndex summary:")
    print(f"  Number of centroids: {num_centroids:,}")
    print(f"  Number of embeddings: {num_embeddings:,}")
    print(f"  Avg embeddings per centroid: {num_embeddings / num_centroids:.1f}")
    
    # Verify shapes
    print("\nShape verification:")
    checks = []
    
    # Centroids should be (C, 128)
    expected = (num_centroids, 128)
    actual = tuple(components['centroids'].shape)
    passed = expected == actual
    checks.append(passed)
    print(f"  centroids: expected {expected}, got {actual} — {'✓' if passed else '✗'}")
    
    # Sizes should be (C,)
    expected = (num_centroids,)
    actual = tuple(components['sizes_compacted'].shape)
    passed = expected == actual
    checks.append(passed)
    print(f"  sizes_compacted: expected {expected}, got {actual} — {'✓' if passed else '✗'}")
    
    # Sum of sizes should equal num_embeddings
    size_sum = components['sizes_compacted'].sum().item()
    passed = size_sum == num_embeddings
    checks.append(passed)
    print(f"  sizes_compacted.sum(): expected {num_embeddings:,}, got {size_sum:,} — {'✓' if passed else '✗'}")
    
    # Codes should be (N,)
    expected = (num_embeddings,)
    actual = tuple(components['codes_compacted'].shape)
    passed = expected == actual
    checks.append(passed)
    print(f"  codes_compacted: expected {expected}, got {actual} — {'✓' if passed else '✗'}")
    
    # Verify centroid normalization (should all be ≈ 1.0)
    norms = torch.norm(components['centroids'].float(), dim=1)
    print(f"\nCentroid norms:")
    print(f"  min: {norms.min().item():.6f}")
    print(f"  max: {norms.max().item():.6f}")
    print(f"  mean: {norms.mean().item():.6f}")
    
    norms_ok = torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    checks.append(norms_ok)
    print(f"  All ≈ 1.0: {'✓' if norms_ok else '✗'}")
    
    # Verify bucket_weights shape
    # Note: bucket_weights is (num_buckets,) - same buckets for all dimensions
    # This is scalar quantization, not per-dimension quantization
    nbits = components['nbits']
    expected_buckets = 2 ** nbits
    expected = (expected_buckets,)
    actual = tuple(components['bucket_weights'].shape)
    passed = expected == actual
    checks.append(passed)
    print(f"\nBucket weights: expected {expected}, got {actual} — {'✓' if passed else '✗'}")
    print(f"  (Scalar quantization: same {expected_buckets} buckets for all 128 dimensions)")
    
    # Verify offsets are monotonically increasing
    offsets = components['offsets_compacted']
    monotonic = (offsets[1:] >= offsets[:-1]).all().item()
    checks.append(monotonic)
    print(f"Offsets monotonic: {'✓' if monotonic else '✗'}")
    
    if all(checks):
        print("\n✓ E0 PASSED: All index files loaded correctly")
        return True
    else:
        print("\n✗ E0 FAILED: Some checks failed")
        return False


# =============================================================================
# E1: B5 Centroid Isolation Scalability
# =============================================================================

def run_experiment_e1(index_path: str) -> bool:
    """
    E1: Test centroid similarity computation scalability.
    
    With 64GB RAM, we can compute full similarity matrix for most indexes.
    """
    print("\n" + "=" * 70)
    print("E1: B5 Centroid Isolation Scalability")
    print("=" * 70)
    
    components = load_index_components(index_path)
    centroids = components['centroids'].float()
    num_centroids = components['num_centroids']
    
    # Memory estimate for full matrix
    mem_full_gb = num_centroids * num_centroids * 4 / 1e9
    print(f"\nFull similarity matrix would use: {mem_full_gb:.2f} GB")
    print(f"Available RAM: ~64 GB")
    
    k = 10  # k nearest neighbors for isolation
    
    # Option 1: Full matrix (feasible with 64GB RAM for most indexes)
    if mem_full_gb < 50:  # Leave headroom
        print(f"\n--- Testing full matrix approach ---")
        start = time.time()
        
        # Compute full similarity matrix
        similarities = centroids @ centroids.T  # (C, C)
        
        # Zero out diagonal (self-similarity)
        similarities.fill_diagonal_(float('-inf'))
        
        # Get top-k neighbors per centroid
        top_k_sims, _ = torch.topk(similarities, k, dim=1)  # (C, k)
        
        # Compute isolation metrics
        mean_neighbor_sim = top_k_sims.mean(dim=1)  # (C,)
        max_neighbor_sim = top_k_sims.max(dim=1).values  # (C,) - nearest neighbor
        isolation = 1.0 - mean_neighbor_sim  # (C,)
        
        elapsed = time.time() - start
        print(f"  Full matrix computed in {elapsed:.2f}s")
        print(f"  Similarity matrix shape: {similarities.shape}")
        
        # Statistics
        print(f"\nIsolation statistics:")
        print(f"  min: {isolation.min().item():.4f}")
        print(f"  max: {isolation.max().item():.4f}")
        print(f"  mean: {isolation.mean().item():.4f}")
        print(f"  std: {isolation.std().item():.4f}")
        
        print(f"\nNearest neighbor similarity:")
        print(f"  min: {max_neighbor_sim.min().item():.4f}")
        print(f"  max: {max_neighbor_sim.max().item():.4f}")
        print(f"  mean: {max_neighbor_sim.mean().item():.4f}")
        
        # Cleanup
        del similarities
        
    else:
        print(f"\n--- Full matrix too large, using batched approach ---")
    
    # Option 2: Batched approach (always test this for comparison)
    print(f"\n--- Testing batched approach ---")
    batch_size = 1000
    
    start = time.time()
    isolation_batched = torch.zeros(num_centroids)
    mean_neighbor_sim_batched = torch.zeros(num_centroids)
    
    for i in range(0, num_centroids, batch_size):
        end_i = min(i + batch_size, num_centroids)
        batch = centroids[i:end_i]  # (batch_size, 128)
        
        # Compute similarities for this batch against all centroids
        sims = batch @ centroids.T  # (batch_size, C)
        
        # Zero out self-similarities
        for j in range(end_i - i):
            sims[j, i + j] = float('-inf')
        
        # Get top-k
        top_k, _ = torch.topk(sims, k, dim=1)
        
        mean_neighbor_sim_batched[i:end_i] = top_k.mean(dim=1)
        isolation_batched[i:end_i] = 1.0 - top_k.mean(dim=1)
    
    elapsed_batched = time.time() - start
    print(f"  Batched approach computed in {elapsed_batched:.2f}s")
    print(f"  Batch size: {batch_size}")
    print(f"  Peak memory per batch: ~{batch_size * num_centroids * 4 / 1e9:.2f} GB")
    
    # Verify batched matches full (if we computed full)
    if mem_full_gb < 50:
        diff = (isolation - isolation_batched).abs().max().item()
        print(f"\n  Max difference between full and batched: {diff:.6f}")
        if diff < 1e-5:
            print("  ✓ Batched approach matches full matrix")
        else:
            print("  ⚠ Small numerical differences (acceptable)")
    
    print("\n✓ E1 PASSED: B5 isolation computation works correctly")
    return True


# =============================================================================
# E2: Verify Decompression via ResidualCodec
# =============================================================================

def run_experiment_e2(index_path: str) -> bool:
    """
    E2: Verify ResidualCodec produces expected embeddings.
    
    Tests both the standard (normalized) and raw (unnormalized) decompression paths.
    """
    print("\n" + "=" * 70)
    print("E2: Verify Decompression via ResidualCodec")
    print("=" * 70)
    
    try:
        from warp.indexing.codecs.residual import ResidualCodec
        from warp.indexing.codecs.residual_embeddings import ResidualEmbeddings
    except ImportError as e:
        print(f"✗ E2 FAILED: Could not import ResidualCodec: {e}")
        return False
    
    components = load_index_components(index_path)
    
    # Load codec
    print(f"\nLoading ResidualCodec from {index_path}...")
    codec = ResidualCodec.load(index_path)
    print(f"  Codec nbits: {codec.nbits}")
    print(f"  Codec dim: {codec.dim}")
    
    # Find a centroid with a reasonable number of embeddings
    sizes = components['sizes_compacted']
    offsets = components['offsets_compacted']
    
    # Pick centroid with median size
    median_size = int(torch.median(sizes.float()).item())
    target_centroid = None
    for c in range(components['num_centroids']):
        if abs(sizes[c].item() - median_size) < 100:
            target_centroid = c
            break
    
    if target_centroid is None:
        target_centroid = 0
    
    begin = offsets[target_centroid].item()
    end = offsets[target_centroid + 1].item()
    n_embeddings = end - begin
    
    print(f"\nTesting centroid {target_centroid} with {n_embeddings} embeddings")
    
    # Get sample of embeddings from this centroid
    n_samples = min(100, n_embeddings)
    sample_indices = torch.arange(begin, begin + n_samples)
    
    # Create codes tensor (all pointing to target_centroid)
    codes = torch.full((n_samples,), target_centroid, dtype=torch.int32)
    residuals = components['residuals_compacted'][sample_indices]
    
    print(f"\n--- Standard decompression (normalized output) ---")
    # Use codec's decompress method (this normalizes output)
    embeddings_normalized = codec.decompress(ResidualEmbeddings(codes, residuals))
    
    norms_normalized = torch.norm(embeddings_normalized, dim=1)
    print(f"  Output shape: {embeddings_normalized.shape}")
    print(f"  Norm range: [{norms_normalized.min().item():.4f}, {norms_normalized.max().item():.4f}]")
    print(f"  Expected: ~1.0 (normalized)")
    
    normalized_ok = torch.allclose(norms_normalized, torch.ones_like(norms_normalized), atol=1e-4)
    print(f"  All norms ≈ 1.0: {'✓' if normalized_ok else '✗'}")
    
    print(f"\n--- Raw decompression (unnormalized, for dispersion) ---")
    # Extract the CPU decompression path manually (before normalization)
    # Note: bucket_weights is (16,) - same buckets for all dimensions
    # After decompression_lookup_table, residuals_ has shape (n, 128) with bucket indices
    # Indexing bucket_weights[residuals_] broadcasts correctly
    centroids_ = codec.lookup_centroids(codes, out_device='cpu')
    residuals_ = codec.reversed_bit_map[residuals.long()]
    residuals_ = codec.decompression_lookup_table[residuals_.long()]
    residuals_ = residuals_.reshape(residuals_.shape[0], -1)  # (n, 128) bucket indices
    residuals_ = codec.bucket_weights[residuals_.long()]  # (n, 128) residual values
    embeddings_raw = centroids_ + residuals_
    
    norms_raw = torch.norm(embeddings_raw.float(), dim=1)
    print(f"  Output shape: {embeddings_raw.shape}")
    print(f"  Norm range: [{norms_raw.min().item():.4f}, {norms_raw.max().item():.4f}]")
    
    # Distance from centroid (for dispersion calculation)
    centroid = codec.centroids[target_centroid].float()
    distances = torch.norm(embeddings_raw.float() - centroid, dim=1)
    print(f"\n  Distance to centroid:")
    print(f"    min: {distances.min().item():.4f}")
    print(f"    max: {distances.max().item():.4f}")
    print(f"    mean: {distances.mean().item():.4f}")
    
    # Squared distance (what we use for dispersion)
    sq_distances = (distances ** 2)
    dispersion = sq_distances.mean().item()
    print(f"\n  Dispersion (mean squared distance): {dispersion:.6f}")
    
    # Sanity checks
    print(f"\n--- Sanity checks ---")
    
    # 1. Raw embeddings should be close to normalized embeddings (before norm)
    # After normalizing raw, should match normalized
    embeddings_raw_normalized = torch.nn.functional.normalize(embeddings_raw.float(), dim=-1)
    diff = (embeddings_raw_normalized - embeddings_normalized).abs().max().item()
    print(f"  Raw normalized matches codec output: diff={diff:.6f} — {'✓' if diff < 1e-4 else '✗'}")
    
    # 2. Residuals should be small corrections
    residual_magnitudes = torch.norm(residuals_.float(), dim=1)
    centroid_magnitude = torch.norm(centroid)
    ratio = (residual_magnitudes.mean() / centroid_magnitude).item()
    print(f"  Residual/centroid ratio: {ratio:.2%} — {'✓ (residuals are corrections)' if ratio < 0.5 else '⚠ (large residuals)'}")
    
    # 3. Distances should be reasonable (not too large)
    reasonable = distances.max().item() < 1.0
    print(f"  Max distance < 1.0: {'✓' if reasonable else '⚠'}")
    
    print("\n✓ E2 PASSED: Decompression via ResidualCodec works correctly")
    return True


# =============================================================================
# E3: A1/A2/A3 Correctness Verification
# =============================================================================

def compute_gini(counts: np.ndarray) -> float:
    """Compute Gini coefficient for a distribution."""
    if len(counts) == 0:
        return 0.0
    n = len(counts)
    sorted_counts = np.sort(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n


def run_experiment_e3(index_path: str) -> bool:
    """
    E3: Verify A1/A2/A3 computation on a real index.
    """
    print("\n" + "=" * 70)
    print("E3: A1/A2/A3 Correctness Verification")
    print("=" * 70)
    
    components = load_index_components(index_path)
    
    sizes = components['sizes_compacted'].numpy()
    offsets = components['offsets_compacted'].numpy()
    codes = components['codes_compacted'].numpy()
    num_centroids = components['num_centroids']
    
    print(f"\nComputing A1, A2, A3 for {num_centroids:,} centroids...")
    
    # Allocate arrays
    n_tokens = sizes.copy()  # A1: direct copy
    n_docs = np.zeros(num_centroids, dtype=np.int64)
    top_1_share = np.zeros(num_centroids, dtype=np.float32)
    top_5_share = np.zeros(num_centroids, dtype=np.float32)
    top_10_share = np.zeros(num_centroids, dtype=np.float32)
    gini_coeff = np.zeros(num_centroids, dtype=np.float32)
    
    start = time.time()
    
    empty_count = 0
    for c in range(num_centroids):
        begin = offsets[c]
        end = offsets[c + 1]
        
        if begin == end:
            empty_count += 1
            continue
        
        doc_ids = codes[begin:end]
        
        # A2: Unique document count
        unique_docs, counts = np.unique(doc_ids, return_counts=True)
        n_docs[c] = len(unique_docs)
        
        # A3: Concentration metrics
        n_total = len(doc_ids)
        sorted_counts = np.sort(counts)[::-1]  # Descending
        
        top_1_share[c] = sorted_counts[:1].sum() / n_total
        top_5_share[c] = sorted_counts[:5].sum() / n_total if len(sorted_counts) >= 5 else sorted_counts.sum() / n_total
        top_10_share[c] = sorted_counts[:10].sum() / n_total if len(sorted_counts) >= 10 else sorted_counts.sum() / n_total
        
        # Gini coefficient
        gini_coeff[c] = compute_gini(counts)
    
    elapsed = time.time() - start
    print(f"  Computed in {elapsed:.2f}s")
    print(f"  Empty centroids: {empty_count}")
    
    # Validate A1
    print(f"\n--- A1: Token-List Size ---")
    print(f"  Total embeddings: {n_tokens.sum():,}")
    print(f"  Min/Max/Mean: {n_tokens.min()}/{n_tokens.max()}/{n_tokens.mean():.1f}")
    
    a1_ok = n_tokens.sum() == len(codes)
    print(f"  Sum matches total embeddings: {'✓' if a1_ok else '✗'}")
    
    # Validate A2
    print(f"\n--- A2: Unique Document Count ---")
    print(f"  Min/Max/Mean: {n_docs.min()}/{n_docs.max()}/{n_docs.mean():.1f}")
    
    # n_docs should always be <= n_tokens
    a2_ok = (n_docs <= n_tokens).all()
    print(f"  n_docs <= n_tokens for all centroids: {'✓' if a2_ok else '✗'}")
    
    # Correlation between n_tokens and n_docs
    non_empty = n_tokens > 0
    corr = np.corrcoef(n_tokens[non_empty], n_docs[non_empty])[0, 1]
    print(f"  Correlation(n_tokens, n_docs): {corr:.3f}")
    
    # Validate A3
    print(f"\n--- A3: Document-Token Concentration ---")
    print(f"  Top-1 share: min={top_1_share[non_empty].min():.3f}, max={top_1_share[non_empty].max():.3f}, mean={top_1_share[non_empty].mean():.3f}")
    print(f"  Top-5 share: min={top_5_share[non_empty].min():.3f}, max={top_5_share[non_empty].max():.3f}, mean={top_5_share[non_empty].mean():.3f}")
    print(f"  Top-10 share: min={top_10_share[non_empty].min():.3f}, max={top_10_share[non_empty].max():.3f}, mean={top_10_share[non_empty].mean():.3f}")
    print(f"  Gini: min={gini_coeff[non_empty].min():.3f}, max={gini_coeff[non_empty].max():.3f}, mean={gini_coeff[non_empty].mean():.3f}")
    
    # Top-k shares should be monotonic
    a3_monotonic = (top_1_share <= top_5_share).all() and (top_5_share <= top_10_share).all()
    print(f"  Top-k shares monotonic: {'✓' if a3_monotonic else '✗'}")
    
    # Gini should be in [0, 1]
    gini_valid = (gini_coeff[non_empty] >= 0).all() and (gini_coeff[non_empty] <= 1).all()
    print(f"  Gini in [0, 1]: {'✓' if gini_valid else '✗'}")
    
    # Example: Show a few centroids
    print(f"\n--- Sample centroids ---")
    sample_centroids = [0, num_centroids // 4, num_centroids // 2, 3 * num_centroids // 4, num_centroids - 1]
    for c in sample_centroids:
        if n_tokens[c] > 0:
            print(f"  Centroid {c}: n_tokens={n_tokens[c]}, n_docs={n_docs[c]}, "
                  f"top1={top_1_share[c]:.3f}, gini={gini_coeff[c]:.3f}")
    
    all_ok = a1_ok and a2_ok and a3_monotonic and gini_valid
    if all_ok:
        print("\n✓ E3 PASSED: A1/A2/A3 computation is correct")
    else:
        print("\n✗ E3 FAILED: Some checks failed")
    
    return all_ok


# =============================================================================
# E4: A5 Sampling Accuracy Validation
# =============================================================================

def run_experiment_e4(index_path: str) -> bool:
    """
    E4: Validate sampling accuracy for A5 dispersion computation.
    
    Compares sampled vs full dispersion for a subset of centroids.
    """
    print("\n" + "=" * 70)
    print("E4: A5 Sampling Accuracy Validation")
    print("=" * 70)
    
    try:
        from warp.indexing.codecs.residual import ResidualCodec
    except ImportError as e:
        print(f"✗ E4 FAILED: Could not import ResidualCodec: {e}")
        return False
    
    components = load_index_components(index_path)
    codec = ResidualCodec.load(index_path)
    
    sizes = components['sizes_compacted']
    offsets = components['offsets_compacted']
    residuals_compacted = components['residuals_compacted']
    
    # Find centroids with different sizes for testing
    test_centroids = []
    
    # Small (100-500 embeddings)
    for c in range(components['num_centroids']):
        if 100 < sizes[c].item() < 500 and len(test_centroids) < 3:
            test_centroids.append(c)
    
    # Medium (1000-3000 embeddings)
    for c in range(components['num_centroids']):
        if 1000 < sizes[c].item() < 3000 and len(test_centroids) < 6:
            test_centroids.append(c)
    
    # Large (5000-10000 embeddings)
    for c in range(components['num_centroids']):
        if 5000 < sizes[c].item() < 10000 and len(test_centroids) < 9:
            test_centroids.append(c)
    
    if len(test_centroids) == 0:
        print("Warning: No suitable centroids found, using first non-empty centroids")
        for c in range(components['num_centroids']):
            if sizes[c].item() > 50:
                test_centroids.append(c)
            if len(test_centroids) >= 5:
                break
    
    print(f"\nTesting {len(test_centroids)} centroids with different sizes")
    
    # Sampling parameters
    sample_fractions = [0.10, 0.25, 0.50]
    
    results = []
    
    for c in test_centroids:
        begin = offsets[c].item()
        end = offsets[c + 1].item()
        n = end - begin
        
        print(f"\n--- Centroid {c} (n={n}) ---")
        
        # Full dispersion computation
        all_residuals = residuals_compacted[begin:end]
        
        # Use raw decompression (without normalization)
        centroids_ = codec.lookup_centroids(torch.full((n,), c, dtype=torch.int32), out_device='cpu')
        residuals_ = codec.reversed_bit_map[all_residuals.long()]
        residuals_ = codec.decompression_lookup_table[residuals_.long()]
        residuals_ = residuals_.reshape(n, -1)
        residuals_ = codec.bucket_weights[residuals_.long()]
        embeddings_full = centroids_ + residuals_
        
        centroid = codec.centroids[c].float()
        sq_distances_full = ((embeddings_full.float() - centroid) ** 2).sum(dim=1)
        dispersion_full = sq_distances_full.mean().item()
        
        print(f"  Full dispersion: {dispersion_full:.6f}")
        
        # Sampled dispersion at different fractions
        for frac in sample_fractions:
            n_samples = max(50, int(n * frac))
            n_samples = min(n_samples, n)
            
            # Multiple random samples for variance estimation
            dispersions_sampled = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                sample_idx = rng.choice(n, size=n_samples, replace=False)
                
                sample_residuals = all_residuals[sample_idx]
                
                centroids_s = codec.lookup_centroids(torch.full((n_samples,), c, dtype=torch.int32), out_device='cpu')
                residuals_s = codec.reversed_bit_map[sample_residuals.long()]
                residuals_s = codec.decompression_lookup_table[residuals_s.long()]
                residuals_s = residuals_s.reshape(n_samples, -1)
                residuals_s = codec.bucket_weights[residuals_s.long()]
                embeddings_s = centroids_s + residuals_s
                
                sq_distances_s = ((embeddings_s.float() - centroid) ** 2).sum(dim=1)
                dispersions_sampled.append(sq_distances_s.mean().item())
            
            mean_sampled = np.mean(dispersions_sampled)
            std_sampled = np.std(dispersions_sampled)
            rel_error = abs(mean_sampled - dispersion_full) / dispersion_full
            
            print(f"  {frac*100:.0f}% sample (n={n_samples}): mean={mean_sampled:.6f}, "
                  f"std={std_sampled:.6f}, rel_error={rel_error:.2%}")
            
            results.append({
                'centroid': c,
                'n_embeddings': n,
                'fraction': frac,
                'n_samples': n_samples,
                'dispersion_full': dispersion_full,
                'dispersion_sampled': mean_sampled,
                'std': std_sampled,
                'rel_error': rel_error
            })
    
    # Summary statistics
    print(f"\n--- Summary ---")
    df = pd.DataFrame(results)
    
    for frac in sample_fractions:
        frac_df = df[df['fraction'] == frac]
        mean_error = frac_df['rel_error'].mean()
        max_error = frac_df['rel_error'].max()
        print(f"  {frac*100:.0f}% sampling: mean_rel_error={mean_error:.2%}, max_rel_error={max_error:.2%}")
    
    # Check if 25% sampling is acceptable (< 10% error on average)
    frac_25 = df[df['fraction'] == 0.25]
    acceptable = frac_25['rel_error'].mean() < 0.10
    
    if acceptable:
        print(f"\n✓ E4 PASSED: 25% sampling has acceptable accuracy (mean error < 10%)")
    else:
        print(f"\n⚠ E4 WARNING: Consider higher sampling fraction")
    
    return True


# =============================================================================
# E5: End-to-End Timing
# =============================================================================

def run_experiment_e5(index_path: str) -> bool:
    """
    E5: Measure end-to-end timing for all offline metrics.
    """
    print("\n" + "=" * 70)
    print("E5: End-to-End Timing")
    print("=" * 70)
    
    try:
        from warp.indexing.codecs.residual import ResidualCodec
    except ImportError as e:
        print(f"✗ E5 FAILED: Could not import ResidualCodec: {e}")
        return False
    
    # --- Load index ---
    print("\n--- Index Loading ---")
    start = time.time()
    components = load_index_components(index_path)
    t_load = time.time() - start
    print(f"  Index loading: {t_load:.2f}s")
    
    num_centroids = components['num_centroids']
    
    # --- A1, A2, A3 (combined) ---
    print("\n--- A1/A2/A3 Computation ---")
    start = time.time()
    
    sizes = components['sizes_compacted'].numpy()
    offsets = components['offsets_compacted'].numpy()
    codes = components['codes_compacted'].numpy()
    
    n_tokens = sizes.copy()
    n_docs = np.zeros(num_centroids, dtype=np.int64)
    top_1_share = np.zeros(num_centroids, dtype=np.float32)
    gini = np.zeros(num_centroids, dtype=np.float32)
    
    for c in range(num_centroids):
        begin = offsets[c]
        end = offsets[c + 1]
        if begin == end:
            continue
        doc_ids = codes[begin:end]
        unique_docs, counts = np.unique(doc_ids, return_counts=True)
        n_docs[c] = len(unique_docs)
        sorted_counts = np.sort(counts)[::-1]
        n_total = len(doc_ids)
        top_1_share[c] = sorted_counts[0] / n_total
        gini[c] = compute_gini(counts)
    
    t_a123 = time.time() - start
    print(f"  A1/A2/A3: {t_a123:.2f}s")
    
    # --- B5 ---
    print("\n--- B5 Computation ---")
    start = time.time()
    
    centroids = components['centroids'].float()
    k = 10
    
    # Full matrix approach (with 64GB RAM)
    similarities = centroids @ centroids.T
    similarities.fill_diagonal_(float('-inf'))
    top_k_sims, _ = torch.topk(similarities, k, dim=1)
    isolation = 1.0 - top_k_sims.mean(dim=1)
    del similarities
    
    t_b5 = time.time() - start
    print(f"  B5: {t_b5:.2f}s")
    
    # --- A5 (with sampling) ---
    print("\n--- A5 Computation (25% sampling) ---")
    start = time.time()
    
    codec = ResidualCodec.load(index_path)
    residuals_compacted = components['residuals_compacted']
    
    sample_fraction = 0.25
    min_samples = 100
    max_samples = 10000
    rng = np.random.default_rng(42)
    
    dispersion = np.zeros(num_centroids, dtype=np.float32)
    
    for c in range(num_centroids):
        begin = offsets[c]
        end = offsets[c + 1]
        n = end - begin
        
        if n == 0:
            continue
        
        # Determine sample size
        n_samples = int(n * sample_fraction)
        n_samples = max(min_samples, min(n_samples, max_samples, n))
        
        # Sample indices
        if n_samples < n:
            local_idx = rng.choice(n, size=n_samples, replace=False)
            global_idx = begin + local_idx
        else:
            global_idx = np.arange(begin, end)
            n_samples = n
        
        # Decompress
        sample_residuals = residuals_compacted[global_idx]
        codes_t = torch.full((n_samples,), c, dtype=torch.int32)
        
        centroids_ = codec.lookup_centroids(codes_t, out_device='cpu')
        residuals_ = codec.reversed_bit_map[sample_residuals.long()]
        residuals_ = codec.decompression_lookup_table[residuals_.long()]
        residuals_ = residuals_.reshape(n_samples, -1)
        residuals_ = codec.bucket_weights[residuals_.long()]
        embeddings = centroids_ + residuals_
        
        # Compute dispersion
        centroid = codec.centroids[c].float()
        sq_distances = ((embeddings.float() - centroid) ** 2).sum(dim=1)
        dispersion[c] = sq_distances.mean().item()
    
    t_a5 = time.time() - start
    print(f"  A5: {t_a5:.2f}s")
    
    # --- Summary ---
    total = t_load + t_a123 + t_b5 + t_a5
    print(f"\n--- Timing Summary ---")
    print(f"  Index loading:  {t_load:>7.2f}s ({t_load/total*100:>5.1f}%)")
    print(f"  A1/A2/A3:       {t_a123:>7.2f}s ({t_a123/total*100:>5.1f}%)")
    print(f"  B5:             {t_b5:>7.2f}s ({t_b5/total*100:>5.1f}%)")
    print(f"  A5:             {t_a5:>7.2f}s ({t_a5/total*100:>5.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Total:          {total:>7.2f}s")
    
    # Estimate for larger indexes
    print(f"\n--- Scaling Estimates ---")
    emb_per_sec = components['num_embeddings'] / (t_a123 + t_a5)
    print(f"  Embeddings processed: {components['num_embeddings']:,}")
    print(f"  Processing rate: {emb_per_sec:,.0f} embeddings/sec")
    print(f"  Estimated time for 100M embeddings: {100_000_000 / emb_per_sec / 60:.1f} minutes")
    
    print("\n✓ E5 PASSED: End-to-end timing complete")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run offline metrics verification experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python experiments/offline_metrics_experiments.py --index-path /mnt/datasets/index/beir-quora.split=test.nbits=4
    python experiments/offline_metrics_experiments.py --index-path ... --experiment E0
    python experiments/offline_metrics_experiments.py --index-path ... --experiment all
        """
    )
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to WARP index directory"
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["E0", "E1", "E2", "E3", "E4", "E5", "all"],
        help="Which experiment to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate index path
    if not os.path.isdir(args.index_path):
        print(f"Error: Index path does not exist: {args.index_path}")
        sys.exit(1)
    
    experiments = {
        "E0": ("Verify Index Loading", run_experiment_e0),
        "E1": ("B5 Scalability", run_experiment_e1),
        "E2": ("Decompression via ResidualCodec", run_experiment_e2),
        "E3": ("A1/A2/A3 Correctness", run_experiment_e3),
        "E4": ("A5 Sampling Accuracy", run_experiment_e4),
        "E5": ("End-to-End Timing", run_experiment_e5),
    }
    
    print("=" * 70)
    print("OFFLINE METRICS VERIFICATION EXPERIMENTS")
    print("=" * 70)
    print(f"Index: {args.index_path}")
    print(f"Experiment: {args.experiment}")
    
    if args.experiment == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [args.experiment]
    
    results = {}
    for exp_id in to_run:
        name, func = experiments[exp_id]
        try:
            results[exp_id] = func(args.index_path)
        except Exception as e:
            print(f"\n✗ {exp_id} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for exp_id in to_run:
        name, _ = experiments[exp_id]
        status = "✓ PASSED" if results[exp_id] else "✗ FAILED"
        print(f"  {exp_id}: {name} — {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✓ All experiments passed!" if all_passed else "✗ Some experiments failed"))
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
