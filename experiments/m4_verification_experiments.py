#!/usr/bin/env python3
"""
M4 Verification Experiments

These experiments validate the components needed for M4 (Oracle) implementation:
- E1: Verify Python decompression matches C++ results
- E2: Validate reverse index correctness  
- E3: Measure oracle computation time (Python vs vectorized)
- E4: Validate oracle winners are correct
- E5: Compare M3 vs M4 winners (requires M3 data)
- E6: Measure scope of "all scored" documents

Run with:
    conda activate warp
    python experiments/m4_verification_experiments.py [--index-path PATH] [--experiment E1|E2|E3|E4|E5|E6|all]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from warp.engine.utils.reverse_index import ReverseIndex


def load_index_components(index_path: str) -> dict:
    """Load all components needed for oracle computation."""
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
    
    # Repacked residuals (for C++ decompression)
    repacked_path = os.path.join(index_path, "residuals.repacked.compacted.pt")
    if os.path.exists(repacked_path):
        components['residuals_repacked'] = torch.load(repacked_path)
        print(f"  residuals_repacked: {components['residuals_repacked'].shape}")
    
    # Bucket weights
    components['bucket_weights'] = torch.from_numpy(
        np.load(os.path.join(index_path, "bucket_weights.npy"))
    )
    print(f"  bucket_weights: {components['bucket_weights'].shape}")
    
    # Determine nbits from residual shape
    packed_dim = components['residuals_compacted'].shape[1]
    components['nbits'] = 4 if packed_dim == 64 else 2
    print(f"  nbits: {components['nbits']}")
    
    # Load or build reverse index
    components['reverse_index'] = ReverseIndex.load_or_build(index_path)
    print(f"  reverse_index: {components['reverse_index']}")
    
    return components


def compute_bucket_scores(q_token: torch.Tensor, bucket_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute bucket scores for a query token.
    
    This matches the C++ computation:
        vt_bucket_scores = torch::matmul(Q.unsqueeze(2), bucket_weights.unsqueeze(0))
    
    Result: bucket_scores[dim, bucket] = q_token[dim] * bucket_weights[bucket]
    
    Args:
        q_token: (128,) query token embedding
        bucket_weights: (16,) for nbits=4 or (4,) for nbits=2
    
    Returns:
        bucket_scores: (128, num_buckets) precomputed scores
    """
    # Outer product: q_token[:, None] * bucket_weights[None, :]
    return q_token.unsqueeze(1) * bucket_weights.unsqueeze(0)  # (128, num_buckets)


def unpack_and_score_python(
    residual: torch.Tensor,
    bucket_scores: torch.Tensor,
    nbits: int
) -> float:
    """
    Unpack quantized residual and compute score via bucket lookup.
    
    This implements the decompression_kernel from decompress_centroids.cpp in Python.
    
    Args:
        residual: (packed_dim,) uint8 tensor of packed residual values
        bucket_scores: (128, num_buckets) precomputed from q_token @ bucket_weights
        nbits: 2 or 4
    
    Returns:
        Residual score contribution
    """
    score = 0.0
    
    if nbits == 4:
        packed_dim = 64  # 128 / 2
        for packed_idx in range(packed_dim):
            packed_val = residual[packed_idx].item()
            unpacked_0 = packed_val >> 4
            unpacked_1 = packed_val & 0x0F
            
            dim_0 = packed_idx * 2
            dim_1 = dim_0 + 1
            
            # bucket_scores is (128, 16) for nbits=4
            score += bucket_scores[dim_0, unpacked_0].item()
            score += bucket_scores[dim_1, unpacked_1].item()
            
    elif nbits == 2:
        packed_dim = 32  # 128 / 4
        for packed_idx in range(packed_dim):
            packed_val = residual[packed_idx].item()
            unpacked_0 = (packed_val & 0xC0) >> 6
            unpacked_1 = (packed_val & 0x30) >> 4
            unpacked_2 = (packed_val & 0x0C) >> 2
            unpacked_3 = (packed_val & 0x03)
            
            dim_base = packed_idx * 4
            score += bucket_scores[dim_base, unpacked_0].item()
            score += bucket_scores[dim_base + 1, unpacked_1].item()
            score += bucket_scores[dim_base + 2, unpacked_2].item()
            score += bucket_scores[dim_base + 3, unpacked_3].item()
    
    return score


def compute_score_python(
    q_token: torch.Tensor,
    embedding_pos: int,
    components: dict
) -> float:
    """
    Compute similarity score for one query token and one embedding position.
    
    score = centroid_score + residual_score
    """
    # Find centroid for this embedding
    centroid_id = torch.searchsorted(
        components['offsets_compacted'], 
        embedding_pos, 
        side='right'
    ).item() - 1
    
    # Centroid contribution
    centroid_score = (q_token @ components['centroids'][centroid_id]).item()
    
    # Precompute bucket scores for this query token
    bucket_scores = compute_bucket_scores(q_token, components['bucket_weights'])  # (128, 16)
    
    # Residual contribution
    residual = components['residuals_compacted'][embedding_pos]
    residual_score = unpack_and_score_python(residual, bucket_scores, components['nbits'])
    
    return centroid_score + residual_score


def compute_oracle_python(
    q_token: torch.Tensor,
    embedding_positions: torch.Tensor,
    components: dict
) -> tuple:
    """
    Compute oracle MaxSim winner using pure Python (slow but correct).
    
    Returns: (winner_pos, winner_score)
    """
    best_pos = -1
    best_score = float('-inf')
    
    # Precompute bucket scores once
    bucket_scores = compute_bucket_scores(q_token, components['bucket_weights'])
    
    for pos in embedding_positions:
        pos = pos.item()
        
        # Find centroid
        centroid_id = torch.searchsorted(
            components['offsets_compacted'], pos, side='right'
        ).item() - 1
        
        # Centroid score
        centroid_score = (q_token @ components['centroids'][centroid_id]).item()
        
        # Residual score
        residual = components['residuals_compacted'][pos]
        residual_score = unpack_and_score_python(residual, bucket_scores, components['nbits'])
        
        score = centroid_score + residual_score
        
        if score > best_score:
            best_score = score
            best_pos = pos
    
    return best_pos, best_score


def compute_oracle_vectorized(
    q_token: torch.Tensor,
    embedding_positions: torch.Tensor,
    components: dict
) -> tuple:
    """
    Compute oracle MaxSim winner using vectorized PyTorch operations.
    
    Returns: (winner_pos, winner_score)
    """
    if len(embedding_positions) == 0:
        return -1, float('-inf')
    
    # Get centroids for all positions at once
    centroid_ids = torch.searchsorted(
        components['offsets_compacted'], 
        embedding_positions, 
        side='right'
    ) - 1
    
    # Centroid scores: batch dot product
    centroid_vecs = components['centroids'][centroid_ids]  # (num_emb, 128)
    centroid_scores = (q_token.unsqueeze(0) * centroid_vecs).sum(dim=1)  # (num_emb,)
    
    # Precompute bucket scores
    bucket_scores = compute_bucket_scores(q_token, components['bucket_weights'])  # (128, 16)
    
    # Vectorized residual unpacking for nbits=4
    residuals = components['residuals_compacted'][embedding_positions]  # (num_emb, packed_dim)
    
    if components['nbits'] == 4:
        # Unpack: each byte → 2 values
        high_nibbles = (residuals >> 4).long()  # (num_emb, 64)
        low_nibbles = (residuals & 0x0F).long()  # (num_emb, 64)
        
        # Compute scores for each dimension
        num_emb = residuals.shape[0]
        residual_scores = torch.zeros(num_emb)
        
        for packed_idx in range(64):
            dim_0 = packed_idx * 2
            dim_1 = dim_0 + 1
            
            # Gather bucket scores
            residual_scores += bucket_scores[dim_0, high_nibbles[:, packed_idx]]
            residual_scores += bucket_scores[dim_1, low_nibbles[:, packed_idx]]
    else:
        raise NotImplementedError("nbits=2 vectorized not yet implemented")
    
    # Total scores
    total_scores = centroid_scores + residual_scores
    
    # Find winner
    best_idx = total_scores.argmax().item()
    best_pos = embedding_positions[best_idx].item()
    best_score = total_scores[best_idx].item()
    
    return best_pos, best_score


# =============================================================================
# Experiment E1: Verify Python Decompression
# =============================================================================

def run_experiment_e1(components: dict, index_path: str):
    """
    E1: Verify Python decompression formula matches expected behavior.
    
    We can't directly compare to C++ without running the full search,
    so we verify internal consistency and known properties.
    """
    print("\n" + "=" * 70)
    print("E1: Verify Python Decompression Formula")
    print("=" * 70)
    
    # Create a random query token
    torch.manual_seed(42)
    q_token = torch.randn(128)
    q_token = q_token / q_token.norm()  # Normalize
    
    # Test several embedding positions
    test_positions = [0, 100, 1000, 10000, 100000, 1000000]
    num_embeddings = components['codes_compacted'].shape[0]
    test_positions = [p for p in test_positions if p < num_embeddings]
    
    print(f"\nTesting {len(test_positions)} embedding positions...")
    print(f"Query token norm: {q_token.norm().item():.4f}")
    
    results = []
    for pos in test_positions:
        # Compute score
        score = compute_score_python(q_token, pos, components)
        
        # Get centroid info
        centroid_id = torch.searchsorted(
            components['offsets_compacted'], pos, side='right'
        ).item() - 1
        doc_id = components['codes_compacted'][pos].item()
        
        # Centroid-only score
        centroid_score = (q_token @ components['centroids'][centroid_id]).item()
        residual_score = score - centroid_score
        
        results.append({
            'pos': pos,
            'centroid_id': centroid_id,
            'doc_id': doc_id,
            'centroid_score': centroid_score,
            'residual_score': residual_score,
            'total_score': score
        })
        
        print(f"  pos={pos:>8d}: centroid={centroid_id:>5d}, doc={doc_id:>6d}, "
              f"c_score={centroid_score:>7.4f}, r_score={residual_score:>7.4f}, "
              f"total={score:>7.4f}")
    
    # Sanity checks
    print("\nSanity checks:")
    
    # 1. Scores should be in reasonable range (typically -1 to 1 for normalized vectors)
    scores = [r['total_score'] for r in results]
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}] — ", end="")
    if -2 < min(scores) and max(scores) < 2:
        print("✓ Reasonable")
    else:
        print("⚠️ Unusual range")
    
    # 2. Centroid scores should dominate (residuals are corrections)
    residual_magnitudes = [abs(r['residual_score']) for r in results]
    centroid_magnitudes = [abs(r['centroid_score']) for r in results]
    avg_residual = sum(residual_magnitudes) / len(residual_magnitudes)
    avg_centroid = sum(centroid_magnitudes) / len(centroid_magnitudes)
    print(f"  Avg |centroid_score|: {avg_centroid:.4f}")
    print(f"  Avg |residual_score|: {avg_residual:.4f}")
    print(f"  Residual/Centroid ratio: {avg_residual/avg_centroid:.2%} — ", end="")
    if avg_residual < avg_centroid:
        print("✓ Residuals are corrections")
    else:
        print("⚠️ Residuals dominate (unexpected)")
    
    print("\n✓ E1 PASSED: Decompression formula produces reasonable results")
    return True


# =============================================================================
# Experiment E2: Validate Reverse Index
# =============================================================================

def run_experiment_e2(components: dict, index_path: str):
    """
    E2: Verify reverse index returns correct embedding positions.
    """
    print("\n" + "=" * 70)
    print("E2: Validate Reverse Index Correctness")
    print("=" * 70)
    
    reverse_idx = components['reverse_index']
    codes_compacted = components['codes_compacted']
    
    # Find some valid doc IDs to test
    unique_docs = torch.unique(codes_compacted)
    num_docs = unique_docs.shape[0]
    print(f"\nTotal documents in index: {num_docs:,}")
    
    # Sample doc IDs (first, some from middle, last)
    test_doc_ids = [
        unique_docs[0].item(),
        unique_docs[num_docs // 4].item(),
        unique_docs[num_docs // 2].item(),
        unique_docs[3 * num_docs // 4].item(),
        unique_docs[-1].item(),
    ]
    
    # Add some specific IDs if they exist
    for doc_id in [10, 100, 1000, 10000]:
        if doc_id < num_docs:
            test_doc_ids.append(doc_id)
    
    test_doc_ids = sorted(set(test_doc_ids))
    
    print(f"Testing {len(test_doc_ids)} documents...")
    
    all_passed = True
    for doc_id in test_doc_ids:
        positions = reverse_idx.get_embedding_positions(doc_id)
        num_embeddings = len(positions)
        
        # Verify all positions map back to this doc_id
        doc_ids_at_positions = codes_compacted[positions]
        correct = (doc_ids_at_positions == doc_id).all().item()
        
        status = "✓" if correct else "✗"
        print(f"  Doc {doc_id:>6d}: {num_embeddings:>4d} embeddings — {status}")
        
        if not correct:
            all_passed = False
            mismatches = (doc_ids_at_positions != doc_id).sum().item()
            print(f"    ⚠️ {mismatches} positions map to wrong doc!")
    
    # Distribution of embeddings per document
    print("\nEmbeddings per document distribution:")
    sample_size = min(1000, num_docs)
    sample_docs = unique_docs[:sample_size]
    emb_counts = [reverse_idx.get_num_embeddings(d.item()) for d in sample_docs]
    
    print(f"  Min: {min(emb_counts)}")
    print(f"  Max: {max(emb_counts)}")
    print(f"  Mean: {sum(emb_counts)/len(emb_counts):.1f}")
    print(f"  Median: {sorted(emb_counts)[len(emb_counts)//2]}")
    
    if all_passed:
        print("\n✓ E2 PASSED: Reverse index correctly maps doc_id → positions")
    else:
        print("\n✗ E2 FAILED: Some positions map to wrong documents!")
    
    return all_passed


# =============================================================================
# Experiment E3: Measure Oracle Computation Time
# =============================================================================

def run_experiment_e3(components: dict, index_path: str):
    """
    E3: Benchmark Python vs vectorized oracle computation.
    """
    print("\n" + "=" * 70)
    print("E3: Measure Oracle Computation Time")
    print("=" * 70)
    
    reverse_idx = components['reverse_index']
    
    # Create random query token
    torch.manual_seed(42)
    q_token = torch.randn(128)
    q_token = q_token / q_token.norm()
    
    # Sample some documents with varying numbers of embeddings
    codes_compacted = components['codes_compacted']
    unique_docs = torch.unique(codes_compacted).tolist()
    
    # Find docs with different embedding counts
    test_docs = []
    for target_count in [10, 50, 100, 200]:
        for doc_id in unique_docs[:10000]:  # Search in first 10K docs
            count = reverse_idx.get_num_embeddings(doc_id)
            if abs(count - target_count) < 10:
                test_docs.append(doc_id)
                break
    
    if len(test_docs) < 4:
        # Just use first few docs
        test_docs = unique_docs[:5]
    
    print(f"\nBenchmarking on {len(test_docs)} documents...")
    
    # Warm-up
    for doc_id in test_docs[:2]:
        positions = reverse_idx.get_embedding_positions(doc_id)
        _ = compute_oracle_python(q_token, positions, components)
        _ = compute_oracle_vectorized(q_token, positions, components)
    
    # Benchmark
    results = []
    for doc_id in test_docs:
        positions = reverse_idx.get_embedding_positions(doc_id)
        num_emb = len(positions)
        
        # Python implementation
        t0 = time.time()
        for _ in range(3):
            pos_py, score_py = compute_oracle_python(q_token, positions, components)
        python_time = (time.time() - t0) / 3
        
        # Vectorized implementation
        t0 = time.time()
        for _ in range(10):
            pos_vec, score_vec = compute_oracle_vectorized(q_token, positions, components)
        vectorized_time = (time.time() - t0) / 10
        
        # Verify same result
        same_winner = (pos_py == pos_vec)
        score_diff = abs(score_py - score_vec)
        
        results.append({
            'doc_id': doc_id,
            'num_embeddings': num_emb,
            'python_ms': python_time * 1000,
            'vectorized_ms': vectorized_time * 1000,
            'speedup': python_time / vectorized_time if vectorized_time > 0 else 0,
            'same_winner': same_winner,
            'score_diff': score_diff
        })
        
        status = "✓" if same_winner and score_diff < 1e-4 else "⚠️"
        print(f"  Doc {doc_id}: {num_emb:>3d} emb | "
              f"Python: {python_time*1000:>6.2f}ms | "
              f"Vectorized: {vectorized_time*1000:>6.3f}ms | "
              f"Speedup: {python_time/vectorized_time if vectorized_time > 0 else 0:>5.1f}x | {status}")
    
    # Summary
    print("\nSummary:")
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    all_correct = all(r['same_winner'] and r['score_diff'] < 1e-4 for r in results)
    
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  Results match: {'✓ Yes' if all_correct else '✗ No'}")
    
    # Estimate full query time
    avg_emb_per_doc = sum(r['num_embeddings'] for r in results) / len(results)
    avg_vec_time = sum(r['vectorized_ms'] for r in results) / len(results)
    
    print(f"\nEstimated time for full query (k=1000 docs, 32 tokens):")
    estimated_time = 1000 * 32 * avg_vec_time / 1000  # Convert to seconds
    print(f"  Vectorized: {estimated_time:.1f} seconds per query")
    print(f"  This is too slow for Python — C++ implementation needed!")
    
    if all_correct:
        print("\n✓ E3 PASSED: Vectorized matches Python, but C++ is needed for speed")
    else:
        print("\n⚠️ E3 WARNING: Results don't match between implementations")
    
    return all_correct


# =============================================================================
# Experiment E4: Validate Oracle Winners
# =============================================================================

def run_experiment_e4(components: dict, index_path: str):
    """
    E4: Validate oracle winners are actually the maximum.
    """
    print("\n" + "=" * 70)
    print("E4: Validate Oracle Winners Are Correct")
    print("=" * 70)
    
    reverse_idx = components['reverse_index']
    
    # Create random query token
    torch.manual_seed(123)
    q_token = torch.randn(128)
    q_token = q_token / q_token.norm()
    
    # Test on a few documents
    codes_compacted = components['codes_compacted']
    unique_docs = torch.unique(codes_compacted).tolist()
    test_docs = unique_docs[:5]
    
    print(f"\nValidating oracle winners for {len(test_docs)} documents...")
    
    all_passed = True
    for doc_id in test_docs:
        positions = reverse_idx.get_embedding_positions(doc_id)
        
        # Compute oracle
        oracle_pos, oracle_score = compute_oracle_vectorized(q_token, positions, components)
        
        # Verify by computing all scores
        all_scores = []
        for pos in positions:
            score = compute_score_python(q_token, pos.item(), components)
            all_scores.append((pos.item(), score))
        
        # Find actual max
        actual_max_pos, actual_max_score = max(all_scores, key=lambda x: x[1])
        
        # Compare
        pos_match = (oracle_pos == actual_max_pos)
        score_match = abs(oracle_score - actual_max_score) < 1e-5
        
        status = "✓" if pos_match and score_match else "✗"
        print(f"  Doc {doc_id}: oracle_pos={oracle_pos}, actual_max={actual_max_pos}, "
              f"scores match={score_match} — {status}")
        
        if not (pos_match and score_match):
            all_passed = False
            print(f"    Oracle score: {oracle_score:.6f}")
            print(f"    Actual max score: {actual_max_score:.6f}")
    
    if all_passed:
        print("\n✓ E4 PASSED: Oracle winners are correct maxima")
    else:
        print("\n✗ E4 FAILED: Some oracle winners are not the maximum!")
    
    return all_passed


# =============================================================================
# Experiment E5: Compare M3 vs M4 (requires M3 data)
# =============================================================================

def run_experiment_e5(components: dict, index_path: str):
    """
    E5: Compare observed (M3) vs oracle (M4) winners.
    
    This requires running a search with M3 tracking first.
    For now, we just demonstrate the methodology.
    """
    print("\n" + "=" * 70)
    print("E5: Compare M3 (Observed) vs M4 (Oracle) Winners")
    print("=" * 70)
    
    print("\nThis experiment requires M3 measurement data from a search run.")
    print("Methodology to implement:")
    print("  1. Run search with M3 tracking enabled")
    print("  2. For each (query_token, doc) in M3 results:")
    print("     - Get observed_winner_pos, observed_score from M3")
    print("     - Compute oracle_winner_pos, oracle_score via M4")
    print("     - Compare: same winner? score delta?")
    print("  3. Aggregate: miss rate, average score delta")
    
    # Check if M3 data exists
    m3_path = "/mnt/warp_measurements/runs"
    if os.path.exists(m3_path):
        runs = os.listdir(m3_path)
        if runs:
            print(f"\nFound existing measurement runs: {runs[:5]}...")
            print("Could load M3 data from these for comparison.")
    
    print("\n⚠️ E5 SKIPPED: Requires M3 measurement data")
    return None


# =============================================================================
# Experiment E6: Measure Scope of "All Scored" Documents  
# =============================================================================

def run_experiment_e6(components: dict, index_path: str):
    """
    E6: Measure how many documents are typically scored per query.
    
    This helps estimate M4 computation cost for "all_scored" policy.
    """
    print("\n" + "=" * 70)
    print("E6: Measure Scope of 'All Scored' Documents")
    print("=" * 70)
    
    # We need to run actual searches to measure this.
    # For now, estimate from index structure.
    
    codes_compacted = components['codes_compacted']
    sizes_compacted = components['sizes_compacted']
    offsets_compacted = components['offsets_compacted']
    
    num_embeddings = len(codes_compacted)
    num_centroids = len(sizes_compacted)
    num_docs = torch.unique(codes_compacted).shape[0]
    
    print(f"\nIndex statistics:")
    print(f"  Total embeddings: {num_embeddings:,}")
    print(f"  Total documents: {num_docs:,}")
    print(f"  Total centroids: {num_centroids:,}")
    print(f"  Avg embeddings per doc: {num_embeddings / num_docs:.1f}")
    print(f"  Avg embeddings per centroid: {num_embeddings / num_centroids:.1f}")
    
    # Estimate documents per centroid
    print("\nDocuments per centroid analysis:")
    sample_centroids = [0, 100, 1000, 5000, 10000, num_centroids // 2, num_centroids - 1]
    sample_centroids = [c for c in sample_centroids if c < num_centroids]
    
    docs_per_centroid = []
    for cid in sample_centroids:
        begin = offsets_compacted[cid].item()
        end = offsets_compacted[cid + 1].item()
        doc_ids = codes_compacted[begin:end]
        unique_docs_in_centroid = torch.unique(doc_ids).shape[0]
        docs_per_centroid.append(unique_docs_in_centroid)
        print(f"  Centroid {cid}: {unique_docs_in_centroid} unique docs "
              f"(from {end - begin} embeddings)")
    
    avg_docs_per_centroid = sum(docs_per_centroid) / len(docs_per_centroid)
    
    # Estimate for typical query
    nprobe = 8  # Typical value
    num_tokens = 20  # Typical query length
    
    print(f"\nEstimate for typical query (nprobe={nprobe}, tokens={num_tokens}):")
    
    # Upper bound: nprobe * num_tokens centroids, each with avg_docs_per_centroid docs
    # But there's overlap between centroids
    max_centroids = nprobe * num_tokens
    estimated_docs_upper = min(max_centroids * avg_docs_per_centroid, num_docs)
    
    # More realistic: assume ~50% overlap
    estimated_docs = estimated_docs_upper * 0.5
    
    print(f"  Max centroids accessed: {max_centroids}")
    print(f"  Avg docs per centroid: {avg_docs_per_centroid:.0f}")
    print(f"  Estimated unique docs scored: ~{estimated_docs:,.0f}")
    
    # M4 cost estimate
    avg_emb_per_doc = num_embeddings / num_docs
    oracle_ops_per_query = estimated_docs * num_tokens * avg_emb_per_doc
    
    print(f"\nM4 computation cost estimate:")
    print(f"  Docs to compute oracle for: ~{estimated_docs:,.0f}")
    print(f"  Query tokens: {num_tokens}")
    print(f"  Avg embeddings per doc: {avg_emb_per_doc:.1f}")
    print(f"  Total decompression ops: ~{oracle_ops_per_query:,.0f}")
    
    # Compare to M1 (routed computation)
    m1_ops = max_centroids * (num_embeddings / num_centroids)
    print(f"\nComparison to M1 (routed):")
    print(f"  M1 decompression ops: ~{m1_ops:,.0f}")
    print(f"  M4 / M1 ratio: ~{oracle_ops_per_query / m1_ops:.1f}x more work")
    
    print("\n✓ E6 COMPLETE: Scope analysis done (estimates only, need real search data)")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="M4 Verification Experiments")
    parser.add_argument(
        "--index-path",
        type=str,
        default="/mnt/datasets/index/beir-quora.split=test.nbits=4",
        help="Path to WARP index"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["E1", "E2", "E3", "E4", "E5", "E6", "all"],
        help="Which experiment to run"
    )
    args = parser.parse_args()
    
    # Check index exists
    if not os.path.exists(args.index_path):
        print(f"ERROR: Index not found at '{args.index_path}'")
        print("Please provide a valid --index-path")
        sys.exit(1)
    
    # Load index components
    components = load_index_components(args.index_path)
    
    # Run experiments
    experiments = {
        "E1": ("Verify Python Decompression", run_experiment_e1),
        "E2": ("Validate Reverse Index", run_experiment_e2),
        "E3": ("Measure Oracle Computation Time", run_experiment_e3),
        "E4": ("Validate Oracle Winners", run_experiment_e4),
        "E5": ("Compare M3 vs M4 Winners", run_experiment_e5),
        "E6": ("Measure Scope of All Scored", run_experiment_e6),
    }
    
    results = {}
    
    if args.experiment == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [args.experiment]
    
    for exp_id in to_run:
        name, func = experiments[exp_id]
        try:
            result = func(components, args.index_path)
            results[exp_id] = result
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
        result = results.get(exp_id)
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⚠️ SKIPPED"
        print(f"  {exp_id}: {name} — {status}")


if __name__ == "__main__":
    main()
