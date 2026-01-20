"""
Hypothesis H10.2: Characterizing "Useful Hubs" vs "Pure Noise Hubs"

Claim: Hub centroids can be classified into "oracle hubs" (contain golden oracle 
       embeddings) and "noise hubs" (no oracles). Oracle hubs contribute 
       disproportionately more to recall per unit of traffic.

Background:
    H10 found that hubs have lower yield (M3/M1). But 99.5% of M3 winners are for
    IRRELEVANT documents. TRUE yield (M3_relevant/M1) is ~200x smaller.
    
    Surprisingly, relevant_fraction (M3_relevant/M3) is POSITIVELY correlated 
    with hubness (ρ = +0.135). This suggests hubs are "relevant attractors" - 
    they concentrate computation in semantically important regions.

Key Questions:
    1. Can we distinguish oracle hubs from noise hubs using offline properties?
    2. What fraction of hub traffic goes to noise hubs (pure waste)?
    3. Can we selectively prune noise hubs without hurting recall?

Metrics:
    - has_oracle: Whether centroid contains any golden oracle embedding
    - oracle_count: Number of oracle embeddings in centroid  
    - true_yield: M3_relevant / M1 (only counting relevant doc wins)
    - relevant_fraction: M3_relevant / M3 (what % of wins matter)
    - hub_type_extended: oracle_hub, noise_hub, oracle_non_hub, noise_non_hub
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency
import torch

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.stats import (
    compute_group_comparison,
    bootstrap_comparison,
    bootstrap_ci
)
from hypothesis.viz import (
    plot_scatter_with_regression,
    plot_stratified_bars,
    plot_stratified_violin,
    plot_hypothesis_summary,
    save_figure
)


class H10_2_TrueYield(HypothesisTest):
    """
    Hypothesis 10.2: Characterizing "Useful Hubs" vs "Pure Noise Hubs"
    
    Extends H10 by incorporating relevance information from qrels.
    Tests whether hubs can be classified by their retrieval utility.
    """
    
    HYPOTHESIS_ID = "H10_2"
    HYPOTHESIS_NAME = "Oracle Hubs vs Noise Hubs"
    CLAIM = "Hub centroids divide into 'oracle hubs' (essential for recall) and 'noise hubs' (pure computational waste)"
    
    # Thresholds for hub sensitivity analysis (Issue 1 in 5.3.1 feedback)
    # Literature uses P95/P99 (HNSW), P75 is our baseline diagnostic cutoff
    HUB_SENSITIVITY_THRESHOLDS = [0.50, 0.75, 0.90, 0.95, 0.99]
    
    def setup(self):
        """Load data including golden metrics and M3 with relevance."""
        from hypothesis.configs import ensure_output_dirs
        from hypothesis.data.standardized_tables import ClusterFrameBuilder
        
        ensure_output_dirs(self.config)
        
        # Only build cluster_frame (H10.2 doesn't need miss_attribution_frame)
        cluster_builder = ClusterFrameBuilder(self.config)
        self.cluster_frame = cluster_builder.build()
        self.query_frame = None  # Not needed for H10.2
        
        print(f"Loaded cluster_frame: {self.cluster_frame.shape}")
        
        # Load additional data for true yield computation
        self._load_oracle_data()
        self._load_m3_with_relevance()
        self._compute_extended_metrics()
    
    def _load_oracle_data(self):
        """Load M4R (oracle embeddings) and compute per-centroid oracle stats."""
        # Find the golden metrics path
        base_path = Path(self.config.paths.run_dir)
        m4r_path = base_path / "golden_metrics_v2" / "M4R.parquet"
        
        if not m4r_path.exists():
            # Try alternative path structure
            for run_dir in base_path.parent.glob("*/golden_metrics_v2"):
                alt_path = run_dir / "M4R.parquet"
                if alt_path.exists():
                    m4r_path = alt_path
                    break
        
        if not m4r_path.exists():
            raise FileNotFoundError(f"M4R.parquet not found. Tried: {m4r_path}")
        
        self.m4r = pd.read_parquet(m4r_path)
        print(f"Loaded M4R: {len(self.m4r):,} oracle embeddings")
        
        # Aggregate by centroid
        self.oracle_by_centroid = self.m4r.groupby('oracle_centroid_id').agg({
            'oracle_is_accessible': ['sum', 'count', 'mean'],
            'query_id': 'nunique',
            'doc_id': 'nunique'
        }).reset_index()
        self.oracle_by_centroid.columns = [
            'centroid_id', 'accessible_oracles', 'total_oracles', 
            'oracle_accessibility_rate', 'unique_queries', 'unique_golden_docs'
        ]
    
    def _load_m3_with_relevance(self):
        """Load M3 and flag relevant vs irrelevant winners."""
        base_path = Path(self.config.paths.run_dir)
        
        # Load M3
        m3_path = base_path / "tier_b" / "M3_observed_winners.parquet"
        if not m3_path.exists():
            for run_dir in base_path.parent.glob("*/tier_b"):
                alt_path = run_dir / "M3_observed_winners.parquet"
                if alt_path.exists():
                    m3_path = alt_path
                    break
        
        if not m3_path.exists():
            raise FileNotFoundError(f"M3_observed_winners.parquet not found")
        
        self.m3 = pd.read_parquet(m3_path)
        print(f"Loaded M3: {len(self.m3):,} MaxSim winners")
        
        # Load routing_status (qrels)
        routing_path = base_path / "golden_metrics_v2" / "routing_status.parquet"
        if not routing_path.exists():
            for run_dir in base_path.parent.glob("*/golden_metrics_v2"):
                alt_path = run_dir / "routing_status.parquet"
                if alt_path.exists():
                    routing_path = alt_path
                    break
        
        if not routing_path.exists():
            raise FileNotFoundError(f"routing_status.parquet not found")
        
        routing = pd.read_parquet(routing_path)
        self.golden_pairs = set(zip(routing['query_id'], routing['doc_id']))
        print(f"Loaded {len(self.golden_pairs):,} golden (query, doc) pairs")
        
        # Flag relevance
        self.m3['is_relevant'] = self.m3.apply(
            lambda r: (r['query_id'], r['doc_id']) in self.golden_pairs, axis=1
        )
        
        # Load embedding-to-centroid mapping
        index_path = Path(self.config.paths.index_path)
        emb_to_centroid_path = index_path / "embedding_to_centroid.pt"
        
        if emb_to_centroid_path.exists():
            emb_to_centroid = torch.load(emb_to_centroid_path)
            self.m3['winner_centroid'] = emb_to_centroid[self.m3['winner_embedding_pos'].values].numpy()
            print(f"Mapped M3 winners to centroids")
        else:
            raise FileNotFoundError(f"embedding_to_centroid.pt not found at {emb_to_centroid_path}")
    
    def _compute_extended_metrics(self):
        """Compute per-centroid true yield and classify hub types."""
        # Aggregate M3 by centroid
        m3_by_centroid = self.m3.groupby('winner_centroid').agg({
            'query_id': 'count',      # M3 total
            'is_relevant': 'sum'      # M3_relevant
        }).reset_index()
        m3_by_centroid.columns = ['centroid_id', 'M3_total', 'M3_relevant']
        
        # Merge everything into cluster_frame
        df = self.cluster_frame.copy()
        
        # Add oracle data
        df = df.merge(self.oracle_by_centroid, on='centroid_id', how='left')
        df['total_oracles'] = df['total_oracles'].fillna(0)
        df['accessible_oracles'] = df['accessible_oracles'].fillna(0)
        df['has_oracle'] = df['total_oracles'] > 0
        
        # Add M3 relevance data
        df = df.merge(m3_by_centroid, on='centroid_id', how='left')
        df['M3_total'] = df['M3_total'].fillna(0)
        df['M3_relevant'] = df['M3_relevant'].fillna(0)
        
        # Compute TRUE YIELD metrics
        df['true_yield'] = df['M3_relevant'] / df['m1_total_sims']
        df['relevant_fraction'] = df['M3_relevant'] / df['M3_total']
        df.loc[df['M3_total'] == 0, 'relevant_fraction'] = np.nan
        
        # Classify extended hub types
        # First, identify hubs (top 25% by sel_freq)
        if 'sel_freq' in df.columns:
            hub_threshold = df['sel_freq'].quantile(0.75)
            df['is_hub'] = df['sel_freq'] >= hub_threshold
        else:
            df['is_hub'] = False
        
        # Four-way classification
        conditions = [
            (df['is_hub'] & df['has_oracle']),
            (df['is_hub'] & ~df['has_oracle']),
            (~df['is_hub'] & df['has_oracle']),
            (~df['is_hub'] & ~df['has_oracle'])
        ]
        choices = ['oracle_hub', 'noise_hub', 'oracle_non_hub', 'noise_non_hub']
        df['hub_type_extended'] = np.select(conditions, choices, default='unknown')
        
        self.extended_frame = df
        
        # Compute hub sensitivity analysis across multiple thresholds
        self.hub_sensitivity_df = self._compute_hub_sensitivity_analysis(df)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXTENDED HUB CLASSIFICATION")
        print("=" * 60)
        type_counts = df['hub_type_extended'].value_counts()
        for ht, count in type_counts.items():
            subset = df[df['hub_type_extended'] == ht]
            print(f"  {ht}: {count:,} centroids, "
                  f"mean sel_freq={subset['sel_freq'].mean():.1f}, "
                  f"mean oracles={subset['total_oracles'].mean():.1f}")
        
        # Print sensitivity analysis summary
        print("\n" + "-" * 60)
        print("HUB THRESHOLD SENSITIVITY ANALYSIS")
        print("-" * 60)
        print(self.hub_sensitivity_df.to_string(index=False))
    
    def _compute_hub_sensitivity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute hub classification metrics across multiple percentile thresholds.
        
        This addresses feedback point 2 in Section 5.3.1: the P75 threshold is
        arbitrary, so we test robustness across P50-P99 following literature
        conventions (HNSW uses P95/P99, H₂O uses top 20%).
        
        Returns:
            DataFrame with columns: threshold, n_hubs, pct_with_oracle,
            traffic_share, oracle_hub_count, noise_hub_count,
            oracle_hub_traffic_share, noise_hub_traffic_share, rho_traffic_oracle
        """
        if 'sel_freq' not in df.columns:
            return pd.DataFrame()
        
        results = []
        total_traffic = df['sel_freq'].sum()
        
        for p in self.HUB_SENSITIVITY_THRESHOLDS:
            hub_threshold = df['sel_freq'].quantile(p)
            is_hub = df['sel_freq'] >= hub_threshold
            
            hubs = df[is_hub]
            n_hubs = len(hubs)
            
            if n_hubs == 0:
                continue
            
            # Core metrics
            pct_with_oracle = hubs['has_oracle'].mean() * 100
            hub_traffic = hubs['sel_freq'].sum()
            traffic_share = (hub_traffic / total_traffic) * 100
            
            # Oracle vs noise hub breakdown
            oracle_hubs = hubs[hubs['has_oracle']]
            noise_hubs = hubs[~hubs['has_oracle']]
            
            oracle_hub_count = len(oracle_hubs)
            noise_hub_count = len(noise_hubs)
            oracle_hub_traffic_share = (oracle_hubs['sel_freq'].sum() / total_traffic) * 100
            noise_hub_traffic_share = (noise_hubs['sel_freq'].sum() / total_traffic) * 100
            
            # Per-threshold correlation ρ(traffic, oracle) among hubs
            if n_hubs > 10:
                rho_traffic_oracle, _ = spearmanr(
                    hubs['sel_freq'],
                    hubs['has_oracle'].astype(int)
                )
            else:
                rho_traffic_oracle = np.nan
            
            results.append({
                'threshold': f'P{int(p * 100)}',
                'percentile': p,
                'n_hubs': n_hubs,
                'pct_with_oracle': pct_with_oracle,
                'traffic_share': traffic_share,
                'oracle_hub_count': oracle_hub_count,
                'noise_hub_count': noise_hub_count,
                'oracle_hub_traffic_share': oracle_hub_traffic_share,
                'noise_hub_traffic_share': noise_hub_traffic_share,
                'rho_traffic_oracle': rho_traffic_oracle
            })
        
        return pd.DataFrame(results)
    
    def compute_null_baseline_correlation(
        self,
        df: pd.DataFrame,
        n_permutations: int = 1000,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Permutation test for traffic-golden correlation null baseline.
        
        This addresses feedback point 4 in Section 5.3.1: the ρ = +0.332 correlation
        between traffic and golden-token presence could be confounded by semantic
        alignment inherent to any nearest-neighbor routing scheme.
        
        Method:
            Shuffle sel_freq values across centroids to break any systematic
            relationship, then recompute correlation with fixed has_oracle.
            Repeat N times to build null distribution.
            
        Statistical Notes:
            - P-value uses (k+1)/(n+1) correction per Phipson & Smyth (2010)
              "Permutation P-values Should Never Be Zero" to ensure valid p-values
            - Effect attribution uses |null_mean| < 2*null_std as principled
              threshold for "null is essentially zero"
            - For publication, consider n_permutations=10000 for more precision
        
        Interpretation:
            - If null ρ ≈ 0 and observed ρ = 0.332: correlation is real, WARP
              routes to golden-containing centroids specifically
            - If null ρ > 0 (e.g., 0.2): some correlation is geometric, only
              the excess (0.332 - 0.2) is attributable to WARP routing
            - If null ρ ≈ observed ρ: correlation is entirely geometric
        
        Args:
            df: DataFrame with 'sel_freq' and 'has_oracle' columns
            n_permutations: Number of permutations for null distribution
            random_state: Random seed for reproducibility
            
        Returns:
            Dict with observed_rho, null distribution stats, and p-value
        """
        # Filter to valid rows
        valid = df[df['sel_freq'].notna() & df['has_oracle'].notna()].copy()
        
        if len(valid) < 100:
            return {
                'observed_rho': np.nan,
                'null_mean': np.nan,
                'null_std': np.nan,
                'null_ci_lower': np.nan,
                'null_ci_upper': np.nan,
                'permutation_p_value': np.nan,
                'n_permutations': 0,
                'effect_attribution': 'insufficient_data'
            }
        
        # Check for degenerate cases (all same value)
        oracle_unique = valid['has_oracle'].nunique()
        if oracle_unique < 2:
            return {
                'observed_rho': np.nan,
                'null_mean': np.nan,
                'null_std': np.nan,
                'null_ci_lower': np.nan,
                'null_ci_upper': np.nan,
                'permutation_p_value': np.nan,
                'n_permutations': 0,
                'effect_attribution': 'degenerate_data (no variance in has_oracle)'
            }
        
        # Observed correlation
        observed_rho, observed_p = spearmanr(
            valid['sel_freq'],
            valid['has_oracle'].astype(int)
        )
        
        # Also compute for relevant_fraction (continuous variable)
        mask_frac = valid['relevant_fraction'].notna()
        if mask_frac.sum() > 100:
            observed_rho_frac, _ = spearmanr(
                valid.loc[mask_frac, 'sel_freq'],
                valid.loc[mask_frac, 'relevant_fraction']
            )
        else:
            observed_rho_frac = np.nan
        
        # Build null distribution via permutation
        rng = np.random.default_rng(random_state)
        null_rhos = np.zeros(n_permutations)
        null_rhos_frac = np.zeros(n_permutations)
        
        has_oracle_values = valid['has_oracle'].astype(int).values
        sel_freq_values = valid['sel_freq'].values
        
        if mask_frac.sum() > 100:
            relevant_frac_values = valid.loc[mask_frac, 'relevant_fraction'].values
            sel_freq_frac_values = valid.loc[mask_frac, 'sel_freq'].values
        
        for i in range(n_permutations):
            # Shuffle traffic assignments (break centroid-traffic link)
            shuffled_traffic = rng.permutation(sel_freq_values)
            rho, _ = spearmanr(shuffled_traffic, has_oracle_values)
            # Handle potential NaN from spearmanr (shouldn't happen with our checks, but defensive)
            null_rhos[i] = rho if not np.isnan(rho) else 0.0
            
            # Also for relevant_fraction
            if mask_frac.sum() > 100:
                shuffled_frac = rng.permutation(sel_freq_frac_values)
                rho_frac, _ = spearmanr(shuffled_frac, relevant_frac_values)
                null_rhos_frac[i] = rho_frac if not np.isnan(rho_frac) else 0.0
        
        # Compute statistics
        null_mean = null_rhos.mean()
        null_std = null_rhos.std()
        null_ci_lower = np.percentile(null_rhos, 2.5)
        null_ci_upper = np.percentile(null_rhos, 97.5)
        
        # Two-tailed p-value with continuity correction (Phipson & Smyth, 2010)
        # Formula: (k + 1) / (n + 1) where k = count of |null| >= |observed|
        # This ensures p-value is never exactly 0 and is a valid probability
        k_extreme = (np.abs(null_rhos) >= np.abs(observed_rho)).sum()
        permutation_p_value = (k_extreme + 1) / (n_permutations + 1)
        
        # Effect attribution interpretation
        # Use principled threshold: null is "essentially zero" if |mean| < 2*std
        # (i.e., zero is within the typical variation of the null distribution)
        null_is_near_zero = abs(null_mean) < 2 * null_std
        
        if permutation_p_value < 0.05:
            if null_is_near_zero:
                effect_attribution = 'routing_specific'
            else:
                excess_rho = observed_rho - null_mean
                effect_attribution = f'partially_geometric (excess={excess_rho:.3f})'
        else:
            effect_attribution = 'entirely_geometric'
        
        results = {
            # Has oracle (binary) correlation
            'observed_rho': observed_rho,
            'observed_p': observed_p,
            'null_mean': null_mean,
            'null_std': null_std,
            'null_ci_lower': null_ci_lower,
            'null_ci_upper': null_ci_upper,
            'permutation_p_value': permutation_p_value,
            'n_permutations': n_permutations,
            'effect_attribution': effect_attribution,
            'null_distribution': null_rhos.tolist(),  # Store for visualization
        }
        
        # Add relevant_fraction results if available
        if mask_frac.sum() > 100:
            k_extreme_frac = (np.abs(null_rhos_frac) >= np.abs(observed_rho_frac)).sum()
            results.update({
                'observed_rho_frac': observed_rho_frac,
                'null_mean_frac': null_rhos_frac.mean(),
                'null_std_frac': null_rhos_frac.std(),
                'null_ci_lower_frac': np.percentile(null_rhos_frac, 2.5),
                'null_ci_upper_frac': np.percentile(null_rhos_frac, 97.5),
                'permutation_p_value_frac': (k_extreme_frac + 1) / (n_permutations + 1),
            })
        
        return results
    
    def analyze(self) -> HypothesisResult:
        """Run the H10.2 analysis."""
        df = self.extended_frame
        
        # Filter to valid data
        valid = df[(df['m1_total_sims'] > 0) & (df['sel_freq'] > 0)].copy()
        
        if len(valid) < 100:
            return self._empty_result("Insufficient data")
        
        # =====================================================================
        # TEST 1: Correlations with hubness
        # =====================================================================
        
        # H10's yield vs hubness
        rho_h10, p_h10 = spearmanr(valid['sel_freq'], valid['yield'])
        
        # TRUE yield vs hubness
        rho_true, p_true = spearmanr(valid['sel_freq'], valid['true_yield'])
        
        # Relevant fraction vs hubness (the surprising finding)
        mask = valid['relevant_fraction'].notna()
        rho_frac, p_frac = spearmanr(
            valid.loc[mask, 'sel_freq'],
            valid.loc[mask, 'relevant_fraction']
        )
        
        # =====================================================================
        # TEST 1b: Permutation test null baseline (Section 5.3.1 feedback)
        # =====================================================================
        # 
        # The observed correlation could be confounded by semantic alignment
        # inherent to any nearest-neighbor scheme. This permutation test
        # establishes what ρ would be under random centroid selection.
        #
        print("\nRunning permutation test for traffic-golden correlation...")
        permutation_results = self.compute_null_baseline_correlation(valid, n_permutations=1000)
        
        # Store permutation results (excluding the full distribution for JSON)
        permutation_stats = {k: v for k, v in permutation_results.items() 
                            if k != 'null_distribution'}
        self._permutation_results = permutation_results  # Keep full results for visualization
        
        print(f"  Observed ρ(traffic, has_oracle): {permutation_results['observed_rho']:.4f}")
        print(f"  Null distribution mean: {permutation_results['null_mean']:.4f} ± {permutation_results['null_std']:.4f}")
        print(f"  Null 95% CI: [{permutation_results['null_ci_lower']:.4f}, {permutation_results['null_ci_upper']:.4f}]")
        print(f"  Permutation p-value: {permutation_results['permutation_p_value']:.4f}")
        print(f"  Effect attribution: {permutation_results['effect_attribution']}")
        
        # =====================================================================
        # TEST 2: Oracle Hub vs Noise Hub comparison
        # =====================================================================
        
        oracle_hubs = valid[valid['hub_type_extended'] == 'oracle_hub']
        noise_hubs = valid[valid['hub_type_extended'] == 'noise_hub']
        
        comparison_stats = {}
        if len(oracle_hubs) > 10 and len(noise_hubs) > 10:
            # Compare properties
            for prop in ['n_tokens', 'dispersion', 'n_docs', 'sel_freq', 'yield']:
                if prop in valid.columns:
                    stat, pval = mannwhitneyu(
                        oracle_hubs[prop].dropna(),
                        noise_hubs[prop].dropna(),
                        alternative='two-sided'
                    )
                    comparison_stats[f'{prop}_U'] = stat
                    comparison_stats[f'{prop}_p'] = pval
                    comparison_stats[f'{prop}_oracle_mean'] = oracle_hubs[prop].mean()
                    comparison_stats[f'{prop}_noise_mean'] = noise_hubs[prop].mean()
        
        # =====================================================================
        # TEST 3: Traffic distribution analysis
        # =====================================================================
        
        total_traffic = valid['sel_freq'].sum()
        hub_traffic = valid[valid['is_hub']]['sel_freq'].sum()
        oracle_hub_traffic = oracle_hubs['sel_freq'].sum() if len(oracle_hubs) > 0 else 0
        noise_hub_traffic = noise_hubs['sel_freq'].sum() if len(noise_hubs) > 0 else 0
        
        traffic_stats = {
            'total_traffic': total_traffic,
            'hub_traffic_share': hub_traffic / total_traffic,
            'oracle_hub_traffic_share': oracle_hub_traffic / total_traffic,
            'noise_hub_traffic_share': noise_hub_traffic / total_traffic,
            'noise_hub_waste_potential': noise_hub_traffic / hub_traffic if hub_traffic > 0 else 0
        }
        
        # =====================================================================
        # TEST 4: True yield by hub type
        # =====================================================================
        
        type_stats = {}
        for ht in ['oracle_hub', 'noise_hub', 'oracle_non_hub', 'noise_non_hub']:
            subset = valid[valid['hub_type_extended'] == ht]
            if len(subset) > 0:
                type_stats[f'{ht}_count'] = len(subset)
                type_stats[f'{ht}_h10_yield'] = subset['yield'].mean()
                type_stats[f'{ht}_true_yield'] = subset['true_yield'].mean()
                type_stats[f'{ht}_relevant_frac'] = subset['relevant_fraction'].mean()
                type_stats[f'{ht}_oracle_count'] = subset['total_oracles'].sum()
                type_stats[f'{ht}_traffic'] = subset['sel_freq'].sum()
        
        # =====================================================================
        # TEST 5: Predictability of hub type from offline properties
        # =====================================================================
        
        # Can we predict has_oracle from offline properties?
        hubs_only = valid[valid['is_hub']].copy()
        if len(hubs_only) > 50:
            # Correlation of offline properties with has_oracle
            for prop in ['n_tokens', 'dispersion', 'n_docs', 'gini_coefficient']:
                if prop in hubs_only.columns:
                    rho, p = spearmanr(
                        hubs_only[prop].dropna(),
                        hubs_only.loc[hubs_only[prop].notna(), 'has_oracle'].astype(int)
                    )
                    comparison_stats[f'has_oracle_vs_{prop}_rho'] = rho
                    comparison_stats[f'has_oracle_vs_{prop}_p'] = p
        
        # =====================================================================
        # Determine if hypothesis is supported
        # =====================================================================
        
        # H10.2 is supported if:
        # 1. Oracle hubs and noise hubs have significantly different properties
        # 2. Noise hubs contribute significant traffic (waste potential)
        # 3. Oracle hubs have higher true_yield than noise hubs
        
        oracle_hub_true_yield = type_stats.get('oracle_hub_true_yield', 0)
        noise_hub_true_yield = type_stats.get('noise_hub_true_yield', 0)
        
        # Noise hubs should have true_yield ≈ 0 (no oracles = no relevant wins possible directly)
        # Oracle hubs should have true_yield > 0
        supported = (
            oracle_hub_true_yield > noise_hub_true_yield and
            traffic_stats['noise_hub_waste_potential'] > 0.1  # >10% of hub traffic is waste
        )
        
        # Effect size: difference in true_yield between oracle and noise hubs
        effect_size = oracle_hub_true_yield - noise_hub_true_yield
        
        # Compile all statistics
        stats = {
            # Correlation tests
            'rho_hubness_h10yield': rho_h10,
            'p_hubness_h10yield': p_h10,
            'rho_hubness_trueyield': rho_true,
            'p_hubness_trueyield': p_true,
            'rho_hubness_relevantfrac': rho_frac,
            'p_hubness_relevantfrac': p_frac,
            
            # Summary
            'n_oracle_hubs': len(oracle_hubs),
            'n_noise_hubs': len(noise_hubs),
        }
        stats.update(traffic_stats)
        stats.update(type_stats)
        stats.update(comparison_stats)
        
        # Add hub sensitivity analysis results
        if hasattr(self, 'hub_sensitivity_df') and not self.hub_sensitivity_df.empty:
            stats['hub_sensitivity'] = self.hub_sensitivity_df.to_dict('records')
        
        # Add permutation test results
        stats.update({f'permutation_{k}': v for k, v in permutation_stats.items()})
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=effect_size,
            effect_size_ci=(np.nan, np.nan),  # Bootstrap CI not implemented
            p_value=p_frac,  # Use relevant_fraction correlation p-value
            statistics=stats,
            config_name=self.config.name,
            n_observations=len(valid),
            timestamp=datetime.now().isoformat()
        )
    
    def visualize(self):
        """Generate visualizations for H10.2."""
        df = self.extended_frame
        valid = df[(df['m1_total_sims'] > 0) & (df['sel_freq'] > 0)].copy()
        
        import matplotlib.pyplot as plt
        
        # Figure 1: Four-panel comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Hub type distribution
        ax1 = axes[0, 0]
        type_counts = valid['hub_type_extended'].value_counts()
        colors = {'oracle_hub': '#2E8B57', 'noise_hub': '#FF6B6B', 
                  'oracle_non_hub': '#90EE90', 'noise_non_hub': '#FFB3BA'}
        bars = ax1.bar(type_counts.index, type_counts.values, 
                       color=[colors.get(t, 'gray') for t in type_counts.index])
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('(a) Centroid Classification', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel 2: Traffic distribution by type
        ax2 = axes[0, 1]
        traffic_by_type = valid.groupby('hub_type_extended')['sel_freq'].sum()
        ax2.pie(traffic_by_type, labels=traffic_by_type.index, autopct='%1.1f%%',
                colors=[colors.get(t, 'gray') for t in traffic_by_type.index])
        ax2.set_title('(b) Traffic Distribution by Type', fontsize=12, fontweight='bold')
        
        # Panel 3: True yield by type
        ax3 = axes[1, 0]
        type_order = ['oracle_hub', 'noise_hub', 'oracle_non_hub', 'noise_non_hub']
        true_yields = [valid[valid['hub_type_extended'] == t]['true_yield'].mean() * 100 
                       for t in type_order if t in valid['hub_type_extended'].values]
        type_labels = [t for t in type_order if t in valid['hub_type_extended'].values]
        bars3 = ax3.bar(type_labels, true_yields, 
                        color=[colors.get(t, 'gray') for t in type_labels])
        ax3.set_ylabel('TRUE Yield (%)', fontsize=11)
        ax3.set_title('(c) TRUE Yield by Centroid Type', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Panel 4: Properties comparison (oracle vs noise hubs)
        ax4 = axes[1, 1]
        oracle_hubs = valid[valid['hub_type_extended'] == 'oracle_hub']
        noise_hubs = valid[valid['hub_type_extended'] == 'noise_hub']
        
        if len(oracle_hubs) > 0 and len(noise_hubs) > 0:
            props = ['n_tokens', 'dispersion', 'n_docs']
            x = np.arange(len(props))
            width = 0.35
            
            # Normalize for visualization
            oracle_vals = [oracle_hubs[p].mean() for p in props]
            noise_vals = [noise_hubs[p].mean() for p in props]
            
            ax4.bar(x - width/2, oracle_vals, width, label='Oracle Hubs', color='#2E8B57')
            ax4.bar(x + width/2, noise_vals, width, label='Noise Hubs', color='#FF6B6B')
            ax4.set_xticks(x)
            ax4.set_xticklabels(props)
            ax4.legend()
        ax4.set_title('(d) Properties: Oracle vs Noise Hubs', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_figure(fig, self.output_dir, f'{self.HYPOTHESIS_ID}_hub_types')
        
        # Figure 2: Scatter plots
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        
        # Scatter 1: sel_freq vs true_yield colored by hub type
        ax = axes2[0]
        for ht in ['oracle_hub', 'noise_hub', 'oracle_non_hub', 'noise_non_hub']:
            subset = valid[valid['hub_type_extended'] == ht]
            ax.scatter(subset['sel_freq'], subset['true_yield'] * 100,
                      alpha=0.3, s=10, label=ht, c=colors.get(ht, 'gray'))
        ax.set_xlabel('Selection Frequency (hubness)', fontsize=11)
        ax.set_ylabel('TRUE Yield (%)', fontsize=11)
        ax.set_title('(a) Hubness vs TRUE Yield', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        
        # Scatter 2: sel_freq vs relevant_fraction
        ax = axes2[1]
        mask = valid['relevant_fraction'].notna()
        for ht in ['oracle_hub', 'noise_hub', 'oracle_non_hub', 'noise_non_hub']:
            subset = valid[mask & (valid['hub_type_extended'] == ht)]
            ax.scatter(subset['sel_freq'], subset['relevant_fraction'] * 100,
                      alpha=0.3, s=10, label=ht, c=colors.get(ht, 'gray'))
        ax.set_xlabel('Selection Frequency (hubness)', fontsize=11)
        ax.set_ylabel('Relevant Fraction (%)', fontsize=11)
        ax.set_title('(b) Hubness vs Relevant Fraction', fontsize=12, fontweight='bold')
        
        # Scatter 3: dispersion vs has_oracle (among hubs)
        ax = axes2[2]
        hubs = valid[valid['is_hub']]
        ax.scatter(hubs['dispersion'], hubs['has_oracle'].astype(int) + np.random.uniform(-0.1, 0.1, len(hubs)),
                  alpha=0.3, s=10, c=hubs['sel_freq'], cmap='YlOrRd')
        ax.set_xlabel('Dispersion', fontsize=11)
        ax.set_ylabel('Has Oracle (jittered)', fontsize=11)
        ax.set_title('(c) Can Dispersion Predict Oracle Presence?', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_figure(fig2, self.output_dir, f'{self.HYPOTHESIS_ID}_scatter')
        
        # Figure 3: Permutation test null distribution
        if hasattr(self, '_permutation_results') and 'null_distribution' in self._permutation_results:
            perm = self._permutation_results
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
            
            # Panel 1: Null distribution for has_oracle correlation
            ax = axes3[0]
            null_dist = np.array(perm['null_distribution'])
            ax.hist(null_dist, bins=50, alpha=0.7, color='#4A90D9', edgecolor='white',
                   label=f'Null distribution (n={perm["n_permutations"]:,})')
            
            # Mark observed value
            ax.axvline(perm['observed_rho'], color='#E74C3C', linewidth=2.5, 
                       label=f'Observed ρ = {perm["observed_rho"]:.3f}')
            
            # Mark null 95% CI
            ax.axvline(perm['null_ci_lower'], color='#95A5A6', linestyle='--', linewidth=1.5,
                       label=f'Null 95% CI: [{perm["null_ci_lower"]:.3f}, {perm["null_ci_upper"]:.3f}]')
            ax.axvline(perm['null_ci_upper'], color='#95A5A6', linestyle='--', linewidth=1.5)
            
            # Mark null mean
            ax.axvline(perm['null_mean'], color='#2ECC71', linewidth=2, linestyle=':',
                       label=f'Null mean = {perm["null_mean"]:.3f}')
            
            ax.set_xlabel('Spearman ρ (traffic vs has_oracle)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('(a) Permutation Test: Traffic-Oracle Correlation\n'
                        f'p = {perm["permutation_p_value"]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            
            # Add effect attribution annotation
            ax.annotate(f'Effect: {perm["effect_attribution"]}',
                       xy=(0.98, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Panel 2: Comparison summary (bar chart)
            ax = axes3[1]
            categories = ['Observed\nρ', 'Null Mean', 'Null 95% CI\nUpper']
            values = [perm['observed_rho'], perm['null_mean'], perm['null_ci_upper']]
            colors_bar = ['#E74C3C', '#2ECC71', '#95A5A6']
            
            bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.2)
            
            # Add error bar for null distribution
            ax.errorbar(1, perm['null_mean'], yerr=perm['null_std'] * 2, 
                       fmt='none', color='black', capsize=5, capthick=2)
            
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
            ax.set_ylabel('Spearman ρ', fontsize=11)
            ax.set_title('(b) Observed vs Null Baseline Comparison', fontsize=12, fontweight='bold')
            
            # Annotate the excess correlation
            excess = perm['observed_rho'] - perm['null_mean']
            ax.annotate(f'Excess ρ = {excess:.3f}\n(attributable to WARP)',
                       xy=(0.5, max(values) * 0.7), xycoords=('axes fraction', 'data'),
                       ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.tight_layout()
            save_figure(fig3, self.output_dir, f'{self.HYPOTHESIS_ID}_permutation_test')
        
        plt.close('all')
    
    def _format_sensitivity_table(self) -> str:
        """Format sensitivity analysis as text table for report."""
        if not hasattr(self, 'hub_sensitivity_df') or self.hub_sensitivity_df.empty:
            return "  No sensitivity data available"
        
        lines = ["  Threshold | # Hubs | % with Oracle | Traffic Share | ρ(traffic,oracle)"]
        lines.append("  " + "-" * 70)
        for _, row in self.hub_sensitivity_df.iterrows():
            rho_str = f"{row['rho_traffic_oracle']:>+.3f}" if not np.isnan(row['rho_traffic_oracle']) else "     N/A"
            lines.append(
                f"  {row['threshold']:>9} | {row['n_hubs']:>6,} | "
                f"{row['pct_with_oracle']:>12.1f}% | {row['traffic_share']:>12.1f}% | {rho_str}"
            )
        return "\n".join(lines)
    
    def report(self) -> str:
        """Generate text report."""
        result = self.results if hasattr(self, 'results') else self.result
        stats = result.statistics
        
        report = f"""
{'='*70}
HYPOTHESIS {self.HYPOTHESIS_ID}: {self.HYPOTHESIS_NAME}
{'='*70}

CLAIM: {self.CLAIM}

RESULT: {'✓ SUPPORTED' if result.supported else '✗ NOT SUPPORTED'}

SUMMARY:
--------
Centroids classified:
  - Oracle Hubs: {stats.get('n_oracle_hubs', 0):,}
  - Noise Hubs: {stats.get('n_noise_hubs', 0):,}

Traffic Distribution:
  - Hub traffic share: {stats.get('hub_traffic_share', 0):.1%}
  - Oracle hub traffic: {stats.get('oracle_hub_traffic_share', 0):.1%}
  - Noise hub traffic: {stats.get('noise_hub_traffic_share', 0):.1%}
  - Waste potential: {stats.get('noise_hub_waste_potential', 0):.1%} of hub traffic

TRUE Yield by Type:
  - Oracle Hubs: {stats.get('oracle_hub_true_yield', 0)*100:.4f}%
  - Noise Hubs: {stats.get('noise_hub_true_yield', 0)*100:.4f}%
  - Oracle Non-Hubs: {stats.get('oracle_non_hub_true_yield', 0)*100:.4f}%
  - Noise Non-Hubs: {stats.get('noise_non_hub_true_yield', 0)*100:.4f}%

CORRELATION TESTS:
  Hubness vs H10 Yield:      ρ = {stats.get('rho_hubness_h10yield', np.nan):.4f}
  Hubness vs TRUE Yield:     ρ = {stats.get('rho_hubness_trueyield', np.nan):.4f}
  Hubness vs Relevant Frac:  ρ = {stats.get('rho_hubness_relevantfrac', np.nan):.4f}

INTERPRETATION:
  The positive correlation on relevant_fraction ({stats.get('rho_hubness_relevantfrac', 0):.3f})
  confirms that hubs are "relevant attractors" - when they win MaxSim,
  they're more likely to win for relevant documents.
  
  However, noise hubs ({stats.get('noise_hub_waste_potential', 0):.1%} of hub traffic) 
  represent pure computational waste with zero direct contribution to recall.

HUB THRESHOLD SENSITIVITY ANALYSIS:
{self._format_sensitivity_table()}

  Note: Results are {'ROBUST' if self._check_sensitivity_robustness() else 'SENSITIVE'} across thresholds.

PERMUTATION TEST NULL BASELINE (Section 5.3.1 Feedback Point 4):
{self._format_permutation_results()}

{'='*70}
"""
        return report
    
    def _format_permutation_results(self) -> str:
        """Format permutation test results for the report."""
        result = self.results if hasattr(self, 'results') else self.result
        stats = result.statistics
        
        # Check if permutation results exist
        if 'permutation_observed_rho' not in stats:
            return "  Permutation test not run"
        
        lines = [
            f"  Question: Is the traffic-golden correlation (ρ = {stats.get('permutation_observed_rho', np.nan):.3f})",
            f"            due to WARP-specific routing or purely geometric?",
            f"",
            f"  Method: Shuffled traffic assignments {stats.get('permutation_n_permutations', 0):,} times",
            f"          to build null distribution under random routing.",
            f"",
            f"  Results:",
            f"    Observed ρ(traffic, has_oracle):  {stats.get('permutation_observed_rho', np.nan):>+.4f}",
            f"    Null distribution mean:           {stats.get('permutation_null_mean', np.nan):>+.4f} ± {stats.get('permutation_null_std', np.nan):.4f}",
            f"    Null 95% CI:                      [{stats.get('permutation_null_ci_lower', np.nan):>+.4f}, {stats.get('permutation_null_ci_upper', np.nan):>+.4f}]",
            f"    Permutation p-value:              {stats.get('permutation_permutation_p_value', np.nan):.4f}",
            f"",
            f"  Interpretation:",
        ]
        
        # Add interpretation based on effect attribution
        effect = stats.get('permutation_effect_attribution', 'unknown')
        if effect == 'routing_specific':
            lines.extend([
                f"    The correlation is STATISTICALLY SIGNIFICANT (p < 0.05) and the null",
                f"    distribution is centered near zero. This suggests the observed correlation",
                f"    is attributable to WARP's routing behavior, not geometric coincidence.",
            ])
        elif 'partially_geometric' in effect:
            excess = stats.get('permutation_observed_rho', 0) - stats.get('permutation_null_mean', 0)
            lines.extend([
                f"    The correlation is PARTIALLY GEOMETRIC. The null baseline shows",
                f"    some correlation exists under random routing (mean = {stats.get('permutation_null_mean', 0):.3f}).",
                f"    Only the excess (Δρ = {excess:.3f}) is attributable to WARP's routing.",
            ])
        elif effect == 'entirely_geometric':
            lines.extend([
                f"    The correlation is NOT SIGNIFICANT (p >= 0.05). The observed value",
                f"    falls within the null distribution, suggesting the correlation is",
                f"    entirely geometric and not specific to WARP's routing.",
            ])
        else:
            lines.append(f"    Effect attribution: {effect}")
        
        # Add relevant_fraction results if available
        if 'permutation_observed_rho_frac' in stats:
            lines.extend([
                f"",
                f"  Relevant Fraction Correlation (continuous):",
                f"    Observed ρ(traffic, relevant_frac): {stats.get('permutation_observed_rho_frac', np.nan):>+.4f}",
                f"    Null distribution mean:             {stats.get('permutation_null_mean_frac', np.nan):>+.4f}",
                f"    Permutation p-value:                {stats.get('permutation_permutation_p_value_frac', np.nan):.4f}",
            ])
        
        return "\n".join(lines)
    
    def _check_sensitivity_robustness(self, min_oracle_pct: float = 80.0) -> bool:
        """
        Check if the oracle-hub finding is robust across thresholds.
        
        Returns True if pct_with_oracle >= min_oracle_pct for all thresholds >= P75.
        """
        if not hasattr(self, 'hub_sensitivity_df') or self.hub_sensitivity_df.empty:
            return False
        
        # Check P75 and above
        high_thresholds = self.hub_sensitivity_df[self.hub_sensitivity_df['percentile'] >= 0.75]
        return (high_thresholds['pct_with_oracle'] >= min_oracle_pct).all()


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_h10_2(config_name: str = "production"):
    """Run H10.2 hypothesis test."""
    from hypothesis.configs import load_config
    
    config = load_config(config_name)
    test = H10_2_TrueYield(config)
    
    test.setup()
    result = test.analyze()
    test.results = result  # Store for report() to access
    result.save(test.output_dir)
    
    test.visualize()
    print(test.report())
    
    return result


if __name__ == "__main__":
    import sys
    config_name = sys.argv[1] if len(sys.argv) > 1 else "production"
    run_h10_2(config_name)
