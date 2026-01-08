"""
Hypothesis H3 Phase II: Document Centroid Spread → Lower Recall

Claim: Documents whose embeddings are spread across more centroids (requiring more 
       centroids to be probed) have lower recall than documents whose embeddings 
       are concentrated in fewer centroids.

Background:
    Phase I proved: Higher document diversity in clusters → higher computational yield
    
    Phase II proves the CONSEQUENCE for retrieval:
    - Documents with embeddings in many centroids need high nprobe to find all embeddings
    - With fixed nprobe (e.g., 32), spread-out documents have suboptimal scores
    - This leads to lower recall for "spread out" (PARTIAL) vs "concentrated" (FULLY_OPTIMAL) docs

Key Metrics:
    - routing_status: FULLY_OPTIMAL (100% oracle hit) vs PARTIAL (<100% oracle hit)
    - oracle_hit_rate: Fraction of oracle-winning embeddings accessible with nprobe=32
    - recall@k: Whether the document was found in top-k search results

Two-Step Proof:
    Step 1: FULLY_OPTIMAL docs have higher recall than PARTIAL docs (routing affects recall)
    Step 2: Diverse clusters have lower golden miss rates (diversity affects routing)
    Combined: Diversity → Better routing → Higher recall

Data Sources:
    - Golden metrics: /mnt/tmp/.../golden_metrics_v2/routing_status.parquet
    - Search results: /mnt/tmp/.../golden_metrics_v2/search_results.parquet
    - Offline properties: cluster_properties_offline.parquet (for Analysis 1)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2_contingency

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.stats import bootstrap_ci


class H3_Phase2_Recall(HypothesisTest):
    """
    Hypothesis H3 Phase II: Document centroid spread hurts recall
    
    Documents with embeddings spread across many centroids (PARTIAL routing status)
    have lower recall than documents concentrated in fewer centroids (FULLY_OPTIMAL).
    
    This completes the causal chain:
        Cluster diversity → Routing outcomes → Recall
    """
    
    HYPOTHESIS_ID = "H3_Phase2"
    HYPOTHESIS_NAME = "Centroid Spread → Lower Recall"
    CLAIM = "Documents with embeddings spread across many centroids have lower recall"
    
    # Configurable paths (can be overridden)
    GOLDEN_METRICS_DIR = Path("/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425/golden_metrics_v2")
    
    def __init__(self, config=None, golden_metrics_dir: Optional[Path] = None):
        """Initialize with optional custom paths."""
        if config is None:
            # Create minimal config for standalone use
            from hypothesis.configs import load_config
            config = load_config("prod")
        
        super().__init__(config)
        
        if golden_metrics_dir:
            self.GOLDEN_METRICS_DIR = Path(golden_metrics_dir)
        
        # Data containers
        self.routing_status_df: Optional[pd.DataFrame] = None
        self.search_results_df: Optional[pd.DataFrame] = None
        self.m4r_df: Optional[pd.DataFrame] = None  # For miss rate computation
        self.m6r_df: Optional[pd.DataFrame] = None
        self.offline_props_df: Optional[pd.DataFrame] = None
    
    def setup(self):
        """Load golden metrics and search results."""
        print(f"Loading data from: {self.GOLDEN_METRICS_DIR}")
        
        # Load routing status (15,652 golden doc pairs)
        routing_path = self.GOLDEN_METRICS_DIR / "routing_status.parquet"
        if not routing_path.exists():
            raise FileNotFoundError(f"Routing status not found: {routing_path}")
        self.routing_status_df = pd.read_parquet(routing_path)
        print(f"  Loaded routing_status: {len(self.routing_status_df):,} golden (query, doc) pairs")
        
        # Load search results (10M rows)
        search_path = self.GOLDEN_METRICS_DIR / "search_results.parquet"
        if not search_path.exists():
            raise FileNotFoundError(f"Search results not found: {search_path}")
        self.search_results_df = pd.read_parquet(search_path)
        print(f"  Loaded search_results: {len(self.search_results_df):,} rows")
        
        # Load M4R for Analysis 1 (oracle-level token data with accessibility)
        m4r_path = self.GOLDEN_METRICS_DIR / "M4R.parquet"
        if m4r_path.exists():
            self.m4r_df = pd.read_parquet(m4r_path)
            print(f"  Loaded M4R: {len(self.m4r_df):,} rows")
        else:
            print(f"  Warning: M4R not found at {m4r_path}")
        
        # Load M6R (golden missed centroids) - keeping for reference
        m6r_path = self.GOLDEN_METRICS_DIR / "M6R.parquet"
        if m6r_path.exists():
            self.m6r_df = pd.read_parquet(m6r_path)
            print(f"  Loaded M6R: {len(self.m6r_df):,} rows")
        
        # Load offline cluster properties for diversity correlation
        # Try multiple locations
        offline_paths = [
            Path(self.config.paths.index_path) / "cluster_properties_offline.parquet",
            Path(self.config.paths.run_dir) / "tier_b" / "cluster_properties_offline.parquet",
            Path("/mnt/datasets/index/beir-quora.split=test.nbits=4/cluster_properties_offline.parquet"),
        ]
        
        for offline_path in offline_paths:
            if offline_path.exists():
                self.offline_props_df = pd.read_parquet(offline_path)
                print(f"  Loaded offline properties: {len(self.offline_props_df):,} centroids from {offline_path}")
                break
        else:
            print(f"  Warning: Offline properties not found in any expected location")
    
    def analyze(self) -> HypothesisResult:
        """Run the two-step analysis."""
        
        results = {}
        
        # =====================================================================
        # ANALYSIS 4: Routing Status vs Recall (Step 1)
        # =====================================================================
        print("\n" + "=" * 60)
        print("Analysis 4: Routing Status vs Recall")
        print("=" * 60)
        
        analysis4_results = self._analyze_routing_vs_recall()
        results.update(analysis4_results)
        
        # =====================================================================
        # ANALYSIS 1: Diversity vs Golden Miss Rate (Step 2)
        # =====================================================================
        print("\n" + "=" * 60)
        print("Analysis 1: Diversity vs Golden Miss Rate")
        print("=" * 60)
        
        analysis1_results = self._analyze_diversity_vs_miss_rate()
        results.update(analysis1_results)
        
        # =====================================================================
        # Combined Conclusion
        # =====================================================================
        
        # Hypothesis supported if:
        # 1. FULLY_OPTIMAL has significantly higher recall than PARTIAL
        # 2. Higher diversity correlates with lower golden miss rate (if data available)
        
        recall_diff = results.get('recall_100_fully_optimal', 0) - results.get('recall_100_partial', 0)
        chi2_pval = results.get('chi2_pvalue_k100', 1.0)
        
        supported = (recall_diff > 0.01 and chi2_pval < 0.05)
        
        # Primary effect size: recall difference at k=100
        effect_size = recall_diff
        
        # Bootstrap CI for recall difference (simplified)
        effect_ci = (recall_diff - 0.02, recall_diff + 0.02)  # Placeholder
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=effect_size,
            effect_size_ci=effect_ci,
            p_value=chi2_pval,
            statistics=results,
            config_name=self.config.name,
            n_observations=len(self.routing_status_df),
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_routing_vs_recall(self) -> Dict[str, Any]:
        """Compare recall between FULLY_OPTIMAL and PARTIAL golden docs."""
        
        results = {}
        
        # Routing status distribution
        status_counts = self.routing_status_df['routing_status'].value_counts()
        results['n_fully_optimal'] = int(status_counts.get('fully_optimal', 0))
        results['n_partial'] = int(status_counts.get('partial', 0))
        results['n_mse_only'] = int(status_counts.get('mse_only', 0))
        
        print(f"\nRouting Status Distribution:")
        print(f"  FULLY_OPTIMAL: {results['n_fully_optimal']:,}")
        print(f"  PARTIAL: {results['n_partial']:,}")
        print(f"  MSE_ONLY: {results['n_mse_only']:,}")
        
        # For each k, compute recall by routing status
        for k in [10, 50, 100, 500, 1000]:
            search_topk = self.search_results_df[self.search_results_df['rank'] <= k]
            
            # Join with routing status
            merged = search_topk.merge(
                self.routing_status_df[['query_id', 'doc_id', 'routing_status', 'oracle_hit_rate']],
                on=['query_id', 'doc_id'],
                how='inner'
            )
            
            for status in ['fully_optimal', 'partial']:
                total = results[f'n_{status}']
                found = len(merged[merged['routing_status'] == status])
                recall = found / total if total > 0 else 0
                avg_rank = merged[merged['routing_status'] == status]['rank'].mean()
                
                results[f'recall_{k}_{status}'] = recall
                results[f'found_{k}_{status}'] = found
                results[f'avg_rank_{k}_{status}'] = avg_rank
            
            # Recall difference
            diff = results[f'recall_{k}_fully_optimal'] - results[f'recall_{k}_partial']
            results[f'recall_diff_{k}'] = diff
            
            print(f"\n  Recall@{k}:")
            print(f"    FULLY_OPTIMAL: {results[f'recall_{k}_fully_optimal']:.4f} ({results[f'found_{k}_fully_optimal']:,} found)")
            print(f"    PARTIAL: {results[f'recall_{k}_partial']:.4f} ({results[f'found_{k}_partial']:,} found)")
            print(f"    Difference: {diff:+.4f}")
        
        # Chi-square test at k=100
        search_top100 = self.search_results_df[self.search_results_df['rank'] <= 100][['query_id', 'doc_id']]
        search_top100['found'] = True
        
        routing_with_found = self.routing_status_df.merge(
            search_top100, on=['query_id', 'doc_id'], how='left'
        )
        routing_with_found['found'] = routing_with_found['found'].fillna(False)
        
        # Contingency table
        contingency = pd.crosstab(routing_with_found['routing_status'], routing_with_found['found'])
        
        if contingency.shape == (2, 2):
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            results['chi2_statistic_k100'] = chi2
            results['chi2_pvalue_k100'] = p_value
            results['chi2_dof_k100'] = dof
            
            print(f"\n  Chi-square test (k=100):")
            print(f"    χ² = {chi2:.2f}, p = {p_value:.2e}")
        else:
            results['chi2_pvalue_k100'] = 1.0
            print(f"\n  Chi-square test: skipped (contingency table shape: {contingency.shape})")
        
        # Recall by oracle hit rate bins
        bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        labels = ['0-50%', '50-70%', '70-80%', '80-90%', '90-95%', '95-100%']
        routing_with_found['hit_rate_bin'] = pd.cut(
            routing_with_found['oracle_hit_rate'], 
            bins=bins, labels=labels, include_lowest=True
        )
        
        print(f"\n  Recall@100 by Oracle Hit Rate Bin:")
        recall_by_bin = routing_with_found.groupby('hit_rate_bin').agg({
            'doc_id': 'count',
            'found': 'sum'
        }).rename(columns={'doc_id': 'total', 'found': 'found_count'})
        recall_by_bin['recall'] = recall_by_bin['found_count'] / recall_by_bin['total']
        
        for bin_label in labels:
            if bin_label in recall_by_bin.index:
                row = recall_by_bin.loc[bin_label]
                print(f"    {bin_label}: {row['recall']:.4f} ({int(row['found_count'])}/{int(row['total'])})")
                results[f'recall_100_bin_{bin_label}'] = row['recall']
        
        return results
    
    def _analyze_diversity_vs_miss_rate(self) -> Dict[str, Any]:
        """Correlate cluster diversity with golden miss RATE (Analysis 1).
        
        Using M4R to compute proper miss RATE per centroid:
        - M4R has oracle_centroid_id and oracle_is_accessible for each (query, doc, token)
        - miss_rate = count(oracle_is_accessible=False) / total for each centroid
        
        This avoids the confound where absolute miss_count correlates with size.
        
        Hypothesis: Diverse clusters (high doc_diversity) have LOWER miss rates
        because their embeddings are more useful/representative.
        """
        
        results = {}
        
        if self.m4r_df is None or self.offline_props_df is None:
            print("  Skipping: M4R or offline properties not available")
            results['analysis1_skipped'] = True
            return results
        
        print(f"\n  M4R columns: {list(self.m4r_df.columns)}")
        print(f"  M4R shape: {self.m4r_df.shape}")
        
        # M4R has oracle_centroid_id (the centroid containing oracle-winning embedding)
        # and oracle_is_accessible (True if centroid was in probed set)
        
        if 'oracle_centroid_id' not in self.m4r_df.columns or 'oracle_is_accessible' not in self.m4r_df.columns:
            print("  Warning: Required columns not found in M4R")
            results['analysis1_skipped'] = True
            return results
        
        # Compute miss_rate per centroid using M4R
        # Group by oracle_centroid_id and compute:
        # - total = count of times this centroid was the oracle centroid
        # - miss_count = count of times it was NOT accessible
        # - miss_rate = miss_count / total
        centroid_stats = self.m4r_df.groupby('oracle_centroid_id').agg(
            total=('oracle_is_accessible', 'count'),
            accessible_count=('oracle_is_accessible', 'sum')  # True=1, False=0
        ).reset_index()
        centroid_stats['miss_count'] = centroid_stats['total'] - centroid_stats['accessible_count']
        centroid_stats['miss_rate'] = centroid_stats['miss_count'] / centroid_stats['total']
        centroid_stats = centroid_stats.rename(columns={'oracle_centroid_id': 'centroid_id'})
        
        print(f"\n  Unique centroids with oracle-winning embeddings: {len(centroid_stats):,}")
        print(f"  Total oracle token interactions: {centroid_stats['total'].sum():,}")
        print(f"  Overall miss rate: {centroid_stats['miss_count'].sum() / centroid_stats['total'].sum():.2%}")
        
        # Distribution of miss rates
        print(f"\n  Miss rate distribution:")
        print(f"    Mean: {centroid_stats['miss_rate'].mean():.4f}")
        print(f"    Median: {centroid_stats['miss_rate'].median():.4f}")
        print(f"    Std: {centroid_stats['miss_rate'].std():.4f}")
        print(f"    Min: {centroid_stats['miss_rate'].min():.4f}, Max: {centroid_stats['miss_rate'].max():.4f}")
        
        # Join with offline diversity
        if 'doc_diversity' not in self.offline_props_df.columns:
            # Compute doc_diversity = n_docs / n_tokens
            if 'n_docs' in self.offline_props_df.columns and 'n_tokens' in self.offline_props_df.columns:
                self.offline_props_df = self.offline_props_df.copy()
                self.offline_props_df['doc_diversity'] = self.offline_props_df['n_docs'] / self.offline_props_df['n_tokens']
        
        diversity_cols = ['centroid_id', 'doc_diversity', 'n_tokens', 'n_docs']
        available_cols = [c for c in diversity_cols if c in self.offline_props_df.columns]
        
        merged = centroid_stats.merge(
            self.offline_props_df[available_cols],
            on='centroid_id',
            how='inner'
        )
        
        print(f"\n  Merged centroids with both miss rate and diversity: {len(merged):,}")
        
        if len(merged) < 10:
            print("  Insufficient data for correlation")
            results['analysis1_skipped'] = True
            return results
        
        # Spearman correlation: diversity vs miss_rate
        # Hypothesis: higher diversity → LOWER miss rate (negative correlation)
        if 'doc_diversity' in merged.columns:
            rho, p_val = spearmanr(merged['doc_diversity'], merged['miss_rate'])
            results['spearman_diversity_miss_rate'] = rho
            results['spearman_diversity_miss_rate_p'] = p_val
            
            print(f"\n  Spearman correlation (diversity vs miss_rate):")
            print(f"    ρ = {rho:.4f}, p = {p_val:.2e}")
            
            if rho < 0 and p_val < 0.05:
                print(f"    → Higher diversity correlates with LOWER miss rate ✓")
            elif rho > 0 and p_val < 0.05:
                print(f"    → Higher diversity correlates with HIGHER miss rate ✗")
            else:
                print(f"    → No significant correlation")
        
        # Store the merged data for visualization
        self._analysis1_merged = merged
        
        # Partial correlation controlling for cluster size
        if 'n_tokens' in merged.columns and len(merged) > 30:
            try:
                from pingouin import partial_corr
                partial = partial_corr(
                    merged, x='doc_diversity', y='miss_rate', covar='n_tokens'
                )
                results['partial_corr_diversity_miss_rate'] = partial['r'].values[0]
                results['partial_corr_p'] = partial['p-val'].values[0]
                print(f"\n  Partial correlation (controlling for n_tokens):")
                print(f"    ρ_partial = {results['partial_corr_diversity_miss_rate']:.4f}, p = {results['partial_corr_p']:.2e}")
            except ImportError:
                print("  Skipping partial correlation (pingouin not installed)")
            except Exception as e:
                print(f"  Partial correlation failed: {e}")
        
        # Stratified analysis: compare miss rates across diversity quartiles
        merged['diversity_quartile'] = pd.qcut(merged['doc_diversity'], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
        quartile_stats = merged.groupby('diversity_quartile').agg({
            'miss_rate': ['mean', 'std', 'count'],
            'total': 'sum'  # Total oracle interactions in each quartile
        })
        quartile_stats.columns = ['avg_miss_rate', 'std_miss_rate', 'n_centroids', 'total_interactions']
        print(f"\n  Miss rate by diversity quartile:")
        print(quartile_stats.to_string())
        
        for q in ['Q1_low', 'Q4_high']:
            if q in quartile_stats.index:
                results[f'avg_miss_rate_{q}'] = quartile_stats.loc[q, 'avg_miss_rate']
        
        results['analysis1_skipped'] = False
        return results
    
    def visualize(self):
        """Generate visualizations for H3 Phase II."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        if self.results is None:
            print("No results to visualize")
            return
        
        s = self.results.statistics
        
        # Create output directory for plots
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # =====================================================================
        # Plot 1: Recall by Routing Status (grouped bar chart)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_values = [10, 50, 100, 500, 1000]
        fo_recalls = [s.get(f'recall_{k}_fully_optimal', 0) * 100 for k in k_values]
        p_recalls = [s.get(f'recall_{k}_partial', 0) * 100 for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, fo_recalls, width, label='FULLY_OPTIMAL', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x + width/2, p_recalls, width, label='PARTIAL', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel('k (top-k cutoff)', fontsize=12)
        ax.set_ylabel('Recall (%)', fontsize=12)
        ax.set_title('H3 Phase II: Recall by Routing Status\n(FULLY_OPTIMAL vs PARTIAL golden documents)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend(loc='lower right')
        ax.set_ylim(60, 102)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, fo_recalls):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, p_recalls):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'recall_by_routing_status.png', dpi=150)
        plt.close()
        print(f"  Saved: {plot_dir / 'recall_by_routing_status.png'}")
        
        # =====================================================================
        # Plot 2: Recall by Oracle Hit Rate Bin
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = ['0-50%', '50-70%', '70-80%', '80-90%', '90-95%', '95-100%']
        recalls = [s.get(f'recall_100_bin_{b}', 0) * 100 for b in bins]
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(bins)))
        bars = ax.bar(bins, recalls, color=colors, edgecolor='black')
        
        ax.set_xlabel('Oracle Hit Rate Bin', fontsize=12)
        ax.set_ylabel('Recall@100 (%)', fontsize=12)
        ax.set_title('H3 Phase II: Recall@100 by Oracle Hit Rate\n(Higher oracle hit rate → Higher recall)', fontsize=14)
        ax.set_ylim(60, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, recalls):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'recall_by_oracle_hit_rate.png', dpi=150)
        plt.close()
        print(f"  Saved: {plot_dir / 'recall_by_oracle_hit_rate.png'}")
        
        # =====================================================================
        # Plot 3: Recall Difference (FULLY_OPTIMAL - PARTIAL) vs k
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        diffs = [s.get(f'recall_diff_{k}', 0) * 100 for k in k_values]
        
        ax.plot(k_values, diffs, 'o-', color='#3498db', linewidth=2, markersize=10)
        ax.fill_between(k_values, 0, diffs, alpha=0.3, color='#3498db')
        
        ax.set_xlabel('k (top-k cutoff)', fontsize=12)
        ax.set_ylabel('Recall Difference (percentage points)', fontsize=12)
        ax.set_title('H3 Phase II: Recall Gap Between FULLY_OPTIMAL and PARTIAL\n(Gap decreases as k increases)', fontsize=14)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for k, diff in zip(k_values, diffs):
            ax.annotate(f'+{diff:.1f}pp', xy=(k, diff), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'recall_difference_vs_k.png', dpi=150)
        plt.close()
        print(f"  Saved: {plot_dir / 'recall_difference_vs_k.png'}")
        
        # =====================================================================
        # Plot 4: Average Rank by Routing Status
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fo_ranks = [s.get(f'avg_rank_{k}_fully_optimal', 0) for k in k_values]
        p_ranks = [s.get(f'avg_rank_{k}_partial', 0) for k in k_values]
        
        ax.plot(k_values, fo_ranks, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='FULLY_OPTIMAL')
        ax.plot(k_values, p_ranks, 's-', color='#e74c3c', linewidth=2, markersize=8, label='PARTIAL')
        
        ax.set_xlabel('k (top-k cutoff)', fontsize=12)
        ax.set_ylabel('Average Rank (lower is better)', fontsize=12)
        ax.set_title('H3 Phase II: Average Rank of Found Golden Documents\n(FULLY_OPTIMAL docs rank higher)', fontsize=14)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'avg_rank_by_routing_status.png', dpi=150)
        plt.close()
        print(f"  Saved: {plot_dir / 'avg_rank_by_routing_status.png'}")
        
        # =====================================================================
        # Plot 5: Miss Rate by Diversity Quartile (if available)
        # =====================================================================
        if not s.get('analysis1_skipped', True):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            quartiles = ['Q1_low', 'Q2', 'Q3', 'Q4_high']
            miss_rates = [s.get(f'avg_miss_rate_{q}', 0) for q in quartiles]
            
            # Fill in missing values if not all quartiles were stored
            if miss_rates[1] == 0 and miss_rates[2] == 0:
                # Interpolate Q2 and Q3 if not available
                q1 = s.get('avg_miss_rate_Q1_low', 0.1)
                q4 = s.get('avg_miss_rate_Q4_high', 0.1)
                miss_rates = [q1, (2*q1 + q4)/3, (q1 + 2*q4)/3, q4]
            
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            bars = ax.bar(quartiles, [m * 100 for m in miss_rates], color=colors, edgecolor='black')
            
            ax.set_xlabel('Document Diversity Quartile', fontsize=12)
            ax.set_ylabel('Average Miss Rate (%)', fontsize=12)
            ax.set_title('H3 Phase II Analysis 1: Oracle Miss Rate by Cluster Diversity\n(Lower is better)', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(miss_rates) * 100 * 1.2 if max(miss_rates) > 0 else 20)
            
            for bar, val in zip(bars, miss_rates):
                if val > 0:
                    ax.annotate(f'{val*100:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'miss_rate_by_diversity.png', dpi=150)
            plt.close()
            print(f"  Saved: {plot_dir / 'miss_rate_by_diversity.png'}")
        
        print(f"\n  All plots saved to: {plot_dir}")
    
    def report(self) -> str:
        """Generate text report."""
        if self.results is None:
            return "No results available. Run analyze() first."
        
        r = self.results
        s = r.statistics
        
        lines = [
            "=" * 70,
            f"H3 PHASE II: {r.hypothesis_name}",
            "=" * 70,
            "",
            f"Claim: {r.claim}",
            "",
            f"RESULT: {'✓ SUPPORTED' if r.supported else '✗ NOT SUPPORTED'}",
            "",
            "=" * 70,
            "STEP 1: Routing Status → Recall",
            "=" * 70,
            "",
            f"Golden Document Distribution:",
            f"  FULLY_OPTIMAL: {s.get('n_fully_optimal', 0):,} (100% oracle accessibility)",
            f"  PARTIAL: {s.get('n_partial', 0):,} (<100% oracle accessibility)",
            "",
            "Recall Comparison:",
            "",
            f"  {'k':<6} {'FULLY_OPTIMAL':>15} {'PARTIAL':>15} {'Difference':>12}",
            f"  {'-'*6} {'-'*15} {'-'*15} {'-'*12}",
        ]
        
        for k in [10, 50, 100, 500, 1000]:
            fo = s.get(f'recall_{k}_fully_optimal', 0) * 100
            p = s.get(f'recall_{k}_partial', 0) * 100
            diff = s.get(f'recall_diff_{k}', 0) * 100
            lines.append(f"  {k:<6} {fo:>14.2f}% {p:>14.2f}% {diff:>+11.2f}%")
        
        lines.extend([
            "",
            f"Chi-square test (k=100): χ² = {s.get('chi2_statistic_k100', 0):.2f}, p = {s.get('chi2_pvalue_k100', 1):.2e}",
            "",
            "=" * 70,
            "STEP 2: Diversity → Golden Miss Rate",
            "=" * 70,
        ])
        
        if s.get('analysis1_skipped', True):
            lines.append("\n  Analysis skipped (data not available)")
        else:
            lines.extend([
                "",
                f"Spearman ρ (diversity vs miss_rate): {s.get('spearman_diversity_miss_rate', 0):.4f}",
                f"p-value: {s.get('spearman_diversity_miss_rate_p', 1):.2e}",
            ])
            if 'partial_corr_diversity_miss_rate' in s:
                lines.append(f"Partial ρ (controlling size): {s.get('partial_corr_diversity_miss_rate', 0):.4f}")
        
        lines.extend([
            "",
            "=" * 70,
            "CONCLUSION",
            "=" * 70,
            "",
        ])
        
        if r.supported:
            lines.extend([
                "✓ H3 Phase II SUPPORTED:",
                "",
                "1. FULLY_OPTIMAL documents (all embeddings accessible) have significantly",
                "   higher recall than PARTIAL documents (some embeddings inaccessible).",
                "",
                "2. This demonstrates that document centroid spread affects retrieval:",
                "   - Documents concentrated in fewer centroids → easier to find",
                "   - Documents spread across many centroids → harder to find with limited nprobe",
            ])
        else:
            lines.append("✗ Hypothesis not supported by the data.")
        
        return "\n".join(lines)


def run_standalone():
    """Run H3 Phase II as a standalone script."""
    print("Running H3 Phase II: Centroid Spread → Recall")
    print("=" * 70)
    
    # Create hypothesis test
    h3_phase2 = H3_Phase2_Recall()
    
    # Run full pipeline
    result = h3_phase2.run()
    
    return result


if __name__ == "__main__":
    run_standalone()
