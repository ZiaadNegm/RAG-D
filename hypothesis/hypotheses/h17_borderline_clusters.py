"""
Hypothesis H17: Borderline Clusters → Elevated Miss Rates

Claim: Clusters that consistently rank just below the nprobe cutoff have elevated 
       miss rates.

Mechanism (WARP-specific):
    WARP selects the top-nprobe centroids per query token based on centroid scores.
    Clusters that frequently rank at positions nprobe+1, nprobe+2, etc. are "borderline" -
    they're almost selected but miss the cutoff. These near-miss clusters may contain
    relevant tokens that get missed entirely.

    The hypothesis is that clusters with low but non-zero selection frequency are
    "borderline" - they're sometimes selected, sometimes not. This inconsistency
    may correlate with higher miss rates.

WORKAROUND NOTE:
    The ideal test would track centroid rankings beyond nprobe to identify true 
    borderline clusters. However, R0 only records selected centroids (ranks 0 to nprobe-1).
    
    As a workaround, we use LOW SELECTION FREQUENCY as a proxy for "borderline":
    - Clusters with sel_freq = 0 are "anti-hubs" (never selected)
    - Clusters with very low sel_freq (bottom quartile, non-zero) are likely borderline
    - These occasionally make the cut but often don't

Key Metrics:
    - sel_freq (B1): Selection frequency (proxy for borderline when low)
    - is_anti_hub: sel_freq == 0 (never selected)
    - m6_miss_rate: Miss rate attributable to this centroid
    - m6_miss_count: Total misses from this centroid

Test Plan:
    1. Identify "borderline" clusters: low sel_freq (Q1 excluding zeros)
    2. Compare miss rates: borderline vs consistently-selected (Q4)
    3. Expect borderline clusters to have higher miss rates when they ARE selected
    4. Anti-hubs (sel_freq=0) analyzed separately
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.stats import (
    compute_group_comparison,
    bootstrap_comparison,
    simple_regression_summary,
    bootstrap_ci
)
from hypothesis.viz import (
    plot_scatter_with_regression,
    plot_stratified_bars,
    plot_stratified_violin,
    plot_hypothesis_summary,
    save_figure
)


class H17_BorderlineClusters(HypothesisTest):
    """
    Hypothesis 17: Borderline clusters → elevated miss rates
    
    WORKAROUND: Since R0 doesn't record near-miss centroids, we use 
    LOW SELECTION FREQUENCY as a proxy for "borderline" behavior.
    
    Logic: Clusters selected infrequently are on the margin - sometimes
    they make nprobe cutoff, sometimes they don't. This inconsistency
    means their tokens may frequently be missed.
    """
    
    HYPOTHESIS_ID = "H17"
    HYPOTHESIS_NAME = "Borderline Clusters → Miss Rates"
    CLAIM = "Clusters that borderline the nprobe cutoff have elevated miss rates"
    
    # Limitation flag
    WORKAROUND_USED = True
    WORKAROUND_NOTE = (
        "R0 only records selected centroids. Low sel_freq used as proxy for borderline. "
        "True test would require extended R0 recording ranks beyond nprobe."
    )
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check required columns
        required = ['sel_freq', 'm6_miss_rate']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return self._empty_result(f"Missing columns: {missing}")
        
        # Build analysis dataframe
        cols = ['sel_freq', 'm6_miss_rate', 'm6_miss_count', 'n_tokens', 
                'yield', 'dispersion', 'is_anti_hub']
        available = [c for c in cols if c in df.columns]
        analysis_df = df[available].dropna(subset=['sel_freq', 'm6_miss_rate'])
        
        if len(analysis_df) < 10:
            return self._empty_result("Insufficient data")
        
        analysis_df = analysis_df.copy()
        
        # Ensure is_anti_hub column exists
        if 'is_anti_hub' not in analysis_df.columns:
            analysis_df['is_anti_hub'] = analysis_df['sel_freq'] == 0
        
        # Separate anti-hubs from selected clusters
        anti_hubs = analysis_df[analysis_df['is_anti_hub']]
        selected = analysis_df[~analysis_df['is_anti_hub']]
        
        if len(selected) < 10:
            return self._empty_result("Insufficient selected clusters for analysis")
        
        # Create borderline categories within selected clusters
        # Q1 = borderline (low sel_freq), Q4 = consistently selected (high sel_freq)
        try:
            selected = selected.copy()
            selected['borderline_bin'] = pd.qcut(
                selected['sel_freq'],
                q=4,
                labels=['Q1_borderline', 'Q2', 'Q3', 'Q4_consistent'],
                duplicates='drop'
            )
        except ValueError:
            # Fallback to cut
            selected['borderline_bin'] = pd.cut(
                selected['sel_freq'],
                bins=4,
                labels=['Q1_borderline', 'Q2', 'Q3', 'Q4_consistent']
            )
        
        # Primary test: Compare borderline (Q1) vs consistent (Q4) miss rates
        borderline = selected[selected['borderline_bin'] == 'Q1_borderline']
        consistent = selected[selected['borderline_bin'] == 'Q4_consistent']
        
        if len(borderline) > 1 and len(consistent) > 1:
            # Bootstrap comparison
            comparison = bootstrap_comparison(
                borderline['m6_miss_rate'].values,
                consistent['m6_miss_rate'].values
            )
            group_diff = comparison['observed_diff']
            group_pval = comparison['pvalue']
            
            # Mann-Whitney U test (non-parametric)
            _, mw_pval = mannwhitneyu(
                borderline['m6_miss_rate'],
                consistent['m6_miss_rate'],
                alternative='greater'  # borderline > consistent
            )
        else:
            group_diff = 0.0
            group_pval = 1.0
            mw_pval = 1.0
        
        # Secondary: Correlation within selected clusters
        # Expect NEGATIVE: lower sel_freq → higher miss rate
        corr_sel, pval_sel = spearmanr(
            selected['sel_freq'],
            selected['m6_miss_rate']
        )
        
        # Bootstrap CI
        def corr_func(data):
            n = len(data) // 2
            return spearmanr(data[:n], data[n:])[0]
        
        combined = np.concatenate([
            selected['sel_freq'].values,
            selected['m6_miss_rate'].values
        ])
        _, ci_lower, ci_upper = bootstrap_ci(combined, corr_func, n_bootstrap=500)
        
        # Stratified analysis
        stratified = compute_group_comparison(
            selected[selected['borderline_bin'].notna()],
            'borderline_bin',
            'm6_miss_rate'
        )
        stratified_results = stratified.to_dict('records')
        
        # Anti-hub statistics
        anti_hub_stats = {}
        if len(anti_hubs) > 0:
            anti_hub_stats['n_anti_hubs'] = len(anti_hubs)
            anti_hub_stats['anti_hub_mean_miss_rate'] = float(anti_hubs['m6_miss_rate'].mean())
            anti_hub_stats['anti_hub_total_misses'] = int(anti_hubs['m6_miss_count'].sum()) if 'm6_miss_count' in anti_hubs else 0
        
        # Hypothesis supported if:
        # 1. Borderline clusters have higher miss rate than consistent (group_diff > 0)
        # 2. Significant difference (p < 0.05)
        # OR negative correlation (low freq → high miss)
        supported = (group_diff > 0 and group_pval < 0.05) or (corr_sel < 0 and pval_sel < 0.05)
        
        # Primary effect: difference in miss rates
        effect_size = group_diff
        p_value = min(group_pval, mw_pval)
        
        stats = {
            'workaround_used': self.WORKAROUND_USED,
            'workaround_note': self.WORKAROUND_NOTE,
            'n_selected_clusters': len(selected),
            'n_borderline_q1': len(borderline),
            'n_consistent_q4': len(consistent),
            'mean_miss_rate_borderline': float(borderline['m6_miss_rate'].mean()) if len(borderline) > 0 else np.nan,
            'mean_miss_rate_consistent': float(consistent['m6_miss_rate'].mean()) if len(consistent) > 0 else np.nan,
            'miss_rate_diff_borderline_minus_consistent': group_diff,
            'bootstrap_pval': group_pval,
            'mann_whitney_pval': mw_pval,
            'spearman_selfreq_missrate': corr_sel,
            'spearman_pval': pval_sel,
            'stratified_analysis': stratified_results,
        }
        stats.update(anti_hub_stats)
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=effect_size,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=p_value,
            statistics=stats,
            config_name=self.config.name,
            n_observations=len(selected),
            timestamp=datetime.now().isoformat()
        )
    
    def _empty_result(self, error_msg: str) -> HypothesisResult:
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=False,
            effect_size=0.0,
            effect_size_ci=(0.0, 0.0),
            p_value=1.0,
            statistics={'error': error_msg, 'workaround_used': self.WORKAROUND_USED},
            config_name=self.config.name,
            n_observations=0,
            timestamp=datetime.now().isoformat()
        )
    
    def visualize(self):
        df = self.cluster_frame
        
        cols = ['sel_freq', 'm6_miss_rate', 'm6_miss_count', 'n_tokens', 'is_anti_hub']
        available = [c for c in cols if c in df.columns]
        plot_df = df[available].dropna(subset=['sel_freq', 'm6_miss_rate'])
        
        if len(plot_df) < 10:
            print("Insufficient data for visualization")
            return
        
        plot_df = plot_df.copy()
        
        if 'is_anti_hub' not in plot_df.columns:
            plot_df['is_anti_hub'] = plot_df['sel_freq'] == 0
        
        # Separate selected clusters
        selected = plot_df[~plot_df['is_anti_hub']]
        
        try:
            selected = selected.copy()
            selected['borderline_bin'] = pd.qcut(
                selected['sel_freq'],
                q=4,
                labels=['Q1_borderline', 'Q2', 'Q3', 'Q4_consistent'],
                duplicates='drop'
            )
        except ValueError:
            selected['borderline_bin'] = 'all'
        
        import matplotlib.pyplot as plt
        
        # 1. Main comparison: borderline vs consistent
        if selected['borderline_bin'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_stratified_bars(
                selected,
                'borderline_bin',
                'm6_miss_rate',
                title=f'{self.HYPOTHESIS_ID}: Miss Rate by Selection Frequency (Borderline Proxy)',
                xlabel='Selection Frequency Quartile (Q1=borderline, Q4=consistent)',
                ylabel='Miss Rate',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_borderline_comparison', str(self.output_dir))
            plt.close(fig)
        
        # 2. Scatter: sel_freq vs miss_rate (log scale for sel_freq)
        fig, ax = plt.subplots(figsize=(10, 8))
        sel_nonzero = selected[selected['sel_freq'] > 0]
        ax.scatter(
            np.log10(sel_nonzero['sel_freq']),
            sel_nonzero['m6_miss_rate'],
            s=50,
            alpha=0.5
        )
        ax.set_xlabel('log₁₀(Selection Frequency)')
        ax.set_ylabel('Miss Rate')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Selection Frequency vs Miss Rate\n(WORKAROUND: low freq = borderline proxy)')
        
        # Add annotation about workaround
        ax.annotate(
            'Lower sel_freq → borderline cluster\n(sometimes selected, sometimes not)',
            xy=(0.02, 0.98), xycoords='axes fraction',
            fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        save_figure(fig, f'{self.HYPOTHESIS_ID}_scatter', str(self.output_dir))
        plt.close(fig)
        
        # 3. Violin by borderline category
        if selected['borderline_bin'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_stratified_violin(
                selected,
                'borderline_bin',
                'm6_miss_rate',
                title=f'{self.HYPOTHESIS_ID}: Miss Rate Distribution by Borderline Status',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_violin', str(self.output_dir))
            plt.close(fig)
        
        # 4. Anti-hub analysis (separate)
        anti_hubs = plot_df[plot_df['is_anti_hub']]
        if len(anti_hubs) > 10:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Miss rate distribution for anti-hubs
            axes[0].hist(anti_hubs['m6_miss_rate'], bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Miss Rate')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'Anti-Hub Miss Rate Distribution (n={len(anti_hubs)})')
            
            # Comparison: anti-hub vs selected
            categories = ['Anti-hubs\n(sel_freq=0)', 'Selected\n(sel_freq>0)']
            means = [anti_hubs['m6_miss_rate'].mean(), selected['m6_miss_rate'].mean()]
            errors = [anti_hubs['m6_miss_rate'].std() / np.sqrt(len(anti_hubs)),
                     selected['m6_miss_rate'].std() / np.sqrt(len(selected))]
            
            axes[1].bar(categories, means, yerr=errors, capsize=5, color=['gray', 'steelblue'])
            axes[1].set_ylabel('Mean Miss Rate')
            axes[1].set_title('Anti-hubs vs Selected Clusters')
            
            plt.tight_layout()
            save_figure(fig, f'{self.HYPOTHESIS_ID}_antihub_analysis', str(self.output_dir))
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    from hypothesis.configs import load_config
    
    parser = argparse.ArgumentParser(description="Run H17: Borderline Clusters → Miss Rates")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config, override_run_dir=args.run_dir)
    test = H17_BorderlineClusters(config)
    test.run()
