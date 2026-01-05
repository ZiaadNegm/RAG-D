"""
Hypothesis H10: Centroid Hubness → Increased Redundant Computation

Claim: Higher centroid hubness (frequently selected clusters) leads to increased 
       redundant computation, holding cluster size constant.

Mechanism (WARP-specific):
    "Hubs" are centroids that get selected by many different query tokens. This can
    happen due to geometric properties (central position in embedding space) or 
    because the centroid happens to be a good approximate match for diverse queries.
    
    When a centroid is frequently selected (high sel_freq), it accumulates more 
    token-token similarity computations. However, not all these computations are 
    useful - many may be "redundant" (M2) because the document already won MaxSim 
    from another token or the similarity doesn't win.

Key Metrics:
    - sel_freq (B1): Number of times centroid was selected across all queries
    - traffic_share (B2): Fraction of total selections this centroid received
    - hub_type (B4): Classification (good_hub, bad_hub, normal)
    - yield (A4): influential / computed (efficiency)
    - redundancy_rate: m2_redundant_sims / m1_total_sims
    - hubness_bin: Quartile bin by sel_freq

Test Plan:
    1. Correlate sel_freq with redundancy_rate
    2. Stratify by hubness_bin, compare redundancy rates
    3. Compare "bad hubs" (high traffic, low yield) vs "good hubs" (high traffic, high yield)
    4. Control for cluster size
    5. Expect: high hubness → higher redundancy_rate (especially for bad hubs)
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


class H10_HubnessRedundancy(HypothesisTest):
    """
    Hypothesis 10: Centroid hubness → increased redundant computation
    
    High hubness = frequently selected by many query tokens → accumulates 
    more computation, but much of it may be wasted (redundant).
    
    Key insight: Distinguish "good hubs" (high yield) from "bad hubs" (low yield).
    Bad hubs are the problematic ones - lots of traffic but poor efficiency.
    """
    
    HYPOTHESIS_ID = "H10"
    HYPOTHESIS_NAME = "Hubness → Redundancy"
    CLAIM = "Higher centroid hubness leads to increased redundant computation"
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check required columns
        required = ['sel_freq', 'redundancy_rate', 'yield']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return self._empty_result(f"Missing columns: {missing}")
        
        # Build analysis dataframe
        cols = ['sel_freq', 'traffic_share', 'redundancy_rate', 'yield', 
                'hub_type', 'n_tokens', 'm1_total_sims', 'm2_redundant_sims']
        available_cols = [c for c in cols if c in df.columns]
        analysis_df = df[available_cols].dropna(subset=['sel_freq', 'redundancy_rate'])
        
        if len(analysis_df) < 10:
            return self._empty_result("Insufficient data")
        
        analysis_df = analysis_df.copy()
        
        # Create hubness bins
        try:
            analysis_df['hubness_bin'] = pd.qcut(
                analysis_df['sel_freq'],
                q=4,
                labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
                duplicates='drop'
            )
        except ValueError:
            # Fallback if not enough unique values
            analysis_df['hubness_bin'] = pd.cut(
                analysis_df['sel_freq'],
                bins=4,
                labels=['Q1_low', 'Q2', 'Q3', 'Q4_high']
            )
        
        # Primary test: Spearman correlation (hubness vs redundancy)
        # Expect POSITIVE: more hub-like → more redundancy
        corr_hub, pval_hub = spearmanr(
            analysis_df['sel_freq'],
            analysis_df['redundancy_rate']
        )
        
        # Secondary: yield vs redundancy (sanity check - should be negative)
        if 'yield' in analysis_df.columns:
            corr_yield, pval_yield = spearmanr(
                analysis_df['yield'].dropna(),
                analysis_df.loc[analysis_df['yield'].notna(), 'redundancy_rate']
            )
        else:
            corr_yield, pval_yield = np.nan, np.nan
        
        # Bootstrap CI for primary correlation
        def corr_func(data):
            n = len(data) // 2
            return spearmanr(data[:n], data[n:])[0]
        
        combined = np.concatenate([
            analysis_df['sel_freq'].values,
            analysis_df['redundancy_rate'].values
        ])
        _, ci_lower, ci_upper = bootstrap_ci(combined, corr_func, n_bootstrap=500)
        
        # Compare low-hubness vs high-hubness clusters
        low_hub = analysis_df[analysis_df['hubness_bin'] == 'Q1_low']
        high_hub = analysis_df[analysis_df['hubness_bin'] == 'Q4_high']
        
        if len(low_hub) > 1 and len(high_hub) > 1:
            comparison = bootstrap_comparison(
                high_hub['redundancy_rate'].values,
                low_hub['redundancy_rate'].values
            )
            group_diff = comparison['observed_diff']
            group_pval = comparison['pvalue']
        else:
            group_diff = 0.0
            group_pval = 1.0
        
        # Compare hub types if available
        hub_type_stats = {}
        if 'hub_type' in analysis_df.columns:
            for ht in analysis_df['hub_type'].unique():
                subset = analysis_df[analysis_df['hub_type'] == ht]
                if len(subset) > 0:
                    hub_type_stats[f'{ht}_count'] = len(subset)
                    hub_type_stats[f'{ht}_mean_redundancy'] = float(subset['redundancy_rate'].mean())
                    hub_type_stats[f'{ht}_mean_yield'] = float(subset['yield'].mean()) if 'yield' in subset else np.nan
        
        # Stratified analysis
        if 'hubness_bin' in analysis_df.columns:
            stratified = compute_group_comparison(
                analysis_df[analysis_df['hubness_bin'].notna()],
                'hubness_bin',
                'redundancy_rate'
            )
            stratified_results = stratified.to_dict('records')
        else:
            stratified_results = []
        
        # Regression controlling for size
        reg_results = simple_regression_summary(
            analysis_df, 'sel_freq', 'redundancy_rate'
        )
        
        # Hypothesis supported if positive correlation with p < 0.05
        supported = corr_hub > 0 and pval_hub < 0.05
        
        stats = {
            'spearman_hubness_redundancy': corr_hub,
            'spearman_hubness_redundancy_p': pval_hub,
            'spearman_yield_redundancy': corr_yield,
            'spearman_yield_redundancy_p': pval_yield,
            'regression_slope': reg_results.get('slope', np.nan),
            'regression_r_squared': reg_results.get('r_squared', np.nan),
            'mean_redundancy_low_hubness': float(low_hub['redundancy_rate'].mean()) if len(low_hub) > 0 else np.nan,
            'mean_redundancy_high_hubness': float(high_hub['redundancy_rate'].mean()) if len(high_hub) > 0 else np.nan,
            'group_diff_high_minus_low': group_diff,
            'group_pval': group_pval,
            'stratified_analysis': stratified_results,
        }
        stats.update(hub_type_stats)
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=corr_hub,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=pval_hub,
            statistics=stats,
            config_name=self.config.name,
            n_observations=len(analysis_df),
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
            statistics={'error': error_msg},
            config_name=self.config.name,
            n_observations=0,
            timestamp=datetime.now().isoformat()
        )
    
    def visualize(self):
        df = self.cluster_frame
        
        cols = ['sel_freq', 'traffic_share', 'redundancy_rate', 'yield', 'hub_type', 'n_tokens']
        available = [c for c in cols if c in df.columns]
        plot_df = df[available].dropna(subset=['sel_freq', 'redundancy_rate'])
        
        if len(plot_df) < 10:
            print("Insufficient data for visualization")
            return
        
        plot_df = plot_df.copy()
        
        try:
            plot_df['hubness_bin'] = pd.qcut(
                plot_df['sel_freq'],
                q=4,
                labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
                duplicates='drop'
            )
        except ValueError:
            plot_df['hubness_bin'] = 'all'
        
        import matplotlib.pyplot as plt
        
        # 1. Summary plot
        try:
            fig = plot_hypothesis_summary(
                plot_df,
                x_col='sel_freq',
                y_col='redundancy_rate',
                bin_col='hubness_bin',
                title=f'{self.HYPOTHESIS_ID}: {self.CLAIM}'
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_summary', str(self.output_dir))
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create summary plot: {e}")
        
        # 2. Scatter: sel_freq vs redundancy, colored by yield
        fig, ax = plt.subplots(figsize=(10, 8))
        if 'yield' in plot_df.columns:
            scatter = ax.scatter(
                plot_df['sel_freq'],
                plot_df['redundancy_rate'],
                c=plot_df['yield'],
                s=50,
                alpha=0.5,
                cmap='RdYlGn'  # Red=low yield (bad), Green=high yield (good)
            )
            plt.colorbar(scatter, label='Yield (efficiency)')
        else:
            ax.scatter(
                plot_df['sel_freq'],
                plot_df['redundancy_rate'],
                s=50,
                alpha=0.5
            )
        ax.set_xlabel('Selection Frequency (hubness)')
        ax.set_ylabel('Redundancy Rate')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Hubness vs Redundancy (colored by yield)')
        save_figure(fig, f'{self.HYPOTHESIS_ID}_scatter_yield', str(self.output_dir))
        plt.close(fig)
        
        # 3. Log-scale scatter for highly skewed sel_freq
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            np.log10(plot_df['sel_freq'] + 1),
            plot_df['redundancy_rate'],
            c=np.log10(plot_df['n_tokens'] + 1) if 'n_tokens' in plot_df else None,
            s=50,
            alpha=0.5,
            cmap='viridis'
        )
        ax.set_xlabel('log₁₀(Selection Frequency + 1)')
        ax.set_ylabel('Redundancy Rate')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Hubness (log scale) vs Redundancy')
        if 'n_tokens' in plot_df:
            plt.colorbar(ax.collections[0], label='log₁₀(n_tokens)')
        save_figure(fig, f'{self.HYPOTHESIS_ID}_scatter_log', str(self.output_dir))
        plt.close(fig)
        
        # 4. Violin by hubness quartile
        if plot_df['hubness_bin'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_stratified_violin(
                plot_df,
                'hubness_bin',
                'redundancy_rate',
                title=f'{self.HYPOTHESIS_ID}: Redundancy by Hubness Level',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_violin', str(self.output_dir))
            plt.close(fig)
        
        # 5. Hub type comparison (if available)
        if 'hub_type' in plot_df.columns and plot_df['hub_type'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_stratified_bars(
                plot_df,
                'hub_type',
                'redundancy_rate',
                title=f'{self.HYPOTHESIS_ID}: Redundancy by Hub Type',
                xlabel='Hub Type',
                ylabel='Redundancy Rate',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_hubtype_bars', str(self.output_dir))
            plt.close(fig)
        
        # 6. Yield vs Redundancy (sanity check)
        if 'yield' in plot_df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_scatter_with_regression(
                plot_df,
                x='yield',
                y='redundancy_rate',
                title=f'{self.HYPOTHESIS_ID}: Yield vs Redundancy (sanity check)',
                xlabel='Yield (influential / computed)',
                ylabel='Redundancy Rate',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_yield_redundancy', str(self.output_dir))
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    from hypothesis.configs import load_config
    
    parser = argparse.ArgumentParser(description="Run H10: Hubness → Redundancy")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config, override_run_dir=args.run_dir)
    test = H10_HubnessRedundancy(config)
    test.run()
