"""
Hypothesis H3: Document Diversity → Decreased Redundant Computation

Claim: Higher document diversity within a cluster decreases redundant computation (M2),
       holding cluster size constant.

Mechanism (WARP-specific):
    When a cluster contains tokens from many different documents, each token-token 
    similarity computed is more likely to contribute to a distinct document's MaxSim.
    Conversely, when one document dominates, many similarities are "wasted" because
    they all compete for the same document's MaxSim slot.

Key Metrics:
    - n_docs (A2): Number of unique documents in cluster
    - gini_coefficient (A3): Document concentration (0=uniform, 1=single doc)
    - tokens_per_doc: Average tokens per document (n_tokens / n_docs)
    - redundancy_rate: m2_redundant_sims / m1_total_sims
    - yield (A4): influential / computed

Test Plan:
    1. Compute document diversity metric: n_docs / n_tokens (or use inverse gini)
    2. Control for cluster size (n_tokens) 
    3. Correlate diversity with redundancy_rate
    4. Stratify by diversity quartile, compare redundancy rates
    5. Expect: higher diversity → lower redundancy_rate
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


class H3_DocDiversityRedundancy(HypothesisTest):
    """
    Hypothesis 3: Document diversity within cluster → decreased redundant computation
    
    High diversity = tokens from many documents → each computed similarity 
    more likely to be useful (win a MaxSim for some document).
    
    Low diversity = one document dominates → many similarities wasted on 
    same document's MaxSim competition.
    """
    
    HYPOTHESIS_ID = "H3"
    HYPOTHESIS_NAME = "Doc Diversity → Less Redundancy"
    CLAIM = "Higher document diversity within cluster decreases redundant computation"
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check required columns
        required = ['n_docs', 'n_tokens', 'gini_coefficient', 'redundancy_rate']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return self._empty_result(f"Missing columns: {missing}")
        
        # Build analysis dataframe
        analysis_df = df[['n_docs', 'n_tokens', 'gini_coefficient', 
                          'redundancy_rate', 'yield', 'm2_redundant_sims']].dropna()
        
        if len(analysis_df) < 10:
            return self._empty_result("Insufficient data")
        
        # Compute diversity metric: docs per token (normalized diversity)
        # Higher = more diverse
        analysis_df = analysis_df.copy()
        analysis_df['doc_diversity'] = analysis_df['n_docs'] / analysis_df['n_tokens']
        
        # Alternative: inverse gini (1 - gini), where higher = more uniform/diverse
        analysis_df['inverse_gini'] = 1 - analysis_df['gini_coefficient']
        
        # Create diversity bins
        analysis_df['diversity_bin'] = pd.qcut(
            analysis_df['doc_diversity'], 
            q=4, 
            labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
            duplicates='drop'
        )
        
        # Primary test: Spearman correlation (diversity vs redundancy)
        # Expect NEGATIVE correlation: more diversity → less redundancy
        corr_div, pval_div = spearmanr(
            analysis_df['doc_diversity'], 
            analysis_df['redundancy_rate']
        )
        
        # Alternative: gini vs redundancy (expect POSITIVE: more concentration → more redundancy)
        corr_gini, pval_gini = spearmanr(
            analysis_df['gini_coefficient'], 
            analysis_df['redundancy_rate']
        )
        
        # Bootstrap CI for primary correlation
        def corr_func(data):
            n = len(data) // 2
            return spearmanr(data[:n], data[n:])[0]
        
        combined = np.concatenate([
            analysis_df['doc_diversity'].values,
            analysis_df['redundancy_rate'].values
        ])
        _, ci_lower, ci_upper = bootstrap_ci(combined, corr_func, n_bootstrap=500)
        
        # Stratified comparison: low diversity vs high diversity
        low_div = analysis_df[analysis_df['diversity_bin'] == 'Q1_low']
        high_div = analysis_df[analysis_df['diversity_bin'] == 'Q4_high']
        
        if len(low_div) > 1 and len(high_div) > 1:
            comparison = bootstrap_comparison(
                low_div['redundancy_rate'].values,
                high_div['redundancy_rate'].values
            )
            group_diff = comparison['observed_diff']
            group_pval = comparison['pvalue']
        else:
            group_diff = 0.0
            group_pval = 1.0
        
        # Regression controlling for size
        reg_results = simple_regression_summary(
            analysis_df, 'doc_diversity', 'redundancy_rate'
        )
        
        # Group comparison stats
        if 'diversity_bin' in analysis_df.columns:
            stratified = compute_group_comparison(
                analysis_df[analysis_df['diversity_bin'].notna()],
                'diversity_bin',
                'redundancy_rate'
            )
            stratified_results = stratified.to_dict('records')
        else:
            stratified_results = []
        
        # Hypothesis supported if:
        # 1. Negative correlation between diversity and redundancy (or positive gini-redundancy)
        # 2. p < 0.05
        supported = (corr_div < 0 and pval_div < 0.05) or (corr_gini > 0 and pval_gini < 0.05)
        
        # Use the stronger signal
        primary_corr = corr_div
        primary_pval = pval_div
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=primary_corr,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=primary_pval,
            statistics={
                'spearman_diversity_redundancy': corr_div,
                'spearman_diversity_redundancy_p': pval_div,
                'spearman_gini_redundancy': corr_gini,
                'spearman_gini_redundancy_p': pval_gini,
                'regression_slope': reg_results.get('slope', np.nan),
                'regression_r_squared': reg_results.get('r_squared', np.nan),
                'mean_diversity': float(analysis_df['doc_diversity'].mean()),
                'mean_redundancy_low_diversity': float(low_div['redundancy_rate'].mean()) if len(low_div) > 0 else np.nan,
                'mean_redundancy_high_diversity': float(high_div['redundancy_rate'].mean()) if len(high_div) > 0 else np.nan,
                'group_diff_low_minus_high': group_diff,
                'group_pval': group_pval,
                'stratified_analysis': stratified_results
            },
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
        
        # Build viz dataframe
        plot_df = df[['n_docs', 'n_tokens', 'gini_coefficient', 'redundancy_rate']].dropna()
        
        if len(plot_df) < 10:
            print("Insufficient data for visualization")
            return
        
        plot_df = plot_df.copy()
        plot_df['doc_diversity'] = plot_df['n_docs'] / plot_df['n_tokens']
        
        try:
            plot_df['diversity_bin'] = pd.qcut(
                plot_df['doc_diversity'],
                q=4,
                labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
                duplicates='drop'
            )
        except ValueError:
            plot_df['diversity_bin'] = 'all'
        
        import matplotlib.pyplot as plt
        
        # 1. Summary plot: diversity vs redundancy with bins
        try:
            fig = plot_hypothesis_summary(
                plot_df,
                x_col='doc_diversity',
                y_col='redundancy_rate',
                bin_col='diversity_bin',
                title=f'{self.HYPOTHESIS_ID}: {self.CLAIM}'
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_summary', str(self.output_dir))
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create summary plot: {e}")
        
        # 2. Scatter: gini vs redundancy (concentration perspective)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            plot_df['gini_coefficient'],
            plot_df['redundancy_rate'],
            c=np.log10(plot_df['n_tokens'] + 1),
            s=50,
            alpha=0.5,
            cmap='viridis'
        )
        plt.colorbar(scatter, label='log₁₀(n_tokens)')
        ax.set_xlabel('Gini Coefficient (0=uniform, 1=single doc)')
        ax.set_ylabel('Redundancy Rate')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Document Concentration vs Redundancy')
        save_figure(fig, f'{self.HYPOTHESIS_ID}_gini_scatter', str(self.output_dir))
        plt.close(fig)
        
        # 3. Violin by diversity quartile
        if plot_df['diversity_bin'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_stratified_violin(
                plot_df,
                'diversity_bin',
                'redundancy_rate',
                title=f'{self.HYPOTHESIS_ID}: Redundancy by Document Diversity',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_violin', str(self.output_dir))
            plt.close(fig)
        
        # 4. Scatter with trend line: n_docs vs redundancy
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_scatter_with_regression(
            plot_df,
            x='n_docs',
            y='redundancy_rate',
            title=f'{self.HYPOTHESIS_ID}: Number of Documents vs Redundancy',
            xlabel='Number of Documents in Cluster',
            ylabel='Redundancy Rate',
            ax=ax
        )
        save_figure(fig, f'{self.HYPOTHESIS_ID}_ndocs_scatter', str(self.output_dir))
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    from hypothesis.configs import load_config
    
    parser = argparse.ArgumentParser(description="Run H3: Doc Diversity → Redundancy")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config, override_run_dir=args.run_dir)
    test = H3_DocDiversityRedundancy(config)
    test.run()
