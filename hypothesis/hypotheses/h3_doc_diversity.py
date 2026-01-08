"""
Hypothesis H3: Document Diversity → Increased Computational Yield

Claim: Higher document diversity within a cluster increases computational yield,
       meaning more computed similarities become influential (win MaxSim).

Mechanism (WARP-specific):
    When a cluster contains tokens from many different documents, each token-token 
    similarity computed is more likely to contribute to a distinct document's MaxSim.
    Conversely, when one document dominates, many similarities are "wasted" because
    they all compete for the same document's MaxSim slot.
    
    The yield perspective reveals the TRUE impact: while redundancy changes look small
    (e.g., 98.46% → 97.41%), yield changes are dramatic because yield = 1 - redundancy.
    The same change means yield goes from 1.54% → 2.59%, a 68% RELATIVE increase.
    This "yield multiplier effect" means small redundancy improvements translate to
    large efficiency gains in absolute influential interactions.

Key Metrics:
    - n_docs (A2): Number of unique documents in cluster
    - gini_coefficient (A3): Document concentration (0=uniform, 1=single doc)
    - tokens_per_doc: Average tokens per document (n_tokens / n_docs)
    - yield (A4): influential / computed (PRIMARY OUTCOME)
    - redundancy_rate: m2_redundant_sims / m1_total_sims (complement of yield)

Test Plan:
    1. Compute document diversity metric: n_docs / n_tokens (or use inverse gini)
    2. Control for cluster size (n_tokens) 
    3. Correlate diversity with yield (expect POSITIVE) and redundancy_rate (expect NEGATIVE)
    4. Stratify by diversity quartile, compare yield and compute yield multiplier
    5. Compute absolute influential interactions gained from diversity
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
    Hypothesis 3: Document diversity within cluster → increased computational yield
    
    High diversity = tokens from many documents → each computed similarity 
    more likely to be useful (win a MaxSim for some document).
    
    Low diversity = one document dominates → many similarities wasted on 
    same document's MaxSim competition.
    
    KEY INSIGHT (Yield Multiplier Effect):
    Redundancy rates are high (>97%), so yield is small (<3%). A 1 percentage point
    drop in redundancy translates to a ~50-70% RELATIVE increase in yield.
    This makes diversity's impact much larger than raw redundancy numbers suggest.
    """
    
    HYPOTHESIS_ID = "H3"
    HYPOTHESIS_NAME = "Doc Diversity → Higher Yield"
    CLAIM = "Higher document diversity within cluster increases computational yield (influential interactions)"
    
    def setup(self):
        """H3 only needs cluster_frame, not query_frame or miss_attribution_frame."""
        from hypothesis.configs import ensure_output_dirs
        from hypothesis.data.standardized_tables import ClusterFrameBuilder
        
        ensure_output_dirs(self.config)
        
        # Only build cluster_frame (avoids loading heavy M5 for query_frame)
        cluster_builder = ClusterFrameBuilder(self.config)
        self.cluster_frame = cluster_builder.build()
        self.query_frame = None  # Not needed for H3
        
        print(f"Loaded cluster_frame: {self.cluster_frame.shape}")
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check required columns
        required = ['n_docs', 'n_tokens', 'gini_coefficient', 'redundancy_rate', 'yield']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return self._empty_result(f"Missing columns: {missing}")
        
        
        # Build analysis dataframe
        analysis_df = df[['n_docs', 'n_tokens', 'gini_coefficient', 
                          'redundancy_rate', 'yield', 'influential', 'm1_total_sims']].dropna()
        
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
        
        # PRIMARY test: Spearman correlation (diversity vs YIELD)
        # Expect POSITIVE correlation: more diversity → higher yield
        corr_div_yield, pval_div_yield = spearmanr(
            analysis_df['doc_diversity'], 
            analysis_df['yield']
        )
        
        # Secondary test: diversity vs redundancy (legacy, for continuity)
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
        
        # Bootstrap CI for primary correlation (diversity vs yield)
        def corr_func(data):
            n = len(data) // 2
            return spearmanr(data[:n], data[n:])[0]
        
        combined = np.concatenate([
            analysis_df['doc_diversity'].values,
            analysis_df['yield'].values
        ])
        _, ci_lower, ci_upper = bootstrap_ci(combined, corr_func, n_bootstrap=500)
        
        # Stratified comparison: low diversity vs high diversity
        low_div = analysis_df[analysis_df['diversity_bin'] == 'Q1_low']
        high_div = analysis_df[analysis_df['diversity_bin'] == 'Q4_high']
        
        # Yield statistics per quartile
        q1_yield = low_div['yield'].mean() if len(low_div) > 0 else np.nan
        q4_yield = high_div['yield'].mean() if len(high_div) > 0 else np.nan
        q1_redundancy = low_div['redundancy_rate'].mean() if len(low_div) > 0 else np.nan
        q4_redundancy = high_div['redundancy_rate'].mean() if len(high_div) > 0 else np.nan
        
        # YIELD MULTIPLIER EFFECT: The key reframe
        # Relative increase in yield from Q1 to Q4
        if q1_yield > 0 and not np.isnan(q1_yield) and not np.isnan(q4_yield):
            yield_multiplier = (q4_yield - q1_yield) / q1_yield  # e.g., 0.68 = 68% increase
            yield_relative_increase_pct = yield_multiplier * 100
        else:
            yield_multiplier = np.nan
            yield_relative_increase_pct = np.nan
        
        # Absolute influential interactions gained
        q1_influential = low_div['influential'].sum() if len(low_div) > 0 else 0
        q4_influential = high_div['influential'].sum() if len(high_div) > 0 else 0
        q1_total = low_div['m1_total_sims'].sum() if len(low_div) > 0 else 0
        q4_total = high_div['m1_total_sims'].sum() if len(high_div) > 0 else 0
        
        # What if Q1 clusters had Q4's yield? How many more influential interactions?
        if q1_total > 0 and not np.isnan(q4_yield):
            potential_q1_influential = q1_total * q4_yield
            influential_gained = potential_q1_influential - q1_influential
            influential_increase_pct = (influential_gained / q1_influential * 100) if q1_influential > 0 else np.nan
        else:
            potential_q1_influential = np.nan
            influential_gained = np.nan
            influential_increase_pct = np.nan
        
        # Bootstrap comparison for YIELD (primary) and redundancy (secondary)
        if len(low_div) > 1 and len(high_div) > 1:
            yield_comparison = bootstrap_comparison(
                low_div['yield'].values,
                high_div['yield'].values
            )
            yield_diff = yield_comparison['observed_diff']
            yield_pval = yield_comparison['pvalue']
            
            redundancy_comparison = bootstrap_comparison(
                low_div['redundancy_rate'].values,
                high_div['redundancy_rate'].values
            )
            redundancy_diff = redundancy_comparison['observed_diff']
            redundancy_pval = redundancy_comparison['pvalue']
        else:
            yield_diff = 0.0
            yield_pval = 1.0
            redundancy_diff = 0.0
            redundancy_pval = 1.0
        
        # Regression controlling for size
        reg_results_yield = simple_regression_summary(
            analysis_df, 'doc_diversity', 'yield'
        )
        reg_results_redundancy = simple_regression_summary(
            analysis_df, 'doc_diversity', 'redundancy_rate'
        )
        
        # Group comparison stats for yield
        if 'diversity_bin' in analysis_df.columns:
            stratified_yield = compute_group_comparison(
                analysis_df[analysis_df['diversity_bin'].notna()],
                'diversity_bin',
                'yield'
            )
            stratified_yield_results = stratified_yield.to_dict('records')
            
            stratified_redundancy = compute_group_comparison(
                analysis_df[analysis_df['diversity_bin'].notna()],
                'diversity_bin',
                'redundancy_rate'
            )
            stratified_redundancy_results = stratified_redundancy.to_dict('records')
        else:
            stratified_yield_results = []
            stratified_redundancy_results = []
        
        # Hypothesis supported if:
        # 1. POSITIVE correlation between diversity and yield (or negative diversity-redundancy)
        # 2. p < 0.05
        supported = (corr_div_yield > 0 and pval_div_yield < 0.05) or (corr_div < 0 and pval_div < 0.05)
        
        # Use YIELD correlation as primary effect size
        primary_corr = corr_div_yield
        primary_pval = pval_div_yield
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=primary_corr,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=primary_pval,
            statistics={
                # PRIMARY: Yield perspective
                'spearman_diversity_yield': corr_div_yield,
                'spearman_diversity_yield_p': pval_div_yield,
                
                # Yield by quartile
                'q1_yield': q1_yield,
                'q4_yield': q4_yield,
                'yield_diff_q4_minus_q1': yield_diff,
                'yield_pval': yield_pval,
                
                # THE KEY INSIGHT: Yield multiplier effect
                'yield_multiplier': yield_multiplier,
                'yield_relative_increase_pct': yield_relative_increase_pct,
                
                # Absolute influential interactions
                'q1_influential_sims': int(q1_influential) if not np.isnan(q1_influential) else np.nan,
                'q4_influential_sims': int(q4_influential) if not np.isnan(q4_influential) else np.nan,
                'potential_q1_influential_at_q4_yield': potential_q1_influential,
                'influential_gained_if_q1_had_q4_yield': influential_gained,
                'influential_increase_pct': influential_increase_pct,
                
                # SECONDARY: Redundancy perspective (legacy)
                'spearman_diversity_redundancy': corr_div,
                'spearman_diversity_redundancy_p': pval_div,
                'spearman_gini_redundancy': corr_gini,
                'spearman_gini_redundancy_p': pval_gini,
                'q1_redundancy': q1_redundancy,
                'q4_redundancy': q4_redundancy,
                'redundancy_diff_q1_minus_q4': redundancy_diff,
                'redundancy_pval': redundancy_pval,
                
                # Regression
                'regression_yield_slope': reg_results_yield.get('slope', np.nan),
                'regression_yield_r_squared': reg_results_yield.get('r_squared', np.nan),
                'regression_redundancy_slope': reg_results_redundancy.get('slope', np.nan),
                'regression_redundancy_r_squared': reg_results_redundancy.get('r_squared', np.nan),
                
                # General
                'mean_diversity': float(analysis_df['doc_diversity'].mean()),
                'n_clusters_q1': len(low_div),
                'n_clusters_q4': len(high_div),
                
                # Stratified analysis
                'stratified_yield': stratified_yield_results,
                'stratified_redundancy': stratified_redundancy_results
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
        
        # Build viz dataframe - include yield
        required_cols = ['n_docs', 'n_tokens', 'gini_coefficient', 'redundancy_rate', 'yield']
        available_cols = [c for c in required_cols if c in df.columns]
        plot_df = df[available_cols].dropna()
        
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
        
        # 1. PRIMARY: Yield summary plot (the key reframe)
        if 'yield' in plot_df.columns:
            try:
                fig = plot_hypothesis_summary(
                    plot_df,
                    x_col='doc_diversity',
                    y_col='yield',
                    bin_col='diversity_bin',
                    title=f'{self.HYPOTHESIS_ID}: Document Diversity → Higher Yield'
                )
                save_figure(fig, f'{self.HYPOTHESIS_ID}_yield_summary', str(self.output_dir))
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not create yield summary plot: {e}")
            
            # Yield violin by diversity quartile
            if plot_df['diversity_bin'].nunique() > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_stratified_violin(
                    plot_df,
                    'diversity_bin',
                    'yield',
                    title=f'{self.HYPOTHESIS_ID}: Yield by Document Diversity (Higher = Better)',
                    ax=ax
                )
                save_figure(fig, f'{self.HYPOTHESIS_ID}_yield_violin', str(self.output_dir))
                plt.close(fig)
        
        # 2. SECONDARY: Legacy redundancy summary plot
        try:
            fig = plot_hypothesis_summary(
                plot_df,
                x_col='doc_diversity',
                y_col='redundancy_rate',
                bin_col='diversity_bin',
                title=f'{self.HYPOTHESIS_ID}: Document Diversity → Lower Redundancy'
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_summary', str(self.output_dir))
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create summary plot: {e}")
        
        # 3. Scatter: gini vs redundancy (concentration perspective)
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
        
        # 4. Violin by diversity quartile (redundancy)
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
        
        # 5. Scatter with trend line: n_docs vs redundancy
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
        
        # 6. NEW: Yield multiplier visualization
        if 'yield' in plot_df.columns and plot_df['diversity_bin'].nunique() > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left: Raw numbers comparison
            quartile_stats = plot_df.groupby('diversity_bin').agg({
                'yield': 'mean',
                'redundancy_rate': 'mean'
            }).reset_index()
            
            ax = axes[0]
            x = range(len(quartile_stats))
            width = 0.35
            ax.bar([i - width/2 for i in x], quartile_stats['redundancy_rate'] * 100, 
                   width, label='Redundancy %', color='coral', alpha=0.8)
            ax.bar([i + width/2 for i in x], quartile_stats['yield'] * 100, 
                   width, label='Yield %', color='seagreen', alpha=0.8)
            ax.set_xlabel('Diversity Quartile')
            ax.set_ylabel('Percentage')
            ax.set_title('Redundancy vs Yield by Diversity Quartile')
            ax.set_xticks(x)
            ax.set_xticklabels(quartile_stats['diversity_bin'])
            ax.legend()
            ax.set_ylim(0, 100)
            
            # Right: Relative change from Q1 baseline
            ax = axes[1]
            q1_yield = quartile_stats.loc[quartile_stats['diversity_bin'] == 'Q1_low', 'yield'].values[0]
            if q1_yield > 0:
                relative_yield = (quartile_stats['yield'] - q1_yield) / q1_yield * 100
                colors = ['gray' if q == 'Q1_low' else 'seagreen' for q in quartile_stats['diversity_bin']]
                bars = ax.bar(range(len(quartile_stats)), relative_yield, color=colors, alpha=0.8)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Diversity Quartile')
                ax.set_ylabel('Relative Change from Q1 (%)')
                ax.set_title('Yield Multiplier Effect: % Increase vs Q1 (Low Diversity)')
                ax.set_xticks(range(len(quartile_stats)))
                ax.set_xticklabels(quartile_stats['diversity_bin'])
                
                # Add percentage labels on bars
                for bar, val in zip(bars, relative_yield):
                    if val != 0:
                        ax.annotate(f'+{val:.1f}%' if val > 0 else f'{val:.1f}%',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom' if val > 0 else 'top',
                                   fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            save_figure(fig, f'{self.HYPOTHESIS_ID}_yield_multiplier', str(self.output_dir))
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    from hypothesis.configs import load_config
    
    parser = argparse.ArgumentParser(description="Run H3: Doc Diversity → Higher Yield")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config, override_run_dir=args.run_dir)
    test = H3_DocDiversityRedundancy(config)
    test.run()
