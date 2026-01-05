"""
Hypothesis Testing Template

This module provides a template/example for implementing hypothesis tests
using the shared framework. Copy this file as a starting point for new hypotheses.

Example Hypothesis (H5): 
    Claim: Greater cluster dispersion → increased routing-evidence misses
    
This template demonstrates:
1. Loading standardized tables
2. Defining hypothesis-specific analysis
3. Computing statistics
4. Generating visualizations
5. Producing a structured report
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hypothesis.configs import RuntimeConfig, load_config, ensure_output_dirs
from hypothesis.data import MetricsLoader
from hypothesis.data.standardized_tables import ClusterFrameBuilder, build_all_standardized_tables
from hypothesis.stats import (
    correlation_matrix,
    stratify_by_column,
    compute_group_comparison,
    compute_effect_sizes,
    bootstrap_comparison,
    simple_regression_summary
)
from hypothesis.viz import (
    plot_scatter_with_regression,
    plot_stratified_bars,
    plot_stratified_violin,
    plot_hypothesis_summary,
    plot_correlation_heatmap,
    save_figure
)


# =============================================================================
# Hypothesis Definition
# =============================================================================

@dataclass
class HypothesisResult:
    """Structured result from hypothesis test."""
    hypothesis_id: str
    hypothesis_name: str
    claim: str
    
    # Statistical results
    supported: bool  # Did the test support the hypothesis?
    effect_size: float  # Primary effect size (e.g., correlation)
    effect_size_ci: tuple  # Confidence interval
    p_value: float
    
    # Summary statistics
    statistics: Dict[str, Any]
    
    # Metadata
    config_name: str
    n_observations: int
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, output_dir: str):
        """Save result as JSON."""
        path = Path(output_dir) / f"{self.hypothesis_id}_result.json"
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Saved: {path}")


class HypothesisTest:
    """
    Base class for hypothesis tests.
    
    Override the following methods to implement a hypothesis:
    - setup(): Load and prepare data
    - analyze(): Run statistical tests
    - visualize(): Generate plots
    - report(): Produce summary report
    """
    
    # Override these in subclasses
    HYPOTHESIS_ID = "H0"
    HYPOTHESIS_NAME = "Template Hypothesis"
    CLAIM = "This is a template claim"
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.output_dir = Path(config.paths.output_root) / self.HYPOTHESIS_ID
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data (populated by setup)
        self.cluster_frame: Optional[pd.DataFrame] = None
        self.query_frame: Optional[pd.DataFrame] = None
        
        # Results (populated by analyze)
        self.results: Optional[HypothesisResult] = None
    
    def setup(self):
        """Load and prepare data. Override if custom data needed."""
        ensure_output_dirs(self.config)
        tables = build_all_standardized_tables(self.config)
        self.cluster_frame = tables['cluster_frame']
        self.query_frame = tables.get('query_frame')
        print(f"Loaded cluster_frame: {self.cluster_frame.shape}")
    
    def analyze(self) -> HypothesisResult:
        """Run statistical analysis. MUST override in subclass."""
        raise NotImplementedError("Subclasses must implement analyze()")
    
    def visualize(self):
        """Generate visualizations. Override for custom plots."""
        pass
    
    def report(self) -> str:
        """Generate text report. Override for custom reporting."""
        if self.results is None:
            return "No results available. Run analyze() first."
        
        r = self.results
        report_lines = [
            f"=" * 60,
            f"HYPOTHESIS TEST: {r.hypothesis_id}",
            f"=" * 60,
            f"",
            f"Name: {r.hypothesis_name}",
            f"Claim: {r.claim}",
            f"",
            f"RESULT: {'SUPPORTED' if r.supported else 'NOT SUPPORTED'}",
            f"",
            f"Primary Effect Size: {r.effect_size:.4f}",
            f"95% CI: [{r.effect_size_ci[0]:.4f}, {r.effect_size_ci[1]:.4f}]",
            f"p-value: {r.p_value:.4e}",
            f"",
            f"Observations: {r.n_observations:,}",
            f"Config: {r.config_name}",
            f"Timestamp: {r.timestamp}",
            f"",
            f"Additional Statistics:",
        ]
        
        for key, value in r.statistics.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")
        
        return "\n".join(report_lines)
    
    def run(self) -> HypothesisResult:
        """Execute full hypothesis test pipeline."""
        print(f"\n{'='*60}")
        print(f"Running: {self.HYPOTHESIS_ID} - {self.HYPOTHESIS_NAME}")
        print(f"{'='*60}\n")
        
        print("Step 1: Setup...")
        self.setup()
        
        print("\nStep 2: Analyze...")
        self.results = self.analyze()
        
        print("\nStep 3: Visualize...")
        self.visualize()
        
        print("\nStep 4: Report...")
        report = self.report()
        print(report)
        
        # Save results
        self.results.save(str(self.output_dir))
        
        # Save report
        report_path = self.output_dir / f"{self.HYPOTHESIS_ID}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Saved: {report_path}")
        
        return self.results


# =============================================================================
# Example: Hypothesis 5 (Dispersion → Misses)
# =============================================================================

class H5_DispersionMisses(HypothesisTest):
    """
    Hypothesis 5: Greater cluster dispersion → increased ranking-evidence misses
    
    Mechanism (WARP-specific): WARP relies on centroid-based routing. If a cluster's 
    tokens are very dispersed (high internal variance), the centroid becomes a poor 
    representative of many tokens in that cluster.
    
    Test Plan:
    1. Measure each cluster's internal dispersion (A5)
    2. Stratify clusters into bins (low, medium, high dispersion)
    3. For each bin, compute miss rate (M6) attributable to clusters in that bin
    4. Expect high-dispersion clusters to have higher miss rates
    """
    
    HYPOTHESIS_ID = "H5"
    HYPOTHESIS_NAME = "Dispersion → Misses"
    CLAIM = "Greater cluster dispersion causes increased ranking-evidence misses"
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check required columns exist
        required = ['dispersion', 'm6_miss_rate', 'dispersion_bin']
        missing = [c for c in required if c not in df.columns or df[c].isna().all()]
        if missing:
            print(f"Warning: Missing columns {missing}, using available data")
        
        # Primary analysis: correlation between dispersion and miss rate
        analysis_df = df[['dispersion', 'm6_miss_rate', 'm6_miss_count', 'n_tokens']].dropna()
        
        if len(analysis_df) < 10:
            # Insufficient data - return null result
            return HypothesisResult(
                hypothesis_id=self.HYPOTHESIS_ID,
                hypothesis_name=self.HYPOTHESIS_NAME,
                claim=self.CLAIM,
                supported=False,
                effect_size=0.0,
                effect_size_ci=(0.0, 0.0),
                p_value=1.0,
                statistics={'error': 'Insufficient data'},
                config_name=self.config.name,
                n_observations=len(analysis_df),
                timestamp=datetime.now().isoformat()
            )
        
        # Spearman correlation (robust to outliers)
        from scipy.stats import spearmanr
        corr, pval = spearmanr(analysis_df['dispersion'], analysis_df['m6_miss_rate'])
        
        # Bootstrap CI for correlation
        from hypothesis.stats import bootstrap_ci
        
        def corr_func(data):
            n = len(data) // 2
            return spearmanr(data[:n], data[n:])[0]
        
        combined = np.concatenate([
            analysis_df['dispersion'].values, 
            analysis_df['m6_miss_rate'].values
        ])
        _, ci_lower, ci_upper = bootstrap_ci(combined, corr_func, n_bootstrap=500)
        
        # Stratified analysis by dispersion quartile
        if 'dispersion_bin' in df.columns:
            stratified = compute_group_comparison(
                df[df['dispersion_bin'].notna()],
                'dispersion_bin',
                'm6_miss_rate'
            )
            stratified_results = stratified.to_dict('records')
        else:
            stratified_results = []
        
        # Regression: dispersion → miss_rate controlling for size
        reg_results = simple_regression_summary(analysis_df, 'dispersion', 'm6_miss_rate')
        
        # Determine if hypothesis is supported
        # Criteria: positive correlation with p < 0.05
        supported = corr > 0 and pval < 0.05
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=corr,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=pval,
            statistics={
                'spearman_correlation': corr,
                'regression_slope': reg_results.get('slope', np.nan),
                'regression_r_squared': reg_results.get('r_squared', np.nan),
                'n_clusters_with_misses': int((analysis_df['m6_miss_count'] > 0).sum()),
                'mean_dispersion': float(analysis_df['dispersion'].mean()),
                'mean_miss_rate': float(analysis_df['m6_miss_rate'].mean()),
                'stratified_analysis': stratified_results
            },
            config_name=self.config.name,
            n_observations=len(analysis_df),
            timestamp=datetime.now().isoformat()
        )
    
    def visualize(self):
        df = self.cluster_frame
        
        # Filter to valid data
        plot_df = df[['dispersion', 'm6_miss_rate', 'dispersion_bin', 'n_tokens']].dropna()
        
        if len(plot_df) < 10:
            print("Insufficient data for visualization")
            return
        
        import matplotlib.pyplot as plt
        
        # 1. Summary plot
        if 'dispersion_bin' in plot_df.columns:
            fig = plot_hypothesis_summary(
                plot_df,
                x_col='dispersion',
                y_col='m6_miss_rate',
                bin_col='dispersion_bin',
                title=f'{self.HYPOTHESIS_ID}: {self.CLAIM}'
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_summary', str(self.output_dir))
            plt.close(fig)
        
        # 2. Scatter with regression, sized by cluster size
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            plot_df['dispersion'],
            plot_df['m6_miss_rate'],
            c=np.log10(plot_df['n_tokens'] + 1),
            s=50,
            alpha=0.5,
            cmap='viridis'
        )
        plt.colorbar(scatter, label='log₁₀(n_tokens)')
        ax.set_xlabel('Dispersion (within-cluster)')
        ax.set_ylabel('Miss Rate')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Dispersion vs Miss Rate (colored by cluster size)')
        
        save_figure(fig, f'{self.HYPOTHESIS_ID}_scatter_sized', str(self.output_dir))
        plt.close(fig)
        
        # 3. Violin plot by dispersion quartile
        if 'dispersion_bin' in plot_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_stratified_violin(
                plot_df,
                'dispersion_bin',
                'm6_miss_rate',
                title=f'{self.HYPOTHESIS_ID}: Miss Rate by Dispersion Quartile',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_violin', str(self.output_dir))
            plt.close(fig)


# =============================================================================
# Example: Hypothesis 4 (Doc Concentration → Redundancy)
# =============================================================================

class H4_ConcentrationRedundancy(HypothesisTest):
    """
    Hypothesis 4: Clusters dominated by a single document → concentration of redundant work
    
    Mechanism (WARP-specific): When one document's tokens make up a large fraction of a 
    cluster, routing can still select that cluster, but the work is essentially over-counting 
    for one document.
    
    Test Plan:
    1. Identify clusters with high document concentration (top-1 doc share > threshold)
    2. Compare contribution to redundant computation (M2) between concentrated and diverse clusters
    3. Expect highly concentrated clusters to contribute disproportionately to M2
    """
    
    HYPOTHESIS_ID = "H4"
    HYPOTHESIS_NAME = "Doc Concentration → Redundancy"
    CLAIM = "Clusters dominated by a single document have concentrated redundant work"
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check required columns
        if 'top_1_doc_share' not in df.columns or 'm2_redundant_sims' not in df.columns:
            return HypothesisResult(
                hypothesis_id=self.HYPOTHESIS_ID,
                hypothesis_name=self.HYPOTHESIS_NAME,
                claim=self.CLAIM,
                supported=False,
                effect_size=0.0,
                effect_size_ci=(0.0, 0.0),
                p_value=1.0,
                statistics={'error': 'Required columns not available'},
                config_name=self.config.name,
                n_observations=0,
                timestamp=datetime.now().isoformat()
            )
        
        # Analysis
        analysis_df = df[['top_1_doc_share', 'top_5_doc_share', 'm2_redundant_sims', 
                          'redundancy_rate', 'n_tokens', 'is_single_doc_dominated']].dropna()
        
        if len(analysis_df) < 10:
            return self._empty_result("Insufficient data")
        
        # Spearman correlation
        from scipy.stats import spearmanr
        corr, pval = spearmanr(analysis_df['top_1_doc_share'], analysis_df['redundancy_rate'])
        
        # Compare concentrated vs diverse clusters
        concentrated = analysis_df[analysis_df['is_single_doc_dominated']]
        diverse = analysis_df[~analysis_df['is_single_doc_dominated']]
        
        if len(concentrated) > 1 and len(diverse) > 1:
            comparison = bootstrap_comparison(
                concentrated['redundancy_rate'].values,
                diverse['redundancy_rate'].values
            )
            group_diff = comparison['observed_diff']
            group_pval = comparison['pvalue']
        else:
            group_diff = 0.0
            group_pval = 1.0
        
        # Contribution to total M2
        total_m2 = analysis_df['m2_redundant_sims'].sum()
        concentrated_m2_share = concentrated['m2_redundant_sims'].sum() / total_m2 if total_m2 > 0 else 0
        concentrated_cluster_share = len(concentrated) / len(analysis_df)
        
        # Disproportionate = concentrated clusters contribute more M2 than their count share
        disproportionate = concentrated_m2_share > concentrated_cluster_share
        
        supported = corr > 0 and pval < 0.05 and disproportionate
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=corr,
            effect_size_ci=(0.0, 0.0),  # TODO: Bootstrap CI
            p_value=pval,
            statistics={
                'spearman_correlation': corr,
                'n_concentrated_clusters': len(concentrated),
                'n_diverse_clusters': len(diverse),
                'concentrated_m2_share': concentrated_m2_share,
                'concentrated_cluster_share': concentrated_cluster_share,
                'disproportionate_contribution': disproportionate,
                'mean_redundancy_concentrated': concentrated['redundancy_rate'].mean() if len(concentrated) > 0 else 0,
                'mean_redundancy_diverse': diverse['redundancy_rate'].mean() if len(diverse) > 0 else 0,
                'group_diff': group_diff,
                'group_pval': group_pval
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
        
        plot_df = df[['top_1_doc_share', 'redundancy_rate', 'is_single_doc_dominated', 'n_tokens']].dropna()
        
        if len(plot_df) < 10:
            return
        
        import matplotlib.pyplot as plt
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_scatter_with_regression(
            plot_df,
            x='top_1_doc_share',
            y='redundancy_rate',
            title=f'{self.HYPOTHESIS_ID}: Document Concentration vs Redundancy Rate',
            xlabel='Top-1 Document Share',
            ylabel='Redundancy Rate',
            ax=ax
        )
        save_figure(fig, f'{self.HYPOTHESIS_ID}_scatter', str(self.output_dir))
        plt.close(fig)
        
        # Comparison bars - convert boolean to string for seaborn compatibility
        if 'is_single_doc_dominated' in plot_df.columns:
            plot_df = plot_df.copy()
            plot_df['concentration_type'] = plot_df['is_single_doc_dominated'].map(
                {True: 'Single-Doc', False: 'Multi-Doc'}
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_stratified_bars(
                plot_df,
                'concentration_type',
                'redundancy_rate',
                title=f'{self.HYPOTHESIS_ID}: Redundancy by Concentration Type',
                xlabel='Concentration Type',
                ylabel='Redundancy Rate',
                ax=ax
            )
            save_figure(fig, f'{self.HYPOTHESIS_ID}_comparison', str(self.output_dir))
            plt.close(fig)


# =============================================================================
# Import Additional Hypotheses
# =============================================================================

from hypothesis.hypotheses.h3_doc_diversity import H3_DocDiversityRedundancy
from hypothesis.hypotheses.h10_hubness_redundancy import H10_HubnessRedundancy
from hypothesis.hypotheses.h15_miss_severity import H15_MissSeverity
from hypothesis.hypotheses.h17_borderline_clusters import H17_BorderlineClusters


# =============================================================================
# CLI Entry Point
# =============================================================================

AVAILABLE_HYPOTHESES = {
    'H3': H3_DocDiversityRedundancy,
    'H4': H4_ConcentrationRedundancy,
    'H5': H5_DispersionMisses,
    'H10': H10_HubnessRedundancy,
    'H15': H15_MissSeverity,
    'H17': H17_BorderlineClusters,
}


def run_hypothesis(hypothesis_id: str, config_name: str = "dev", run_dir: str = None):
    """
    Run a specific hypothesis test.
    
    Args:
        hypothesis_id: Hypothesis identifier (e.g., "H5")
        config_name: Config to use ("smoke", "dev", or "prod")
        run_dir: Optional override for measurement run directory
        
    Returns:
        HypothesisResult
    """
    if hypothesis_id not in AVAILABLE_HYPOTHESES:
        raise ValueError(f"Unknown hypothesis: {hypothesis_id}. Available: {list(AVAILABLE_HYPOTHESES.keys())}")
    
    config = load_config(config_name, override_run_dir=run_dir)
    hypothesis_class = AVAILABLE_HYPOTHESES[hypothesis_id]
    
    test = hypothesis_class(config)
    return test.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hypothesis tests")
    parser.add_argument("hypothesis", choices=list(AVAILABLE_HYPOTHESES.keys()) + ['all'],
                       help="Hypothesis to test (or 'all')")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev",
                       help="Configuration to use")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Override measurement run directory")
    args = parser.parse_args()
    
    if args.hypothesis == 'all':
        for h_id in AVAILABLE_HYPOTHESES:
            print(f"\n{'#'*60}")
            print(f"# Running {h_id}")
            print(f"{'#'*60}")
            try:
                run_hypothesis(h_id, args.config, args.run_dir)
            except Exception as e:
                print(f"Error running {h_id}: {e}")
    else:
        run_hypothesis(args.hypothesis, args.config, args.run_dir)
