"""
Hypothesis H15: High-Similarity Token Clusters → More Severe Misses

Claim: Clusters containing high-similarity tokens have more severe misses 
       (larger score_delta) when not selected.

Mechanism (WARP-specific):
    Some clusters may contain tokens that are "high-quality" - when they participate
    in MaxSim computation, they achieve high scores. If such a cluster is not selected
    by routing, the miss is severe because the oracle score (what we would have gotten)
    is much higher than the observed score.
    
    The severity of a miss is measured by score_delta = oracle_score - observed_score.
    If a cluster consistently has high score_delta when missed, it means the cluster
    contains tokens that would have contributed significantly to scoring.

Key Metrics (from M5 aggregation by centroid):
    - mean_score_delta: Average miss severity for this centroid
    - max_score_delta: Worst-case miss severity
    - miss_severity_std: Variability in miss severity
    - oracle_score_mean: Average oracle score (what we would have gotten)
    
Centroid properties to correlate:
    - dispersion (A5): Tight clusters might have more consistent high scores
    - n_tokens (A1): Larger clusters have more potential high-scorers
    - yield (A4): Low yield might indicate routing mismatch

Test Plan:
    1. Aggregate M5 by oracle_centroid_id to get per-centroid miss severity
    2. Join with cluster_frame to get centroid properties
    3. Correlate centroid properties with mean/max score_delta
    4. Identify which properties predict severe misses
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from hypothesis.configs import ensure_output_dirs
from hypothesis.data import MetricsLoader
from hypothesis.data.standardized_tables import ClusterFrameBuilder, build_all_standardized_tables
from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.stats import (
    correlation_matrix,
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
    plot_correlation_heatmap,
    save_figure
)


class H15_MissSeverity(HypothesisTest):
    """
    Hypothesis 15: Clusters with high-quality tokens → more severe misses
    
    This hypothesis explores WHICH centroid properties predict the severity
    of misses when that centroid is not selected by routing.
    
    Key insight: A miss from a high-quality cluster hurts more than a miss
    from a low-quality cluster.
    """
    
    HYPOTHESIS_ID = "H15"
    HYPOTHESIS_NAME = "Token Quality → Miss Severity"
    CLAIM = "Clusters with high-similarity tokens have more severe misses when not selected"
    
    def setup(self):
        """Extended setup: load M5 and aggregate by centroid using chunked streaming."""
        ensure_output_dirs(self.config)
        
        # Load base tables
        tables = build_all_standardized_tables(self.config)
        self.cluster_frame = tables['cluster_frame']
        self.query_frame = tables.get('query_frame')
        
        # Load M5 via chunked reader and aggregate by centroid
        self.loader = MetricsLoader(
            index_path=self.config.paths.index_path,
            run_dir=self.config.paths.run_dir,
            chunk_size=self.config.processing.m4_chunk_size,
            verbose=self.config.verbose
        )
        
        try:
            self.centroid_severity = self._aggregate_m5_chunked()
            if len(self.centroid_severity) > 0:
                print(f"Aggregated M5 severity for {len(self.centroid_severity)} centroids")
            else:
                print("Warning: No misses found in M5")
                
        except FileNotFoundError:
            print("Warning: M5 not available, using m6_miss_rate as severity proxy")
            self.centroid_severity = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Could not load M5: {e}")
            import traceback
            traceback.print_exc()
            self.centroid_severity = pd.DataFrame()
        
        print(f"Loaded cluster_frame: {self.cluster_frame.shape}")
    
    def _aggregate_m5_chunked(self) -> pd.DataFrame:
        """
        Aggregate M5 by centroid using streaming chunks.
        
        Uses online aggregation to compute mean, max, std, count without
        loading the entire file into memory.
        """
        m5_reader = self.loader.get_m5_reader()
        
        # Track running statistics per centroid using Welford's online algorithm
        # For each centroid: count, mean, M2 (for variance), max
        centroid_stats: Dict[int, Dict] = {}
        
        columns_to_read = ['oracle_centroid_id', 'is_miss', 'score_delta', 'oracle_score']
        
        total_misses = 0
        total_rows = 0
        n_chunks = m5_reader.num_chunks
        
        print(f"Processing M5: {m5_reader.num_queries} queries in {n_chunks} chunks...")
        
        # Use tqdm with chunk count and show running stats
        pbar = tqdm(
            m5_reader.iter_chunks(columns=columns_to_read),
            total=n_chunks,
            desc="Aggregating M5",
            unit="chunk"
        )
        
        for chunk in pbar:
            total_rows += len(chunk)
            
            # Filter to actual misses
            if 'is_miss' in chunk.columns:
                chunk = chunk[chunk['is_miss'] == True]
            
            chunk_misses = len(chunk)
            total_misses += chunk_misses
            
            # Update progress bar with stats
            pbar.set_postfix({
                'rows': f'{total_rows:,}',
                'misses': f'{total_misses:,}',
                'centroids': len(centroid_stats)
            })
            
            if chunk_misses == 0:
                continue
            
            # Update running stats for each centroid
            for centroid_id, group in chunk.groupby('oracle_centroid_id'):
                deltas = group['score_delta'].values
                oracle_scores = group['oracle_score'].values if 'oracle_score' in group.columns else None
                
                if centroid_id not in centroid_stats:
                    centroid_stats[centroid_id] = {
                        'count': 0,
                        'mean': 0.0,
                        'M2': 0.0,  # For Welford's variance
                        'max_delta': float('-inf'),
                        'oracle_sum': 0.0,
                        'oracle_max': float('-inf'),
                        'oracle_count': 0,
                    }
                
                stats = centroid_stats[centroid_id]
                
                # Update count, mean, M2 using Welford's online algorithm
                for delta in deltas:
                    stats['count'] += 1
                    delta_from_mean = delta - stats['mean']
                    stats['mean'] += delta_from_mean / stats['count']
                    delta_from_new_mean = delta - stats['mean']
                    stats['M2'] += delta_from_mean * delta_from_new_mean
                    stats['max_delta'] = max(stats['max_delta'], delta)
                
                # Update oracle score stats
                if oracle_scores is not None:
                    for os in oracle_scores:
                        if not np.isnan(os):
                            stats['oracle_sum'] += os
                            stats['oracle_count'] += 1
                            stats['oracle_max'] = max(stats['oracle_max'], os)
        
        pbar.close()
        print(f"Processed {total_rows:,} rows, {total_misses:,} misses across {len(centroid_stats)} centroids")
        
        if not centroid_stats:
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for centroid_id, stats in centroid_stats.items():
            count = stats['count']
            std = np.sqrt(stats['M2'] / count) if count > 1 else 0.0
            
            record = {
                'centroid_id': centroid_id,
                'mean_score_delta': stats['mean'],
                'max_score_delta': stats['max_delta'],
                'score_delta_std': std,
                'miss_count_m5': count,
            }
            
            if stats['oracle_count'] > 0:
                record['mean_oracle_score'] = stats['oracle_sum'] / stats['oracle_count']
                record['max_oracle_score'] = stats['oracle_max']
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def analyze(self) -> HypothesisResult:
        df = self.cluster_frame
        
        # Check if we have severity data
        if len(self.centroid_severity) == 0:
            # Fallback: use m6_miss_rate correlation with properties
            return self._analyze_fallback(df)
        
        # Join severity data with cluster properties
        analysis_df = df.merge(
            self.centroid_severity,
            on='centroid_id',
            how='inner'
        )
        
        if len(analysis_df) < 10:
            return self._empty_result("Insufficient data after joining severity metrics")
        
        # Properties to correlate with miss severity
        property_cols = ['dispersion', 'n_tokens', 'n_docs', 'yield', 
                        'gini_coefficient', 'top_1_doc_share', 'sel_freq']
        available_props = [c for c in property_cols if c in analysis_df.columns]
        
        # Primary analysis: what predicts mean_score_delta?
        correlations = {}
        for prop in available_props:
            valid = analysis_df[[prop, 'mean_score_delta']].dropna()
            if len(valid) > 10:
                corr, pval = spearmanr(valid[prop], valid['mean_score_delta'])
                correlations[prop] = {'correlation': corr, 'p_value': pval}
        
        # Find strongest predictor
        if correlations:
            best_predictor = max(correlations.keys(), 
                               key=lambda k: abs(correlations[k]['correlation']))
            best_corr = correlations[best_predictor]['correlation']
            best_pval = correlations[best_predictor]['p_value']
        else:
            best_predictor = 'none'
            best_corr = 0.0
            best_pval = 1.0
        
        # Bootstrap CI for best predictor
        if best_predictor != 'none':
            valid = analysis_df[[best_predictor, 'mean_score_delta']].dropna()
            
            def corr_func(data):
                n = len(data) // 2
                return spearmanr(data[:n], data[n:])[0]
            
            combined = np.concatenate([
                valid[best_predictor].values,
                valid['mean_score_delta'].values
            ])
            _, ci_lower, ci_upper = bootstrap_ci(combined, corr_func, n_bootstrap=500)
        else:
            ci_lower, ci_upper = 0.0, 0.0
        
        # Stratified analysis: dispersion bins vs severity
        if 'dispersion' in analysis_df.columns:
            analysis_df = analysis_df.copy()
            try:
                analysis_df['dispersion_bin'] = pd.qcut(
                    analysis_df['dispersion'],
                    q=4,
                    labels=['Q1_tight', 'Q2', 'Q3', 'Q4_dispersed'],
                    duplicates='drop'
                )
                stratified = compute_group_comparison(
                    analysis_df[analysis_df['dispersion_bin'].notna()],
                    'dispersion_bin',
                    'mean_score_delta'
                )
                stratified_results = stratified.to_dict('records')
            except (ValueError, KeyError):
                stratified_results = []
        else:
            stratified_results = []
        
        # Summary statistics
        summary_stats = {
            'mean_score_delta_overall': float(analysis_df['mean_score_delta'].mean()),
            'max_score_delta_overall': float(analysis_df['max_score_delta'].max()),
            'std_score_delta_overall': float(analysis_df['mean_score_delta'].std()),
            'n_centroids_with_misses': len(analysis_df),
            'best_predictor': best_predictor,
            'all_correlations': correlations,
            'stratified_by_dispersion': stratified_results,
        }
        
        # Add oracle score stats if available
        if 'mean_oracle_score' in analysis_df.columns:
            summary_stats['mean_oracle_score'] = float(analysis_df['mean_oracle_score'].mean())
            summary_stats['max_oracle_score'] = float(analysis_df['max_oracle_score'].max())
        
        # Hypothesis supported if any property significantly correlates with severity
        significant_predictors = [k for k, v in correlations.items() 
                                 if v['p_value'] < 0.05 and abs(v['correlation']) > 0.1]
        supported = len(significant_predictors) > 0
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=best_corr,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=best_pval,
            statistics=summary_stats,
            config_name=self.config.name,
            n_observations=len(analysis_df),
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_fallback(self, df: pd.DataFrame) -> HypothesisResult:
        """Fallback analysis using m6_miss_rate when M5 not available."""
        if 'm6_miss_rate' not in df.columns:
            return self._empty_result("Neither M5 nor M6 available for severity analysis")
        
        analysis_df = df[['dispersion', 'n_tokens', 'm6_miss_rate', 'm6_miss_count']].dropna()
        
        if len(analysis_df) < 10:
            return self._empty_result("Insufficient data for fallback analysis")
        
        # Use miss count as severity proxy (more misses = more severe impact)
        corr, pval = spearmanr(analysis_df['dispersion'], analysis_df['m6_miss_count'])
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=corr > 0 and pval < 0.05,
            effect_size=corr,
            effect_size_ci=(0.0, 0.0),
            p_value=pval,
            statistics={
                'fallback_mode': True,
                'note': 'M5 not available, using m6_miss_count as severity proxy',
                'dispersion_miss_correlation': corr,
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
        
        # Check if we have severity data
        if len(self.centroid_severity) == 0:
            self._visualize_fallback(df)
            return
        
        # Join for plotting
        plot_df = df.merge(
            self.centroid_severity,
            on='centroid_id',
            how='inner'
        )
        
        if len(plot_df) < 10:
            print("Insufficient data for visualization")
            return
        
        import matplotlib.pyplot as plt
        
        # 1. Correlation heatmap: properties vs severity metrics
        severity_cols = ['mean_score_delta', 'max_score_delta']
        if 'mean_oracle_score' in plot_df.columns:
            severity_cols.append('mean_oracle_score')
        
        property_cols = ['dispersion', 'n_tokens', 'n_docs', 'yield', 'sel_freq']
        available = [c for c in property_cols + severity_cols if c in plot_df.columns]
        
        if len(available) > 3:
            corr_df = plot_df[available].dropna()
            if len(corr_df) > 10:
                fig = plot_correlation_heatmap(
                    corr_df,
                    title=f'{self.HYPOTHESIS_ID}: Property-Severity Correlations'
                )
                save_figure(fig, f'{self.HYPOTHESIS_ID}_correlation_heatmap', str(self.output_dir))
                plt.close(fig)
        
        # 2. Scatter: dispersion vs mean_score_delta
        if 'dispersion' in plot_df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                plot_df['dispersion'],
                plot_df['mean_score_delta'],
                c=np.log10(plot_df['miss_count_m5'] + 1) if 'miss_count_m5' in plot_df else None,
                s=50,
                alpha=0.5,
                cmap='Reds'
            )
            if 'miss_count_m5' in plot_df:
                plt.colorbar(scatter, label='log₁₀(miss count)')
            ax.set_xlabel('Dispersion (within-cluster)')
            ax.set_ylabel('Mean Score Delta (miss severity)')
            ax.set_title(f'{self.HYPOTHESIS_ID}: Cluster Dispersion vs Miss Severity')
            save_figure(fig, f'{self.HYPOTHESIS_ID}_dispersion_severity', str(self.output_dir))
            plt.close(fig)
        
        # 3. Scatter: n_tokens vs severity
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            np.log10(plot_df['n_tokens'] + 1),
            plot_df['mean_score_delta'],
            s=50,
            alpha=0.5
        )
        ax.set_xlabel('log₁₀(n_tokens)')
        ax.set_ylabel('Mean Score Delta (miss severity)')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Cluster Size vs Miss Severity')
        save_figure(fig, f'{self.HYPOTHESIS_ID}_size_severity', str(self.output_dir))
        plt.close(fig)
        
        # 4. Stratified by dispersion quartile
        if 'dispersion' in plot_df.columns:
            plot_df = plot_df.copy()
            try:
                plot_df['dispersion_bin'] = pd.qcut(
                    plot_df['dispersion'],
                    q=4,
                    labels=['Q1_tight', 'Q2', 'Q3', 'Q4_dispersed'],
                    duplicates='drop'
                )
                
                if plot_df['dispersion_bin'].nunique() > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_stratified_violin(
                        plot_df,
                        'dispersion_bin',
                        'mean_score_delta',
                        title=f'{self.HYPOTHESIS_ID}: Miss Severity by Cluster Dispersion',
                        ax=ax
                    )
                    save_figure(fig, f'{self.HYPOTHESIS_ID}_violin_dispersion', str(self.output_dir))
                    plt.close(fig)
            except ValueError:
                pass
        
        # 5. Distribution of miss severity
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(plot_df['mean_score_delta'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Mean Score Delta')
        axes[0].set_ylabel('Number of Centroids')
        axes[0].set_title('Distribution of Mean Miss Severity')
        
        axes[1].hist(plot_df['max_score_delta'], bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1].set_xlabel('Max Score Delta')
        axes[1].set_ylabel('Number of Centroids')
        axes[1].set_title('Distribution of Max Miss Severity')
        
        plt.tight_layout()
        save_figure(fig, f'{self.HYPOTHESIS_ID}_severity_distributions', str(self.output_dir))
        plt.close(fig)
    
    def _visualize_fallback(self, df: pd.DataFrame):
        """Fallback visualization using M6 data."""
        if 'm6_miss_rate' not in df.columns:
            print("No miss data available for visualization")
            return
        
        plot_df = df[['dispersion', 'n_tokens', 'm6_miss_rate', 'm6_miss_count']].dropna()
        
        if len(plot_df) < 10:
            return
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            plot_df['dispersion'],
            plot_df['m6_miss_count'],
            c=np.log10(plot_df['n_tokens'] + 1),
            s=50,
            alpha=0.5,
            cmap='viridis'
        )
        plt.colorbar(ax.collections[0], label='log₁₀(n_tokens)')
        ax.set_xlabel('Dispersion')
        ax.set_ylabel('Miss Count (M6)')
        ax.set_title(f'{self.HYPOTHESIS_ID}: Fallback - Dispersion vs Miss Count')
        save_figure(fig, f'{self.HYPOTHESIS_ID}_fallback_scatter', str(self.output_dir))
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    from hypothesis.configs import load_config
    
    parser = argparse.ArgumentParser(description="Run H15: Token Quality → Miss Severity")
    parser.add_argument("--config", choices=["smoke", "dev", "prod"], default="dev")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config, override_run_dir=args.run_dir)
    test = H15_MissSeverity(config)
    test.run()
