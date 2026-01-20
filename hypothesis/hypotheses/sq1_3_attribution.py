"""
SQ1.3: Attribution of Distortion to Routing-Side vs Representation-Side Causes

Research Question:
    Of the total distortion observed between WARP and Oracle, how much is attributable to:
    1. Routing-side (availability): tokens never computed because routing missed them
    2. Representation-side (fidelity): tokens computed but with perturbed values

Methodology:
    - Use M4R (golden metrics) for oracle scores and routing status
    - Use M3 (observed winners) for WARP scores
    - Join on (query_id, q_token_id, doc_id) for golden documents
    - Stratify by routing status: FULLY_OPTIMAL vs PARTIAL
    - Compute score gaps within each stratum

Key Insight:
    - For FULLY_OPTIMAL tokens: routing was perfect, any gap is representation-side
    - For PARTIAL tokens: routing contributed to the gap
    - The difference quantifies the attribution split

References:
    - H3: Routing-side recall gap (21.33pp)
    - SQ1.2C: Centroid representation error (~48% L2)
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.configs import RuntimeConfig, ensure_output_dirs


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class ScoreGapResult:
    """Score gap statistics for a stratum."""
    stratum_name: str
    num_observations: int
    
    # Absolute gap: oracle_score - warp_score
    abs_gap_mean: float
    abs_gap_std: float
    abs_gap_median: float
    abs_gap_p10: float
    abs_gap_p90: float
    abs_gap_p99: float
    
    # Relative gap: (oracle_score - warp_score) / oracle_score
    rel_gap_mean: float
    rel_gap_std: float
    rel_gap_median: float
    rel_gap_p10: float
    rel_gap_p90: float
    rel_gap_p99: float
    
    # Score statistics
    oracle_score_mean: float
    warp_score_mean: float


@dataclass
class AttributionResult:
    """Final attribution decomposition."""
    # Overall
    total_observations: int
    total_abs_gap_mean: float
    total_rel_gap_mean: float
    
    # By routing status
    fully_optimal: ScoreGapResult
    partial: ScoreGapResult
    
    # Attribution percentages
    routing_side_fraction: float  # % of gap attributable to routing
    representation_side_fraction: float  # % of gap attributable to representation
    
    # Statistical test: is the difference significant?
    routing_vs_rep_ttest_pvalue: float
    routing_vs_rep_effect_size: float  # Cohen's d


# =============================================================================
# Main Hypothesis Class
# =============================================================================

class SQ1_3_Attribution(HypothesisTest):
    """
    SQ1.3: Attribution of Distortion to Routing vs Representation
    
    Answers: Of the total score gap between WARP and Oracle, how much is due to
    routing failures (availability) vs representation errors (fidelity)?
    """
    
    HYPOTHESIS_ID = "SQ1_3"
    HYPOTHESIS_NAME = "Attribution: Routing-Side vs Representation-Side Distortion"
    CLAIM = (
        "Routing-side distortion (missed tokens) dominates representation-side "
        "distortion (perturbed values) as the primary source of score gaps."
    )
    
    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        
        # Data paths
        self.golden_metrics_dir = Path(config.paths.run_dir) / "golden_metrics"
        self.tier_b_dir = Path(config.paths.run_dir) / "tier_b"
        
        # Loaded data
        self.m4r: Optional[pd.DataFrame] = None  # Oracle scores + routing status
        self.m3: Optional[pd.DataFrame] = None   # WARP observed scores
        self.routing_status: Optional[pd.DataFrame] = None
        self.merged: Optional[pd.DataFrame] = None
        
        # Results
        self.attribution_result: Optional[AttributionResult] = None
        
    def setup(self):
        """Load M4R (oracle) and M3 (WARP) data."""
        ensure_output_dirs(self.config)
        
        print(f"\nLoading data from: {self.config.paths.run_dir}")
        
        # Load M4R (oracle scores per token-doc pair)
        m4r_path = self.golden_metrics_dir / "M4R.parquet"
        print(f"  Loading M4R from {m4r_path}...")
        self.m4r = pd.read_parquet(m4r_path)
        print(f"    M4R shape: {self.m4r.shape}")
        
        # Load routing status (per query-doc pair)
        rs_path = self.golden_metrics_dir / "routing_status.parquet"
        print(f"  Loading routing_status from {rs_path}...")
        self.routing_status = pd.read_parquet(rs_path)
        print(f"    routing_status shape: {self.routing_status.shape}")
        print(f"    Status distribution:")
        print(f"      {self.routing_status['routing_status'].value_counts().to_dict()}")
        
        # Load M3 (WARP observed winners)
        m3_path = self.tier_b_dir / "M3_observed_winners.parquet"
        print(f"  Loading M3 from {m3_path}...")
        self.m3 = pd.read_parquet(m3_path)
        print(f"    M3 shape: {self.m3.shape}")
        
        # Sample if needed (for smoke/dev tiers)
        if self.config.num_queries is not None:
            unique_queries = self.m4r['query_id'].unique()
            if len(unique_queries) > self.config.num_queries:
                np.random.seed(self.config.query_sample_seed)
                sampled_queries = np.random.choice(
                    unique_queries, 
                    size=self.config.num_queries, 
                    replace=False
                )
                print(f"  Sampling {self.config.num_queries} queries...")
                self.m4r = self.m4r[self.m4r['query_id'].isin(sampled_queries)]
                self.m3 = self.m3[self.m3['query_id'].isin(sampled_queries)]
                self.routing_status = self.routing_status[
                    self.routing_status['query_id'].isin(sampled_queries)
                ]
        
    def analyze(self) -> HypothesisResult:
        """Run attribution analysis."""
        print("\n" + "="*60)
        print("Running SQ1.3 Attribution Analysis")
        print("="*60)
        
        # Step 1: Merge M4R (oracle) with M3 (WARP) on (query_id, q_token_id, doc_id)
        print("\n--- Step 1: Merging Oracle and WARP scores ---")
        self._merge_scores()
        
        # Step 2: Add routing status
        print("\n--- Step 2: Adding routing status ---")
        self._add_routing_status()
        
        # Step 3: Compute score gaps by stratum
        print("\n--- Step 3: Computing score gaps by routing status ---")
        self._compute_attribution()
        
        # Determine if hypothesis is supported
        # Hypothesis: Routing-side dominates (>50% of gap)
        supported = self.attribution_result.routing_side_fraction > 0.5
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=self.attribution_result.routing_side_fraction,
            effect_size_ci=(
                self.attribution_result.representation_side_fraction,
                self.attribution_result.routing_side_fraction
            ),
            p_value=self.attribution_result.routing_vs_rep_ttest_pvalue,
            statistics={
                'total_observations': self.attribution_result.total_observations,
                'total_rel_gap_mean': self.attribution_result.total_rel_gap_mean,
                'fully_optimal_rel_gap': self.attribution_result.fully_optimal.rel_gap_mean,
                'partial_rel_gap': self.attribution_result.partial.rel_gap_mean,
                'routing_side_fraction': self.attribution_result.routing_side_fraction,
                'representation_side_fraction': self.attribution_result.representation_side_fraction,
                'interpretation': self._generate_interpretation(),
            },
            config_name=self.config.name,
            n_observations=self.attribution_result.total_observations,
            timestamp=datetime.now().isoformat(),
        )
    
    def _merge_scores(self):
        """Merge M4R (oracle) with M3 (WARP) scores."""
        # M4R has: query_id, q_token_id, doc_id, oracle_score, oracle_is_accessible
        # M3 has: query_id, q_token_id, doc_id, winner_score (WARP score)
        
        # For golden documents, we want to compare oracle_score vs warp_score
        # The oracle winner for a token might be in a different embedding position
        # than what WARP found, but we compare the scores
        
        # Rename M3 columns for clarity
        m3_renamed = self.m3.rename(columns={
            'winner_score': 'warp_score',
            'winner_embedding_pos': 'warp_embedding_pos'
        })
        
        # Merge on (query_id, q_token_id, doc_id)
        # This gives us: for each (query, token, golden_doc), what was oracle score vs warp score
        self.merged = pd.merge(
            self.m4r[['query_id', 'q_token_id', 'doc_id', 'oracle_score', 'oracle_is_accessible']],
            m3_renamed[['query_id', 'q_token_id', 'doc_id', 'warp_score']],
            on=['query_id', 'q_token_id', 'doc_id'],
            how='inner'  # Only keep pairs that exist in both
        )
        
        print(f"  Merged shape: {self.merged.shape}")
        print(f"  (M4R had {len(self.m4r)}, M3 had {len(self.m3)})")
        
        # Compute gaps
        self.merged['abs_gap'] = self.merged['oracle_score'] - self.merged['warp_score']
        self.merged['rel_gap'] = self.merged['abs_gap'] / (self.merged['oracle_score'] + 1e-10)
        
        print(f"\n  Overall score gap statistics:")
        print(f"    Mean absolute gap: {self.merged['abs_gap'].mean():.6f}")
        print(f"    Mean relative gap: {self.merged['rel_gap'].mean():.4f} ({100*self.merged['rel_gap'].mean():.2f}%)")
        print(f"    Median relative gap: {self.merged['rel_gap'].median():.4f} ({100*self.merged['rel_gap'].median():.2f}%)")
        
    def _add_routing_status(self):
        """Add routing status to merged data."""
        # Routing status is per (query_id, doc_id)
        # Merge it into our token-level data
        
        self.merged = pd.merge(
            self.merged,
            self.routing_status[['query_id', 'doc_id', 'routing_status']],
            on=['query_id', 'doc_id'],
            how='left'
        )
        
        # Fill missing with 'unknown' (shouldn't happen for golden docs)
        self.merged['routing_status'] = self.merged['routing_status'].fillna('unknown')
        
        print(f"  Routing status distribution in merged data:")
        status_counts = self.merged['routing_status'].value_counts()
        for status, count in status_counts.items():
            print(f"    {status}: {count:,} ({100*count/len(self.merged):.2f}%)")
    
    def _compute_score_gap_stats(self, df: pd.DataFrame, stratum_name: str) -> ScoreGapResult:
        """Compute score gap statistics for a stratum."""
        abs_gaps = df['abs_gap'].values
        rel_gaps = df['rel_gap'].values
        
        return ScoreGapResult(
            stratum_name=stratum_name,
            num_observations=len(df),
            abs_gap_mean=float(np.mean(abs_gaps)),
            abs_gap_std=float(np.std(abs_gaps)),
            abs_gap_median=float(np.median(abs_gaps)),
            abs_gap_p10=float(np.percentile(abs_gaps, 10)),
            abs_gap_p90=float(np.percentile(abs_gaps, 90)),
            abs_gap_p99=float(np.percentile(abs_gaps, 99)),
            rel_gap_mean=float(np.mean(rel_gaps)),
            rel_gap_std=float(np.std(rel_gaps)),
            rel_gap_median=float(np.median(rel_gaps)),
            rel_gap_p10=float(np.percentile(rel_gaps, 10)),
            rel_gap_p90=float(np.percentile(rel_gaps, 90)),
            rel_gap_p99=float(np.percentile(rel_gaps, 99)),
            oracle_score_mean=float(df['oracle_score'].mean()),
            warp_score_mean=float(df['warp_score'].mean()),
        )
    
    def _compute_attribution(self):
        """Compute attribution decomposition."""
        # Split by routing status
        fully_optimal = self.merged[self.merged['routing_status'] == 'fully_optimal']
        partial = self.merged[self.merged['routing_status'] == 'partial']
        
        print(f"\n  FULLY_OPTIMAL stratum: {len(fully_optimal):,} observations")
        print(f"  PARTIAL stratum: {len(partial):,} observations")
        
        # Compute stats for each stratum
        fo_stats = self._compute_score_gap_stats(fully_optimal, 'fully_optimal')
        partial_stats = self._compute_score_gap_stats(partial, 'partial')
        
        print(f"\n  Score Gap by Routing Status:")
        print(f"    FULLY_OPTIMAL (representation-side only):")
        print(f"      Mean relative gap: {100*fo_stats.rel_gap_mean:.4f}%")
        print(f"      Median relative gap: {100*fo_stats.rel_gap_median:.4f}%")
        print(f"    PARTIAL (routing + representation):")
        print(f"      Mean relative gap: {100*partial_stats.rel_gap_mean:.4f}%")
        print(f"      Median relative gap: {100*partial_stats.rel_gap_median:.4f}%")
        
        # Overall stats
        total_rel_gap = self.merged['rel_gap'].mean()
        total_abs_gap = self.merged['abs_gap'].mean()
        
        # Attribution decomposition
        # Representation-side = gap when routing is perfect (fully_optimal)
        # Routing-side = additional gap from routing failures
        representation_side_gap = fo_stats.rel_gap_mean
        total_gap = total_rel_gap
        
        # If total gap is very small, avoid division issues
        if abs(total_gap) < 1e-10:
            representation_side_fraction = 0.5
            routing_side_fraction = 0.5
        else:
            representation_side_fraction = representation_side_gap / total_gap
            routing_side_fraction = 1.0 - representation_side_fraction
        
        # Clamp to [0, 1] for interpretability
        representation_side_fraction = max(0, min(1, representation_side_fraction))
        routing_side_fraction = max(0, min(1, routing_side_fraction))
        
        print(f"\n  *** ATTRIBUTION DECOMPOSITION ***")
        print(f"    Total mean relative gap: {100*total_rel_gap:.4f}%")
        print(f"    Representation-side (FULLY_OPTIMAL gap): {100*representation_side_gap:.4f}%")
        print(f"    Routing-side (additional gap): {100*(total_gap - representation_side_gap):.4f}%")
        print(f"    ")
        print(f"    Attribution:")
        print(f"      Representation-side: {100*representation_side_fraction:.1f}%")
        print(f"      Routing-side: {100*routing_side_fraction:.1f}%")
        
        # Statistical test: is the gap difference significant?
        if len(fully_optimal) > 0 and len(partial) > 0:
            ttest = stats.ttest_ind(
                fully_optimal['rel_gap'].values,
                partial['rel_gap'].values,
                equal_var=False
            )
            # Cohen's d effect size
            pooled_std = np.sqrt(
                (fo_stats.rel_gap_std**2 + partial_stats.rel_gap_std**2) / 2
            )
            cohens_d = (partial_stats.rel_gap_mean - fo_stats.rel_gap_mean) / (pooled_std + 1e-10)
        else:
            ttest = type('obj', (object,), {'pvalue': 1.0})()
            cohens_d = 0.0
        
        print(f"\n  Statistical test (PARTIAL vs FULLY_OPTIMAL):")
        print(f"    t-test p-value: {ttest.pvalue:.2e}")
        print(f"    Cohen's d: {cohens_d:.3f}")
        
        self.attribution_result = AttributionResult(
            total_observations=len(self.merged),
            total_abs_gap_mean=float(total_abs_gap),
            total_rel_gap_mean=float(total_rel_gap),
            fully_optimal=fo_stats,
            partial=partial_stats,
            routing_side_fraction=float(routing_side_fraction),
            representation_side_fraction=float(representation_side_fraction),
            routing_vs_rep_ttest_pvalue=float(ttest.pvalue),
            routing_vs_rep_effect_size=float(cohens_d),
        )
    
    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation."""
        r = self.attribution_result
        
        if r.routing_side_fraction > 0.7:
            verdict = "Routing-side distortion DOMINATES"
            explanation = (
                f"Routing failures account for {100*r.routing_side_fraction:.0f}% of score gaps. "
                f"Improving routing (more probes, better centroid selection) would address most distortion."
            )
        elif r.routing_side_fraction > 0.5:
            verdict = "Routing-side distortion is PRIMARY but not dominant"
            explanation = (
                f"Routing failures account for {100*r.routing_side_fraction:.0f}% of score gaps. "
                f"Both routing and representation improvements would help."
            )
        elif r.routing_side_fraction > 0.3:
            verdict = "Representation-side distortion is PRIMARY"
            explanation = (
                f"Representation errors (quantization, compression) account for {100*r.representation_side_fraction:.0f}% of score gaps. "
                f"Even perfect routing would not eliminate most distortion."
            )
        else:
            verdict = "Representation-side distortion DOMINATES"
            explanation = (
                f"Representation errors account for {100*r.representation_side_fraction:.0f}% of score gaps. "
                f"Routing is not the bottleneck; representation fidelity is."
            )
        
        return f"{verdict}. {explanation}"
    
    def visualize(self, output_dir: Optional[Path] = None):
        """Generate visualizations."""
        import matplotlib.pyplot as plt
        
        if output_dir is None:
            output_dir = Path(self.config.paths.output_root) / self.HYPOTHESIS_ID / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations in: {output_dir}")
        
        # =================================================================
        # Figure 1: Score Gap Distribution by Routing Status
        # =================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        fully_optimal = self.merged[self.merged['routing_status'] == 'fully_optimal']
        partial = self.merged[self.merged['routing_status'] == 'partial']
        
        # Panel 1: Absolute gap histogram
        ax = axes[0]
        ax.hist(partial['abs_gap'], bins=100, alpha=0.6, label=f'Partial (n={len(partial):,})', density=True)
        if len(fully_optimal) > 0:
            ax.hist(fully_optimal['abs_gap'], bins=100, alpha=0.6, label=f'Fully Optimal (n={len(fully_optimal):,})', density=True)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Absolute Score Gap (oracle - warp)')
        ax.set_ylabel('Density')
        ax.set_title('Score Gap Distribution by Routing Status')
        ax.legend()
        
        # Panel 2: Relative gap histogram
        ax = axes[1]
        ax.hist(partial['rel_gap'].clip(-1, 1), bins=100, alpha=0.6, label='Partial', density=True)
        if len(fully_optimal) > 0:
            ax.hist(fully_optimal['rel_gap'].clip(-1, 1), bins=100, alpha=0.6, label='Fully Optimal', density=True)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Relative Score Gap (oracle - warp) / oracle')
        ax.set_ylabel('Density')
        ax.set_title('Relative Score Gap Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_gap_by_routing.png', dpi=150)
        plt.close()
        
        # =================================================================
        # Figure 2: Attribution Pie Chart
        # =================================================================
        fig, ax = plt.subplots(figsize=(8, 8))
        
        r = self.attribution_result
        sizes = [r.representation_side_fraction, r.routing_side_fraction]
        labels = [
            f'Representation-side\n({100*r.representation_side_fraction:.1f}%)',
            f'Routing-side\n({100*r.routing_side_fraction:.1f}%)'
        ]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
               shadow=True, startangle=90)
        ax.set_title('Attribution of Score Gap to Distortion Sources', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'attribution_pie.png', dpi=150)
        plt.close()
        
        # =================================================================
        # Figure 3: Summary Statistics Table
        # =================================================================
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        summary = f"""
SQ1.3: ATTRIBUTION OF DISTORTION
{'='*60}

Research Question:
  Of the total score gap between WARP and Oracle, how much is due to
  routing failures (availability) vs representation errors (fidelity)?

DATA:
  Total observations: {r.total_observations:,}
  FULLY_OPTIMAL (perfect routing): {r.fully_optimal.num_observations:,}
  PARTIAL (routing had misses): {r.partial.num_observations:,}

SCORE GAP ANALYSIS:
  
  Overall:
    Mean relative gap: {100*r.total_rel_gap_mean:.4f}%
  
  FULLY_OPTIMAL (representation-side only):
    Mean relative gap: {100*r.fully_optimal.rel_gap_mean:.4f}%
    Median: {100*r.fully_optimal.rel_gap_median:.4f}%
    P90: {100*r.fully_optimal.rel_gap_p90:.4f}%
  
  PARTIAL (routing + representation):
    Mean relative gap: {100*r.partial.rel_gap_mean:.4f}%
    Median: {100*r.partial.rel_gap_median:.4f}%
    P90: {100*r.partial.rel_gap_p90:.4f}%

ATTRIBUTION DECOMPOSITION:
  
  Representation-side: {100*r.representation_side_fraction:.1f}%
    (Gap that persists even with perfect routing)
  
  Routing-side: {100*r.routing_side_fraction:.1f}%
    (Additional gap from routing failures)

STATISTICAL TEST:
  PARTIAL vs FULLY_OPTIMAL t-test p-value: {r.routing_vs_rep_ttest_pvalue:.2e}
  Cohen's d effect size: {r.routing_vs_rep_effect_size:.3f}

{'='*60}
CONCLUSION: {self._generate_interpretation()}
"""
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'summary.png', dpi=150)
        plt.close()
        
        print(f"  Saved visualizations to {output_dir}")
    
    def report(self) -> str:
        """Generate text report."""
        r = self.attribution_result
        
        lines = [
            "="*70,
            "SQ1.3: ATTRIBUTION OF DISTORTION",
            "="*70,
            "",
            f"Claim: {self.CLAIM}",
            "",
            "-"*70,
            "DATA SUMMARY",
            "-"*70,
            f"  Total observations: {r.total_observations:,}",
            f"  FULLY_OPTIMAL: {r.fully_optimal.num_observations:,}",
            f"  PARTIAL: {r.partial.num_observations:,}",
            "",
            "-"*70,
            "SCORE GAP BY ROUTING STATUS",
            "-"*70,
            "",
            "  FULLY_OPTIMAL (representation-side only):",
            f"    Mean relative gap: {100*r.fully_optimal.rel_gap_mean:.4f}%",
            f"    Median: {100*r.fully_optimal.rel_gap_median:.4f}%",
            f"    P10-P90: [{100*r.fully_optimal.rel_gap_p10:.4f}%, {100*r.fully_optimal.rel_gap_p90:.4f}%]",
            "",
            "  PARTIAL (routing + representation):",
            f"    Mean relative gap: {100*r.partial.rel_gap_mean:.4f}%",
            f"    Median: {100*r.partial.rel_gap_median:.4f}%",
            f"    P10-P90: [{100*r.partial.rel_gap_p10:.4f}%, {100*r.partial.rel_gap_p90:.4f}%]",
            "",
            "-"*70,
            "ATTRIBUTION DECOMPOSITION",
            "-"*70,
            f"  Representation-side: {100*r.representation_side_fraction:.1f}%",
            f"  Routing-side: {100*r.routing_side_fraction:.1f}%",
            "",
            "-"*70,
            "CONCLUSION",
            "-"*70,
            "",
            f"  {self._generate_interpretation()}",
            "",
            f"Result: {'SUPPORTED' if self.results.supported else 'NOT SUPPORTED'}",
            "",
        ]
        
        return "\n".join(lines)


# =============================================================================
# Entry Point
# =============================================================================

def run_sq1_3(config_name: str = "smoke") -> HypothesisResult:
    """Run SQ1.3 attribution analysis."""
    from hypothesis.configs import load_config
    
    config = load_config(config_name)
    hypothesis = SQ1_3_Attribution(config)
    return hypothesis.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SQ1.3 Attribution Analysis")
    parser.add_argument("--config", default="smoke", choices=["smoke", "dev", "prod", "full"],
                        help="Configuration tier")
    
    args = parser.parse_args()
    result = run_sq1_3(args.config)
