"""
Hypothesis H3 Phase 2.3: Rescue Analysis — How Do PARTIAL Docs Get Retrieved?

Background:
    From H3 Phase 2.1, we know:
    - PARTIAL docs (<100% oracle accessibility) have 93.5% recall@100
    - FULLY_OPTIMAL docs (100% oracle accessibility) have 99.7% recall@100
    
    PARTIAL docs are missing some oracle embeddings (their best-scoring embeddings
    for certain query tokens are in centroids that weren't probed). Yet most are
    still retrieved. HOW?

Key Insight on Oracle Embeddings:
    For a (query, doc) pair with N query tokens:
    - Each query token has its OWN oracle embedding (the doc embedding scoring highest
      with that specific query token)
    - So there are N oracle embeddings per (query, doc), potentially in different centroids
    - PARTIAL means: some (but not all) of these N oracles are inaccessible
    
Possible Mechanisms:

    Mechanism A: "Lucky Oracle"
        The doc has multiple oracle embeddings (one per query token).
        Even if some are inaccessible, at least one oracle embedding happens to be
        in a probed centroid. That accessible oracle is sufficient for retrieval.
        
    Mechanism B: "Non-Oracle Rescue"  
        ALL oracle embeddings are inaccessible (no probed centroid contains any oracle).
        But the doc is still retrieved via non-oracle embeddings — embeddings that
        aren't the best for any query token but still score well enough.

Why This Matters:
    If Mechanism A dominates:
        - Oracle accessibility is critical
        - Having at least one accessible oracle is enough
        - Our framing in Phase 2.1 is correct
        
    If Mechanism B is significant (>10%):
        - Oracle-centric analysis misses part of the story
        - Document embedding redundancy provides a "safety net"
        - Spread docs might be more resilient than Phase 2.1 suggests

Test Design:
    For each (query, doc) where:
        - routing_status = PARTIAL
        - doc WAS retrieved (found in search_results at k=100)
    
    Count from M4R: how many oracle embeddings were accessible?
    
    If num_accessible_oracles > 0: Mechanism A (Lucky Oracle)
    If num_accessible_oracles = 0: Mechanism B (Non-Oracle Rescue)
    
Expected Output:
    | Mechanism | Count | Percentage |
    |-----------|-------|------------|
    | Lucky Oracle | ? | ?% |
    | Non-Oracle Rescue | ? | ?% |
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.configs import RuntimeConfig, load_config


class H3_Phase2_Rescue(HypothesisTest):
    """
    H3 Phase 2.3: Analyze how PARTIAL documents get retrieved.
    
    Determines whether retrieval happens via:
    - Lucky Oracle: At least one oracle embedding was accessible
    - Non-Oracle Rescue: Zero oracle embeddings accessible, retrieved via other embeddings
    """
    
    HYPOTHESIS_ID = "H3_Phase2_Rescue"
    HYPOTHESIS_NAME = "PARTIAL Doc Rescue Mechanism"
    CLAIM = "PARTIAL documents are retrieved primarily via accessible oracle embeddings (Lucky Oracle), not non-oracle embeddings"
    
    # Default paths (can be overridden)
    GOLDEN_METRICS_DIR = Path("/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425/golden_metrics_v2")
    
    def __init__(self, config: Optional[RuntimeConfig] = None, golden_metrics_dir: Optional[str] = None):
        if config is None:
            config = load_config("prod")
        super().__init__(config)
        
        if golden_metrics_dir:
            self.GOLDEN_METRICS_DIR = Path(golden_metrics_dir)
        
        # Data containers
        self.routing_status_df: Optional[pd.DataFrame] = None
        self.search_results_df: Optional[pd.DataFrame] = None
        self.m4r_df: Optional[pd.DataFrame] = None
    
    def setup(self):
        """Load required data."""
        print(f"Loading data from: {self.GOLDEN_METRICS_DIR}")
        
        # Load routing status
        routing_path = self.GOLDEN_METRICS_DIR / "routing_status.parquet"
        if not routing_path.exists():
            raise FileNotFoundError(f"Routing status not found: {routing_path}")
        self.routing_status_df = pd.read_parquet(routing_path)
        print(f"  Loaded routing_status: {len(self.routing_status_df):,} rows")
        
        # Load search results
        search_path = self.GOLDEN_METRICS_DIR / "search_results.parquet"
        if not search_path.exists():
            raise FileNotFoundError(f"Search results not found: {search_path}")
        self.search_results_df = pd.read_parquet(search_path)
        print(f"  Loaded search_results: {len(self.search_results_df):,} rows")
        
        # Load M4R (oracle accessibility per query token)
        m4r_path = self.GOLDEN_METRICS_DIR / "M4R.parquet"
        if not m4r_path.exists():
            raise FileNotFoundError(f"M4R not found: {m4r_path}")
        self.m4r_df = pd.read_parquet(m4r_path)
        print(f"  Loaded M4R: {len(self.m4r_df):,} rows")
        
        print(f"\n  M4R columns: {list(self.m4r_df.columns)}")
    
    def analyze(self) -> HypothesisResult:
        """Analyze rescue mechanisms for PARTIAL documents."""
        
        print("\n" + "=" * 60)
        print("H3 Phase 2.3: Rescue Analysis")
        print("=" * 60)
        
        # Step 1: Identify PARTIAL docs
        partial_docs = self.routing_status_df[
            self.routing_status_df['routing_status'] == 'partial'  # lowercase!
        ][['query_id', 'doc_id']].copy()
        
        print(f"\nTotal PARTIAL (query, doc) pairs: {len(partial_docs):,}")
        
        # Step 2: Identify which PARTIAL docs were retrieved at k=100
        # Filter search results to k <= 100
        search_k100 = self.search_results_df[self.search_results_df['rank'] <= 100].copy()
        
        # Mark which (query, doc) pairs were retrieved
        search_k100['retrieved'] = True
        retrieved_pairs = search_k100[['query_id', 'doc_id', 'retrieved']].drop_duplicates()
        
        # Join with PARTIAL docs
        partial_with_retrieval = partial_docs.merge(
            retrieved_pairs,
            on=['query_id', 'doc_id'],
            how='left'
        )
        partial_with_retrieval['retrieved'] = partial_with_retrieval['retrieved'].fillna(False)
        
        n_partial_retrieved = partial_with_retrieval['retrieved'].sum()
        n_partial_not_retrieved = (~partial_with_retrieval['retrieved']).sum()
        
        print(f"\nPARTIAL docs retrieved at k=100: {n_partial_retrieved:,} ({n_partial_retrieved/len(partial_docs)*100:.1f}%)")
        print(f"PARTIAL docs NOT retrieved at k=100: {n_partial_not_retrieved:,} ({n_partial_not_retrieved/len(partial_docs)*100:.1f}%)")
        
        # Step 3: For PARTIAL docs that WERE retrieved, count accessible oracles
        partial_retrieved = partial_with_retrieval[partial_with_retrieval['retrieved']][['query_id', 'doc_id']]
        
        # Join with M4R to get oracle accessibility per query token
        # M4R has: query_id, q_token_id, doc_id, oracle_is_accessible
        partial_m4r = partial_retrieved.merge(
            self.m4r_df[['query_id', 'doc_id', 'q_token_id', 'oracle_is_accessible']],
            on=['query_id', 'doc_id'],
            how='inner'
        )
        
        print(f"\nM4R rows for retrieved PARTIAL docs: {len(partial_m4r):,}")
        
        # Count accessible oracles per (query, doc)
        oracle_counts = partial_m4r.groupby(['query_id', 'doc_id']).agg(
            total_oracles=('oracle_is_accessible', 'count'),
            accessible_oracles=('oracle_is_accessible', 'sum')
        ).reset_index()
        
        oracle_counts['mechanism'] = oracle_counts['accessible_oracles'].apply(
            lambda x: 'Lucky Oracle' if x > 0 else 'Non-Oracle Rescue'
        )
        
        # Step 4: Count mechanisms
        mechanism_counts = oracle_counts['mechanism'].value_counts()
        
        print("\n" + "=" * 60)
        print("RESULTS: Rescue Mechanism Analysis")
        print("=" * 60)
        
        total = len(oracle_counts)
        
        for mechanism in ['Lucky Oracle', 'Non-Oracle Rescue']:
            count = mechanism_counts.get(mechanism, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"\n  {mechanism}:")
            print(f"    Count: {count:,}")
            print(f"    Percentage: {pct:.2f}%")
        
        # Step 5: Deeper analysis of Lucky Oracle cases
        lucky_oracle = oracle_counts[oracle_counts['mechanism'] == 'Lucky Oracle']
        
        if len(lucky_oracle) > 0:
            print("\n" + "-" * 40)
            print("Lucky Oracle: Accessibility Distribution")
            print("-" * 40)
            
            # How many oracles were accessible on average?
            lucky_oracle = lucky_oracle.copy()
            lucky_oracle['accessibility_rate'] = lucky_oracle['accessible_oracles'] / lucky_oracle['total_oracles']
            
            print(f"\n  Average accessible oracles: {lucky_oracle['accessible_oracles'].mean():.2f}")
            print(f"  Average total oracles: {lucky_oracle['total_oracles'].mean():.2f}")
            print(f"  Average accessibility rate: {lucky_oracle['accessibility_rate'].mean()*100:.1f}%")
            
            # Distribution of accessible oracle counts
            print("\n  Distribution of accessible oracle counts:")
            for n in [1, 2, 3, 5, 10]:
                count = (lucky_oracle['accessible_oracles'] >= n).sum()
                print(f"    >= {n} accessible oracles: {count:,} ({count/len(lucky_oracle)*100:.1f}%)")
        
        # Step 6: Analyze Non-Oracle Rescue cases (if any)
        non_oracle = oracle_counts[oracle_counts['mechanism'] == 'Non-Oracle Rescue']
        
        if len(non_oracle) > 0:
            print("\n" + "-" * 40)
            print("Non-Oracle Rescue: Analysis")
            print("-" * 40)
            
            print(f"\n  Total cases: {len(non_oracle):,}")
            print(f"  Average total oracles (all inaccessible): {non_oracle['total_oracles'].mean():.2f}")
            
            # These docs were retrieved despite having ZERO accessible oracle embeddings
            # This means non-oracle embeddings contributed to retrieval
            print("\n  These documents were retrieved purely via non-oracle embeddings!")
        
        # Compile statistics
        statistics = {
            'total_partial_docs': len(partial_docs),
            'partial_retrieved_k100': int(n_partial_retrieved),
            'partial_not_retrieved_k100': int(n_partial_not_retrieved),
            'partial_recall_k100': n_partial_retrieved / len(partial_docs),
            'lucky_oracle_count': int(mechanism_counts.get('Lucky Oracle', 0)),
            'lucky_oracle_pct': mechanism_counts.get('Lucky Oracle', 0) / total * 100 if total > 0 else 0,
            'non_oracle_rescue_count': int(mechanism_counts.get('Non-Oracle Rescue', 0)),
            'non_oracle_rescue_pct': mechanism_counts.get('Non-Oracle Rescue', 0) / total * 100 if total > 0 else 0,
        }
        
        if len(lucky_oracle) > 0:
            statistics['lucky_oracle_avg_accessible'] = lucky_oracle['accessible_oracles'].mean()
            statistics['lucky_oracle_avg_total'] = lucky_oracle['total_oracles'].mean()
            statistics['lucky_oracle_avg_rate'] = lucky_oracle['accessibility_rate'].mean()
        
        # Determine if hypothesis is supported
        # Claim: PARTIAL docs retrieved primarily via Lucky Oracle (>80%)
        lucky_oracle_pct = statistics['lucky_oracle_pct']
        supported = lucky_oracle_pct > 80
        
        interpretation = []
        if lucky_oracle_pct > 90:
            interpretation.append("Strong evidence: Almost all PARTIAL retrievals use accessible oracles.")
            interpretation.append("Oracle accessibility is the dominant mechanism.")
        elif lucky_oracle_pct > 80:
            interpretation.append("Good evidence: Most PARTIAL retrievals use accessible oracles.")
            interpretation.append("Oracle accessibility is important, but non-oracle rescue exists.")
        else:
            interpretation.append("Surprising: Significant non-oracle rescue mechanism.")
            interpretation.append("Document embedding redundancy provides meaningful safety net.")
        
        # Store interpretation in statistics
        statistics['interpretation'] = interpretation
        
        # Store for visualization
        self._oracle_counts = oracle_counts
        self._mechanism_counts = mechanism_counts
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            p_value=0.0,  # Not applicable for this analysis
            effect_size=lucky_oracle_pct / 100,  # Proportion as effect size
            effect_size_ci=(None, None),
            statistics=statistics,
            config_name=self.config.name,
            n_observations=total,
            timestamp=datetime.now().isoformat()
        )
    
    def visualize(self):
        """Generate visualizations."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        if self.results is None:
            print("No results to visualize")
            return
        
        s = self.results.statistics
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Plot 1: Pie chart of rescue mechanisms
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = ['Lucky Oracle\n(≥1 accessible oracle)', 'Non-Oracle Rescue\n(0 accessible oracles)']
        sizes = [s['lucky_oracle_count'], s['non_oracle_rescue_count']]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.02, 0.02)
        
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
        )
        
        ax.set_title('H3 Phase 2.3: How Do PARTIAL Documents Get Retrieved?\n(k=100)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'rescue_mechanism_pie.png', dpi=150)
        plt.close()
        print(f"  Saved: {plot_dir / 'rescue_mechanism_pie.png'}")
        
        # Plot 2: Distribution of accessible oracles for Lucky Oracle cases
        if hasattr(self, '_oracle_counts') and len(self._oracle_counts) > 0:
            lucky = self._oracle_counts[self._oracle_counts['mechanism'] == 'Lucky Oracle']
            
            if len(lucky) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Histogram of accessible oracle counts
                max_accessible = min(lucky['accessible_oracles'].max(), 50)
                bins = range(1, int(max_accessible) + 2)
                
                ax.hist(lucky['accessible_oracles'], bins=bins, color='#3498db', 
                       edgecolor='black', alpha=0.7)
                
                ax.set_xlabel('Number of Accessible Oracle Embeddings', fontsize=12)
                ax.set_ylabel('Count of (query, doc) pairs', fontsize=12)
                ax.set_title('Lucky Oracle Cases: Distribution of Accessible Oracles\n(1+ oracles accessible)', fontsize=14)
                ax.grid(axis='y', alpha=0.3)
                
                # Add mean line
                mean_val = lucky['accessible_oracles'].mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.1f}')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(plot_dir / 'accessible_oracle_distribution.png', dpi=150)
                plt.close()
                print(f"  Saved: {plot_dir / 'accessible_oracle_distribution.png'}")
        
        print(f"\n  All plots saved to: {plot_dir}")
    
    def report(self) -> str:
        """Generate text report."""
        if self.results is None:
            return "No results available."
        
        s = self.results.statistics
        
        lines = [
            "=" * 70,
            f"H3 PHASE 2.3: {self.results.hypothesis_name}",
            "=" * 70,
            "",
            f"Claim: {self.results.claim}",
            "",
            f"RESULT: {'✓ SUPPORTED' if self.results.supported else '✗ NOT SUPPORTED'}",
            "",
            "=" * 70,
            "PARTIAL Document Retrieval Analysis (k=100)",
            "=" * 70,
            "",
            f"Total PARTIAL (query, doc) pairs: {s['total_partial_docs']:,}",
            f"Retrieved at k=100: {s['partial_retrieved_k100']:,} ({s['partial_recall_k100']*100:.1f}%)",
            f"Not retrieved: {s['partial_not_retrieved_k100']:,}",
            "",
            "=" * 70,
            "Rescue Mechanism Breakdown",
            "=" * 70,
            "",
            f"Lucky Oracle (≥1 accessible oracle): {s['lucky_oracle_count']:,} ({s['lucky_oracle_pct']:.1f}%)",
            f"Non-Oracle Rescue (0 accessible):    {s['non_oracle_rescue_count']:,} ({s['non_oracle_rescue_pct']:.1f}%)",
            "",
        ]
        
        if 'lucky_oracle_avg_accessible' in s:
            lines.extend([
                "Lucky Oracle Statistics:",
                f"  Avg accessible oracles per doc: {s['lucky_oracle_avg_accessible']:.1f}",
                f"  Avg total oracles per doc: {s['lucky_oracle_avg_total']:.1f}",
                f"  Avg accessibility rate: {s['lucky_oracle_avg_rate']*100:.1f}%",
                "",
            ])
        
        lines.extend([
            "=" * 70,
            "INTERPRETATION",
            "=" * 70,
            "",
        ])
        
        for interp in s.get('interpretation', []):
            lines.append(f"• {interp}")
        
        return "\n".join(lines)


def run_standalone():
    """Run H3 Phase 2.3 as a standalone script."""
    print("Running H3 Phase 2.3: Rescue Analysis")
    print("=" * 70)
    
    h3_rescue = H3_Phase2_Rescue()
    result = h3_rescue.run()
    
    return result


if __name__ == "__main__":
    run_standalone()
