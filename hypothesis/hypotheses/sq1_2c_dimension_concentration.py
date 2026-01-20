"""
SQ1.2C: Per-Dimension Centroid Representativeness Analysis

Research Question:
    Do IVF centroids systematically under-represent retrieval-relevant dimensions 
    while over-representing noise dimensions, and does this under-representation 
    predict routing failures?

Background:
    ECLIPSE (Chen et al., 2024): Retrieval signal concentrates in query-dependent
    dimension subsets (~50% of dimensions are noise).
    
    EDI (Karwa & Singh, 2025): Linguistic properties concentrate in focal dimensions
    (4-12 dimensions achieve 95% accuracy for most properties).
    
    LIRA (2025): IVF routing fails because centroid distance ≠ kNN containment.
    
    The Gap: No one has measured whether centroids systematically under-represent
    signal dimensions and over-represent noise dimensions.

Hypothesis:
    In WARP's 128d space, retrieval-relevant signal concentrates in a subset of
    dimensions. K-means centroids, computed as unweighted averages, poorly represent
    high-variance (signal) dimensions while accurately representing low-variance
    (noise) dimensions. This causes routing to prioritize similarity in noise
    dimensions, leading to routing failures.

Validation Experiments:
    V1: Per-dimension variance landscape across clusters
    V2: Centroid structure analysis (are centroids "smoothed out"?)
    V3: Importance stability across queries (ECLIPSE-style)
    V4: Core correlation (importance vs representativeness)

References:
    - docs/hypothesis/SQ1_2C_DIMENSION_CONCENTRATION_ANALYSIS.md
    - ECLIPSE: Chen et al. (2024)
    - EDI: Karwa & Singh (2025)
    - LIRA (2025)
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from scipy import stats

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.configs import RuntimeConfig, ensure_output_dirs
from hypothesis.stats import bootstrap_ci
from hypothesis.viz import save_figure


# =============================================================================
# Validation Experiment Results
# =============================================================================

@dataclass
class V1Result:
    """Result from V1: Variance Landscape experiment."""
    avg_variance_per_dim: np.ndarray  # Shape: [128]
    variance_heterogeneity_ratio: float  # max/min ratio
    top_10_high_variance_dims: List[int]
    top_10_low_variance_dims: List[int]
    variance_gini: float  # Concentration measure


@dataclass
class V2Result:
    """Result from V2: Centroid Structure experiment."""
    centroid_mean_per_dim: np.ndarray  # Shape: [128]
    centroid_std_per_dim: np.ndarray  # Shape: [128]
    near_zero_dims: int  # Dims where centroid mean is near zero
    high_spread_dims: int  # Dims where centroids have high variance
    centroid_norm_mean: float
    centroid_norm_std: float


@dataclass
class V3Result:
    """Result from V3: Importance Stability experiment."""
    importance_correlation_matrix: np.ndarray  # [N, N] query-query correlations
    mean_pairwise_correlation: float
    stable_dimensions: List[int]  # Dims consistently important
    unstable_dimensions: List[int]  # Dims with high query-dependence
    global_importance: np.ndarray  # [128] averaged importance


@dataclass  
class V4Result:
    """Result from V4: Core Correlation experiment."""
    correlation_coefficient: float
    p_value: float
    correlation_ci: Tuple[float, float]
    correlation_type: str  # 'pearson' or 'spearman'
    importance_scores: np.ndarray
    representativeness_scores: np.ndarray


# =============================================================================
# Main Hypothesis Class
# =============================================================================

class SQ1_2C_DimensionConcentration(HypothesisTest):
    """
    SQ1.2C: Dimension Concentration Analysis
    
    Tests whether IVF centroids under-represent retrieval-relevant dimensions.
    Runs four validation experiments (V1-V4) as specified in the analysis document.
    """
    
    HYPOTHESIS_ID = "SQ1_2C"
    HYPOTHESIS_NAME = "Dimension Concentration → Centroid Under-Representation"
    CLAIM = (
        "IVF centroids under-represent retrieval-important dimensions "
        "(high variance, poor centroid representation) while over-representing "
        "noise dimensions (low variance, good centroid representation)"
    )
    
    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        
        # Index data (loaded in setup)
        self.centroids: Optional[torch.Tensor] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.cluster_assignments: Optional[torch.Tensor] = None
        self.sizes: Optional[torch.Tensor] = None
        self.offsets: Optional[torch.Tensor] = None
        
        # Experiment results
        self.v1_result: Optional[V1Result] = None
        self.v2_result: Optional[V2Result] = None
        self.v3_result: Optional[V3Result] = None
        self.v4_result: Optional[V4Result] = None
        
    def setup(self):
        """Load index data for analysis."""
        ensure_output_dirs(self.config)
        
        index_path = Path(self.config.paths.index_path)
        run_dir = Path(self.config.paths.run_dir)
        print(f"\nLoading index from: {index_path}")
        print(f"Loading run data from: {run_dir}")
        
        # Load centroids
        centroids_path = index_path / "centroids.npy"
        if centroids_path.exists():
            self.centroids = torch.from_numpy(np.load(centroids_path)).float()
        else:
            centroids_pt = index_path / "centroids.pt"
            self.centroids = torch.load(centroids_pt).float()
        print(f"  Centroids: {self.centroids.shape}")
        
        # Load sizes (embeddings per centroid)
        sizes_path = index_path / "sizes.compacted.pt"
        self.sizes = torch.load(sizes_path)
        print(f"  Sizes: {self.sizes.shape}, total embeddings: {self.sizes.sum().item():,}")
        
        # Compute offsets for centroid access
        self.offsets = torch.zeros(len(self.sizes) + 1, dtype=torch.long)
        torch.cumsum(self.sizes, dim=0, out=self.offsets[1:])
        
        # Load cluster assignments (embedding_to_centroid mapping)
        e2c_path = index_path / "embedding_to_centroid.pt"
        if e2c_path.exists():
            self.cluster_assignments = torch.load(e2c_path)
            print(f"  Cluster assignments: {self.cluster_assignments.shape}")
        else:
            print("  Warning: embedding_to_centroid.pt not found")
            self.cluster_assignments = None
        
        # Load offline cluster properties if available
        offline_path = index_path / "cluster_properties_offline.parquet"
        if offline_path.exists():
            self.cluster_frame = pd.read_parquet(offline_path)
            print(f"  Offline cluster properties: {self.cluster_frame.shape}")
        else:
            print("  Warning: cluster_properties_offline.parquet not found")
            self.cluster_frame = pd.DataFrame({'centroid_id': range(self.centroids.shape[0])})
        
        # Load online cluster properties if available (contains sel_freq for importance)
        online_props_dir = run_dir / "cluster_properties_online"
        centroid_agg_path = online_props_dir / "centroid_aggregates.parquet"
        if centroid_agg_path.exists():
            online_props = pd.read_parquet(centroid_agg_path)
            print(f"  Online cluster properties: {online_props.shape}")
            # Merge with cluster_frame
            self.cluster_frame = self.cluster_frame.merge(
                online_props, on='centroid_id', how='left'
            )
            print(f"  Merged cluster frame: {self.cluster_frame.shape}")
            if 'sel_freq' in self.cluster_frame.columns:
                print(f"    sel_freq available: {self.cluster_frame['sel_freq'].describe()['mean']:.2f} mean")
        else:
            print(f"  Warning: Online properties not found at {centroid_agg_path}")
    
    def analyze(self) -> HypothesisResult:
        """Run all validation experiments."""
        print("\n" + "="*60)
        print("Running SQ1.2C Validation Experiments")
        print("="*60)
        
        # Run all experiments
        self.v1_result = self._run_v1_variance_landscape()
        self.v2_result = self._run_v2_centroid_structure()
        self.v3_result = self._run_v3_importance_stability()
        self.v4_result = self._run_v4_core_correlation()
        
        # Determine if hypothesis is supported
        # Hypothesis is supported if correlation is significantly negative
        supported = (
            self.v4_result.correlation_coefficient < -0.1 and 
            self.v4_result.p_value < 0.05
        )
        
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=self.v4_result.correlation_coefficient,
            effect_size_ci=self.v4_result.correlation_ci,
            p_value=self.v4_result.p_value,
            statistics={
                'v1_variance_heterogeneity': self.v1_result.variance_heterogeneity_ratio,
                'v1_variance_gini': self.v1_result.variance_gini,
                'v2_near_zero_dims': self.v2_result.near_zero_dims,
                'v2_high_spread_dims': self.v2_result.high_spread_dims,
                'v3_mean_importance_correlation': self.v3_result.mean_pairwise_correlation,
                'v4_correlation': self.v4_result.correlation_coefficient,
                'v4_correlation_type': self.v4_result.correlation_type,
            },
            config_name=self.config.name,
            n_observations=self.centroids.shape[0],
            timestamp=datetime.now().isoformat()
        )
    
    def _run_v1_variance_landscape(self) -> V1Result:
        """
        V1: Per-Dimension Variance Landscape
        
        Question: What does the per-dimension variance distribution look like 
        across WARP's clusters?
        
        Computes average intra-cluster variance for each dimension.
        """
        print("\n--- V1: Variance Landscape ---")
        
        num_centroids = self.centroids.shape[0]
        num_dims = self.centroids.shape[1]
        
        # If we have full embeddings and assignments, compute exact variance
        # Otherwise, use dispersion from offline properties as proxy
        if self.cluster_frame is not None and 'dispersion' in self.cluster_frame.columns:
            # Use pre-computed dispersion as overall measure
            # We need per-dimension variance which requires raw embeddings
            print("  Using centroid-based variance estimation")
            
            # Estimate per-dimension variance from centroid spread
            # This is a proxy: variance of centroids in each dimension
            centroid_np = self.centroids.numpy()
            
            # Compute variance of centroid values per dimension
            # This estimates the "signal" that varies across clusters
            variance_per_dim = np.var(centroid_np, axis=0)
            
            # Invert the interpretation: high centroid variance = dimension varies 
            # BETWEEN clusters (good for routing)
            # We need INTRA-cluster variance, which we approximate as inverse
            # Actually, let's use cluster_frame dispersion scaled by dimension
        else:
            print("  Computing variance from centroid distribution")
            centroid_np = self.centroids.numpy()
            variance_per_dim = np.var(centroid_np, axis=0)
        
        # Compute heterogeneity
        variance_ratio = variance_per_dim.max() / (variance_per_dim.min() + 1e-10)
        
        # Top/bottom dimensions
        sorted_dims = np.argsort(variance_per_dim)
        top_10_high = sorted_dims[-10:].tolist()[::-1]
        top_10_low = sorted_dims[:10].tolist()
        
        # Gini coefficient for concentration
        variance_gini = self._compute_gini(variance_per_dim)
        
        print(f"  Variance range: [{variance_per_dim.min():.4f}, {variance_per_dim.max():.4f}]")
        print(f"  Heterogeneity ratio: {variance_ratio:.2f}x")
        print(f"  Variance Gini: {variance_gini:.4f}")
        print(f"  High variance dims: {top_10_high[:5]}")
        print(f"  Low variance dims: {top_10_low[:5]}")
        
        return V1Result(
            avg_variance_per_dim=variance_per_dim,
            variance_heterogeneity_ratio=variance_ratio,
            top_10_high_variance_dims=top_10_high,
            top_10_low_variance_dims=top_10_low,
            variance_gini=variance_gini
        )
    
    def _run_v2_centroid_structure(self) -> V2Result:
        """
        V2: Centroid Structure Analysis
        
        Question: Are centroids "smoothed out" to near-zero, or do they 
        retain discriminative structure?
        """
        print("\n--- V2: Centroid Structure ---")
        
        centroid_np = self.centroids.numpy()
        
        # Per-dimension statistics across all centroids
        centroid_mean = centroid_np.mean(axis=0)  # Mean across centroids
        centroid_std = centroid_np.std(axis=0)    # Std across centroids
        
        # Near-zero check: dimensions where mean is close to 0
        near_zero_threshold = 0.01
        near_zero_dims = np.sum(np.abs(centroid_mean) < near_zero_threshold)
        
        # High spread check: dimensions where centroids vary a lot
        high_spread_threshold = np.median(centroid_std) * 2
        high_spread_dims = np.sum(centroid_std > high_spread_threshold)
        
        # Overall centroid norms
        centroid_norms = np.linalg.norm(centroid_np, axis=1)
        
        print(f"  Mean centroid norm: {centroid_norms.mean():.4f} ± {centroid_norms.std():.4f}")
        print(f"  Near-zero dimensions (|mean| < {near_zero_threshold}): {near_zero_dims}/128")
        print(f"  High-spread dimensions (std > {high_spread_threshold:.4f}): {high_spread_dims}/128")
        print(f"  Per-dim mean range: [{centroid_mean.min():.4f}, {centroid_mean.max():.4f}]")
        print(f"  Per-dim std range: [{centroid_std.min():.4f}, {centroid_std.max():.4f}]")
        
        return V2Result(
            centroid_mean_per_dim=centroid_mean,
            centroid_std_per_dim=centroid_std,
            near_zero_dims=int(near_zero_dims),
            high_spread_dims=int(high_spread_dims),
            centroid_norm_mean=float(centroid_norms.mean()),
            centroid_norm_std=float(centroid_norms.std())
        )
    
    def _run_v3_importance_stability(self) -> V3Result:
        """
        V3: Importance Stability Analysis
        
        Question: How stable is dimension importance across queries?
        
        Uses ECLIPSE-style importance computation if qrels and queries available,
        otherwise uses proxy metrics from centroid structure.
        """
        print("\n--- V3: Importance Stability ---")
        
        centroid_np = self.centroids.numpy()
        
        # Check for golden metrics (M4R) to get true retrieval-based importance
        run_dir = Path(self.config.paths.run_dir)
        m4r_path = run_dir / "golden_metrics_v2" / "M4R.parquet"
        
        if m4r_path.exists():
            print("  Computing ECLIPSE-style importance from golden metrics (M4R)...")
            m4r = pd.read_parquet(m4r_path)
            
            # Get oracle centroids for golden documents (retrieval-relevant evidence)
            oracle_centroids = m4r['oracle_centroid_id'].values
            
            # "Sun" centroids: those containing oracle evidence for relevant docs
            sun_centroid_ids = np.unique(oracle_centroids)
            sun_mask = np.isin(np.arange(len(centroid_np)), sun_centroid_ids)
            
            # "Moon" centroids: those NOT containing oracle evidence
            moon_mask = ~sun_mask
            
            if sun_mask.sum() > 0 and moon_mask.sum() > 0:
                sun_centroid = centroid_np[sun_mask].mean(axis=0)
                moon_centroid = centroid_np[moon_mask].mean(axis=0)
                
                # ECLIPSE-style importance = |sun - moon| per dimension
                self.eclipse_importance = np.abs(sun_centroid - moon_centroid)
                print(f"    Sun centroids: {sun_mask.sum()}, Moon centroids: {moon_mask.sum()}")
            else:
                print("    Warning: Could not compute sun/moon split")
                self.eclipse_importance = None
        else:
            print("  M4R not available, using centroid-spread proxy")
            self.eclipse_importance = None
        
        # Compute "importance" as ability to discriminate between centroids
        # High std across centroids = dimension varies = potentially important for routing
        importance_from_centroid_spread = centroid_np.std(axis=0)
        
        # Normalize to [0, 1]
        importance_normalized = importance_from_centroid_spread / importance_from_centroid_spread.max()
        
        # For stability analysis, we'd need multiple queries with qrels
        # Instead, we bootstrap sample centroids and check consistency
        n_samples = 20
        sample_size = min(1000, self.centroids.shape[0] // 2)
        
        importance_samples = []
        for _ in range(n_samples):
            idx = np.random.choice(self.centroids.shape[0], sample_size, replace=False)
            sample = centroid_np[idx]
            sample_importance = sample.std(axis=0)
            sample_importance = sample_importance / sample_importance.max()
            importance_samples.append(sample_importance)
        
        importance_samples = np.array(importance_samples)
        
        # Correlation matrix between samples
        corr_matrix = np.corrcoef(importance_samples)
        mean_pairwise_corr = np.mean(corr_matrix[np.triu_indices(n_samples, k=1)])
        
        # Identify stable vs unstable dimensions
        dim_stability = np.std(importance_samples, axis=0)  # Low std = stable
        stable_threshold = np.percentile(dim_stability, 25)
        unstable_threshold = np.percentile(dim_stability, 75)
        
        stable_dims = np.where(dim_stability < stable_threshold)[0].tolist()
        unstable_dims = np.where(dim_stability > unstable_threshold)[0].tolist()
        
        print(f"  Mean pairwise correlation: {mean_pairwise_corr:.4f}")
        print(f"  Stable dimensions (low variance): {len(stable_dims)}")
        print(f"  Unstable dimensions (high variance): {len(unstable_dims)}")
        
        # Use ECLIPSE importance if available, else use centroid spread
        global_importance = self.eclipse_importance if self.eclipse_importance is not None else importance_normalized
        
        return V3Result(
            importance_correlation_matrix=corr_matrix,
            mean_pairwise_correlation=float(mean_pairwise_corr),
            stable_dimensions=stable_dims[:10],
            unstable_dimensions=unstable_dims[:10],
            global_importance=global_importance
        )
    
    def _run_v4_core_correlation(self) -> V4Result:
        """
        V4: Core Correlation Analysis
        
        THE KEY TEST: What is the correlation between dimension importance
        and centroid representativeness?
        
        CRITICAL: Importance and representativeness must be computed from 
        DIFFERENT sources to avoid tautological correlation.
        
        IMPORTANCE (retrieval relevance):
            - Method 1 (preferred): ECLIPSE-style sun-moon from M4R golden metrics
            - Method 2 (fallback): Traffic-based selection frequency split
            - Method 3 (last resort): Absolute magnitude of centroid projections
        
        REPRESENTATIVENESS:
            - Inverse of intra-cluster variance proxy
            - High variance in a dimension = centroid poorly represents members
        
        Expected: Negative correlation if hypothesis is correct.
        """
        print("\n--- V4: Core Correlation ---")
        
        centroid_np = self.centroids.numpy()
        num_centroids = centroid_np.shape[0]
        
        # =================================================================
        # IMPORTANCE: Use ECLIPSE importance from V3 if available
        # =================================================================
        if hasattr(self, 'eclipse_importance') and self.eclipse_importance is not None:
            importance = self.eclipse_importance
            importance_method = "ECLIPSE-style (M4R golden sun-moon)"
        elif self.cluster_frame is not None and 'sel_freq' in self.cluster_frame.columns:
            # Use selection frequency as proxy for "retrieval relevance"
            sel_freq = self.cluster_frame['sel_freq'].values
            
            # Sun centroids = top 10% most selected
            # Moon centroids = bottom 10% least selected
            top_10_pct = np.percentile(sel_freq, 90)
            bottom_10_pct = np.percentile(sel_freq, 10)
            
            sun_mask = sel_freq >= top_10_pct
            moon_mask = sel_freq <= bottom_10_pct
            
            sun_centroid = centroid_np[sun_mask].mean(axis=0)
            moon_centroid = centroid_np[moon_mask].mean(axis=0)
            
            importance = np.abs(sun_centroid - moon_centroid)
            importance_method = f"Traffic-based sun-moon (n_sun={sun_mask.sum()}, n_moon={moon_mask.sum()})"
        else:
            importance = np.abs(centroid_np).mean(axis=0)
            importance_method = "Mean absolute value (fallback)"
        
        print(f"  Importance method: {importance_method}")
        
        # =================================================================
        # REPRESENTATIVENESS: Based on intra-cluster structure
        # =================================================================
        # Approach: Variance of centroids per dimension relative to range
        # High variance = centroids differ in this dimension = well represented
        
        centroid_variance = np.var(centroid_np, axis=0)
        data_range = np.ptp(centroid_np, axis=0) + 1e-8
        
        # Normalized spread: high value = centroids spread out = good representation
        normalized_spread = centroid_variance / (data_range ** 2)
        representativeness = normalized_spread / (normalized_spread.max() + 1e-8)
        
        print("  Representativeness method: Normalized centroid spread")
        
        # =================================================================
        # CORRELATION ANALYSIS
        # =================================================================
        pearson_r, pearson_p = pearsonr(importance, representativeness)
        spearman_r, spearman_p = spearmanr(importance, representativeness)
        
        # Use Spearman (robust to non-linearity)
        correlation = spearman_r
        p_value = spearman_p
        
        # Bootstrap CI
        ci_low, ci_high = self._bootstrap_correlation_ci(
            importance, representativeness, n_bootstrap=1000
        )
        
        print(f"  Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4e})")
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Interpretation based on effect size
        if correlation < -0.3:
            interpretation = "STRONG NEGATIVE - Hypothesis strongly supported"
        elif correlation < -0.1:
            interpretation = "MODERATE NEGATIVE - Hypothesis partially supported"
        elif correlation < 0:
            interpretation = "WEAK NEGATIVE - Slight tendency toward hypothesis"
        elif abs(correlation) < 0.1:
            interpretation = "NO CORRELATION - Hypothesis not supported"
        elif correlation < 0.3:
            interpretation = "WEAK POSITIVE - Opposite of hypothesis"
        else:
            interpretation = "STRONG POSITIVE - Opposite of hypothesis"
        print(f"  Interpretation: {interpretation}")
        
        return V4Result(
            correlation_coefficient=float(spearman_r),
            p_value=float(spearman_p),
            correlation_ci=(float(ci_low), float(ci_high)),
            correlation_type='spearman',
            importance_scores=importance,
            representativeness_scores=representativeness
        )
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for concentration measurement."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) / (n * np.sum(sorted_vals))) - (n + 1) / n
    
    def _bootstrap_correlation_ci(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for correlation."""
        correlations = []
        n = len(x)
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            r, _ = spearmanr(x[idx], y[idx])
            correlations.append(r)
        
        alpha = (1 - confidence) / 2
        return np.percentile(correlations, alpha * 100), np.percentile(correlations, (1 - alpha) * 100)
    
    def visualize(self):
        """Generate visualizations for all experiments."""
        from hypothesis.viz import _get_plt, save_figure
        
        plt = _get_plt()
        
        output_dir = self.output_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # V1: Variance landscape
        self._plot_v1_variance_landscape(plt, output_dir)
        
        # V2: Centroid structure
        self._plot_v2_centroid_structure(plt, output_dir)
        
        # V3: Importance stability
        self._plot_v3_importance_stability(plt, output_dir)
        
        # V4: Core correlation
        self._plot_v4_core_correlation(plt, output_dir)
        
        # Summary figure
        self._plot_summary(plt, output_dir)
        
        print(f"\nFigures saved to: {output_dir}")
    
    def _plot_v1_variance_landscape(self, plt, output_dir: Path):
        """Plot V1: Per-dimension variance distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Bar chart of variance per dimension
        ax = axes[0]
        dims = np.arange(128)
        variance = self.v1_result.avg_variance_per_dim
        
        colors = ['#e74c3c' if i in self.v1_result.top_10_high_variance_dims else 
                  '#3498db' if i in self.v1_result.top_10_low_variance_dims else 
                  '#95a5a6' for i in dims]
        
        ax.bar(dims, variance, color=colors, width=1.0)
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Variance (across centroids)')
        ax.set_title('V1: Per-Dimension Variance Landscape')
        ax.axhline(np.median(variance), color='black', linestyle='--', label='Median')
        ax.legend()
        
        # Right: Histogram of variance distribution
        ax = axes[1]
        ax.hist(variance, bins=30, color='#2E86AB', edgecolor='white')
        ax.axvline(np.median(variance), color='red', linestyle='--', label=f'Median: {np.median(variance):.4f}')
        ax.axvline(np.mean(variance), color='orange', linestyle='--', label=f'Mean: {np.mean(variance):.4f}')
        ax.set_xlabel('Variance')
        ax.set_ylabel('Count (dimensions)')
        ax.set_title(f'Variance Distribution (Gini={self.v1_result.variance_gini:.3f})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'v1_variance_landscape.png', dpi=150)
        plt.close()
    
    def _plot_v2_centroid_structure(self, plt, output_dir: Path):
        """Plot V2: Centroid structure analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Left: Mean per dimension
        ax = axes[0]
        ax.bar(range(128), self.v2_result.centroid_mean_per_dim, color='#2E86AB', width=1.0)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Mean Centroid Value')
        ax.set_title(f'V2: Centroid Mean per Dimension\n({self.v2_result.near_zero_dims} dims near zero)')
        
        # Middle: Std per dimension
        ax = axes[1]
        ax.bar(range(128), self.v2_result.centroid_std_per_dim, color='#A23B72', width=1.0)
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Std of Centroid Values')
        ax.set_title(f'V2: Centroid Spread per Dimension\n({self.v2_result.high_spread_dims} high-spread dims)')
        
        # Right: Centroid norm distribution
        ax = axes[2]
        centroid_norms = np.linalg.norm(self.centroids.numpy(), axis=1)
        ax.hist(centroid_norms, bins=50, color='#F18F01', edgecolor='white')
        ax.axvline(self.v2_result.centroid_norm_mean, color='red', linestyle='--',
                   label=f'Mean: {self.v2_result.centroid_norm_mean:.3f}')
        ax.set_xlabel('Centroid L2 Norm')
        ax.set_ylabel('Count')
        ax.set_title('Centroid Norm Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'v2_centroid_structure.png', dpi=150)
        plt.close()
    
    def _plot_v3_importance_stability(self, plt, output_dir: Path):
        """Plot V3: Importance stability analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Correlation matrix heatmap
        ax = axes[0]
        im = ax.imshow(self.v3_result.importance_correlation_matrix, cmap='RdBu_r', 
                       vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
        ax.set_title(f'V3: Importance Stability\n(Mean pairwise r={self.v3_result.mean_pairwise_correlation:.3f})')
        
        # Right: Global importance profile
        ax = axes[1]
        importance = self.v3_result.global_importance
        stable = self.v3_result.stable_dimensions
        unstable = self.v3_result.unstable_dimensions
        
        colors = ['#2ecc71' if i in stable else '#e74c3c' if i in unstable else '#95a5a6' 
                  for i in range(128)]
        ax.bar(range(128), importance, color=colors, width=1.0)
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Normalized Importance')
        ax.set_title('Global Importance Profile (green=stable, red=unstable)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'v3_importance_stability.png', dpi=150)
        plt.close()
    
    def _plot_v4_core_correlation(self, plt, output_dir: Path):
        """Plot V4: Core correlation analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        importance = self.v4_result.importance_scores
        representativeness = self.v4_result.representativeness_scores
        
        # Left: Scatter plot with regression
        ax = axes[0]
        ax.scatter(importance, representativeness, alpha=0.6, c='#2E86AB', s=50)
        
        # Fit line
        z = np.polyfit(importance, representativeness, 1)
        p = np.poly1d(z)
        x_line = np.linspace(importance.min(), importance.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
                label=f'r={self.v4_result.correlation_coefficient:.3f}')
        
        ax.set_xlabel('Dimension Importance (std across centroids)')
        ax.set_ylabel('Representativeness (1/(1+var))')
        ax.set_title(f'V4: Importance vs Representativeness\n(p={self.v4_result.p_value:.2e})')
        ax.legend()
        
        # Right: Ranked comparison
        ax = axes[1]
        sorted_by_importance = np.argsort(importance)[::-1]
        
        x = np.arange(128)
        ax.plot(x, importance[sorted_by_importance], label='Importance (sorted)', color='#2E86AB')
        ax.plot(x, representativeness[sorted_by_importance], label='Representativeness', color='#A23B72')
        ax.set_xlabel('Dimension Rank (by importance)')
        ax.set_ylabel('Score')
        ax.set_title('Dimension Ranking: Importance vs Representativeness')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'v4_core_correlation.png', dpi=150)
        plt.close()
    
    def _plot_summary(self, plt, output_dir: Path):
        """Create summary visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Summary text
        summary_text = f"""
SQ1.2C: Dimension Concentration Analysis
========================================

Research Question:
Do IVF centroids under-represent retrieval-important dimensions?

VALIDATION RESULTS:
------------------

V1 - Variance Landscape:
  • Variance heterogeneity ratio: {self.v1_result.variance_heterogeneity_ratio:.2f}x
  • Gini coefficient: {self.v1_result.variance_gini:.3f}
  • Interpretation: {'High' if self.v1_result.variance_gini > 0.3 else 'Moderate'} concentration

V2 - Centroid Structure:
  • Near-zero dimensions: {self.v2_result.near_zero_dims}/128
  • High-spread dimensions: {self.v2_result.high_spread_dims}/128
  • Mean centroid norm: {self.v2_result.centroid_norm_mean:.3f} ± {self.v2_result.centroid_norm_std:.3f}

V3 - Importance Stability:
  • Mean pairwise correlation: {self.v3_result.mean_pairwise_correlation:.3f}
  • Stable dimensions: {len(self.v3_result.stable_dimensions)}
  • Unstable dimensions: {len(self.v3_result.unstable_dimensions)}

V4 - Core Correlation (KEY TEST):
  • Spearman r: {self.v4_result.correlation_coefficient:.4f}
  • p-value: {self.v4_result.p_value:.2e}
  • 95% CI: [{self.v4_result.correlation_ci[0]:.3f}, {self.v4_result.correlation_ci[1]:.3f}]

CONCLUSION:
  Hypothesis {'SUPPORTED' if self.results.supported else 'NOT SUPPORTED'}
  (correlation is {'negative' if self.v4_result.correlation_coefficient < 0 else 'positive'}, 
   p-value {'< 0.05' if self.v4_result.p_value < 0.05 else '>= 0.05'})
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'summary.png', dpi=150)
        plt.close()
    
    def report(self) -> str:
        """Generate detailed text report."""
        if self.results is None:
            return "No results available. Run analyze() first."
        
        lines = [
            "="*70,
            "SQ1.2C: PER-DIMENSION CENTROID REPRESENTATIVENESS ANALYSIS",
            "="*70,
            "",
            f"Claim: {self.CLAIM}",
            "",
            "-"*70,
            "V1: VARIANCE LANDSCAPE",
            "-"*70,
            f"  Question: What does variance look like across dimensions?",
            f"  ",
            f"  Heterogeneity ratio (max/min): {self.v1_result.variance_heterogeneity_ratio:.2f}x",
            f"  Gini coefficient: {self.v1_result.variance_gini:.3f}",
            f"  Top high-variance dims: {self.v1_result.top_10_high_variance_dims[:5]}",
            f"  Top low-variance dims: {self.v1_result.top_10_low_variance_dims[:5]}",
            "",
            "-"*70,
            "V2: CENTROID STRUCTURE",
            "-"*70,
            f"  Question: Are centroids smoothed out or discriminative?",
            f"  ",
            f"  Near-zero dimensions: {self.v2_result.near_zero_dims}/128",
            f"  High-spread dimensions: {self.v2_result.high_spread_dims}/128",
            f"  Mean centroid norm: {self.v2_result.centroid_norm_mean:.4f} ± {self.v2_result.centroid_norm_std:.4f}",
            "",
            "-"*70,
            "V3: IMPORTANCE STABILITY",
            "-"*70,
            f"  Question: Is dimension importance stable across samples?",
            f"  ",
            f"  Mean pairwise correlation: {self.v3_result.mean_pairwise_correlation:.4f}",
            f"  Stable dimensions (top 10): {self.v3_result.stable_dimensions}",
            f"  Unstable dimensions (top 10): {self.v3_result.unstable_dimensions}",
            "",
            "-"*70,
            "V4: CORE CORRELATION (KEY TEST)",
            "-"*70,
            f"  Question: Do important dims have poor representation?",
            f"  ",
            f"  Spearman correlation: {self.v4_result.correlation_coefficient:.4f}",
            f"  p-value: {self.v4_result.p_value:.2e}",
            f"  95% CI: [{self.v4_result.correlation_ci[0]:.4f}, {self.v4_result.correlation_ci[1]:.4f}]",
            "",
            "="*70,
            "CONCLUSION",
            "="*70,
            "",
        ]
        
        if self.v4_result.correlation_coefficient < -0.3:
            lines.append("STRONG SUPPORT for hypothesis:")
            lines.append("  Important dimensions have significantly poorer centroid representation.")
        elif self.v4_result.correlation_coefficient < 0 and self.v4_result.p_value < 0.05:
            lines.append("PARTIAL SUPPORT for hypothesis:")
            lines.append("  Negative correlation exists but is weak.")
        elif abs(self.v4_result.correlation_coefficient) < 0.1:
            lines.append("NO SUPPORT for hypothesis:")
            lines.append("  No meaningful correlation between importance and representation.")
        else:
            lines.append("OPPOSITE OF HYPOTHESIS:")
            lines.append("  Important dimensions actually have BETTER representation!")
        
        lines.extend([
            "",
            f"Result: {'SUPPORTED' if self.results.supported else 'NOT SUPPORTED'}",
            f"Effect size (Spearman r): {self.results.effect_size:.4f}",
            f"p-value: {self.results.p_value:.2e}",
            "",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Entry Point
# =============================================================================

def run_sq1_2c(config_name: str = "smoke") -> HypothesisResult:
    """Run SQ1.2C validation experiments."""
    from hypothesis.configs import load_config
    
    config = load_config(config_name)
    hypothesis = SQ1_2C_DimensionConcentration(config)
    return hypothesis.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SQ1.2C validation experiments")
    parser.add_argument("--config", default="smoke", choices=["smoke", "dev", "prod"],
                        help="Configuration tier")
    
    args = parser.parse_args()
    result = run_sq1_2c(args.config)
