"""
Hypothesis H_PROJ: Projection Spectrum → Hub Traffic

Claim: Centroids aligned with dominant eigenvectors of the projection-induced 
       metric M = WW^T receive disproportionate routing traffic (become hubs).

Background:
    The XTR/ColBERT projection layer (768d → 128d) is a linear transformation W.
    After projection and L2-normalization, cosine similarity operates under the
    induced metric M = WW^T. The eigenspectrum of M determines which directions
    are "cheap" (high eigenvalue → high achievable similarity) vs "expensive"
    (low eigenvalue → similarity attenuated).
    
    Prior work (Clavié et al. 2025, Kisung You 2025) establishes that:
    1. Projections induce non-uniform geometry
    2. Anisotropic geometry leads to hubness under cosine similarity
    
    This experiment bridges these findings to IVF routing: centroids in "cheap"
    directions should become hubs because they're easier to reach under cosine.

Key Metrics:
    - projection_alignment_k: Fraction of centroid variance in top-k eigenspace
      = ||proj_{top-k}(c)||² / ||c||²
    - traffic_share (B1): Existing metric - fraction of total routing traffic

Expected Finding:
    - Positive correlation: high alignment → high traffic
    - Hub centroids cluster in dominant eigenspace

References:
    - COT_projection_spectrum_experiment.md
    - SQ1_2B_projection_spectrum_motivation.md
    - Clavié et al. (2025): "Simple Projection Variants Improve ColBERT"
    - Kisung You (2025): "Semantics at an Angle"
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from hypothesis.hypotheses.template import HypothesisTest, HypothesisResult
from hypothesis.stats import bootstrap_ci
from hypothesis.viz import save_figure


class H_ProjectionSpectrum(HypothesisTest):
    """
    Hypothesis: Projection eigenspectrum predicts hub traffic concentration.
    
    Centroids aligned with dominant eigenvectors of M = WW^T receive more
    routing traffic because these directions are "geometrically cheap" under
    the projection-induced similarity metric.
    """
    
    HYPOTHESIS_ID = "H_PROJ"
    HYPOTHESIS_NAME = "Projection Spectrum → Hub Traffic"
    CLAIM = "Centroids aligned with dominant eigenvectors of projection metric M=WW^T receive disproportionate routing traffic"
    
    # Analysis parameters
    DEFAULT_K = 10  # Number of top eigenvectors for primary analysis
    SENSITIVITY_K_VALUES = [5, 10, 20, 50]  # For sensitivity analysis
    
    def __init__(self, config):
        super().__init__(config)
        
        # Projection analysis results (populated by setup)
        self.W: Optional[torch.Tensor] = None
        self.M: Optional[torch.Tensor] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[torch.Tensor] = None
        self.centroids: Optional[torch.Tensor] = None
        self.alignment_scores: Optional[Dict[int, np.ndarray]] = None
    
    def setup(self):
        """Load data and compute projection alignment metrics."""
        from hypothesis.configs import ensure_output_dirs
        from hypothesis.data.standardized_tables import ClusterFrameBuilder
        
        ensure_output_dirs(self.config)
        
        # Load cluster_frame (contains traffic_share)
        cluster_builder = ClusterFrameBuilder(self.config)
        self.cluster_frame = cluster_builder.build()
        print(f"Loaded cluster_frame: {self.cluster_frame.shape}")
        
        # Extract projection matrix and compute eigendecomposition
        self._extract_projection_matrix()
        
        # Load centroids from index
        self._load_centroids()
        
        # Compute alignment scores for all k values
        self._compute_all_alignments()
        
        # Add alignment columns to cluster_frame
        self._extend_cluster_frame()
    
    def _extract_projection_matrix(self):
        """Extract W from XTR model and compute eigendecomposition of M = WW^T."""
        print("\nExtracting projection matrix from XTR model...")
        
        from warp.modeling.xtr import build_xtr_model
        
        # Build XTR and extract W
        xtr = build_xtr_model()
        self.W = xtr.linear.linear.weight.detach().cpu()  # Shape: [128, 768]
        print(f"  W shape: {self.W.shape}")
        
        # Compute induced metric M = W @ W.T
        self.M = self.W @ self.W.T  # Shape: [128, 128]
        print(f"  M shape: {self.M.shape}")
        
        # Eigendecomposition (M is symmetric PSD)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.M)
        
        # Sort descending (eigh returns ascending)
        self.eigenvalues = eigenvalues.flip(0).numpy()
        self.eigenvectors = eigenvectors.flip(1)  # Shape: [128, 128], columns are eigenvectors
        
        # Report spectrum concentration
        total_trace = self.eigenvalues.sum()
        cumsum = np.cumsum(self.eigenvalues)
        for k in [5, 10, 20, 50]:
            pct = cumsum[k-1] / total_trace * 100
            print(f"  Top-{k} eigenvalues explain {pct:.1f}% of trace")
    
    def _load_centroids(self):
        """Load centroids from index and compute PCA for comparison."""
        index_path = Path(self.config.paths.index_path)
        centroids_path = index_path / "centroids.npy"
        
        if not centroids_path.exists():
            raise FileNotFoundError(f"centroids.npy not found at {centroids_path}")
        
        self.centroids = torch.from_numpy(np.load(centroids_path)).float()
        print(f"  Loaded centroids: {self.centroids.shape}")
        
        # Compute centroid PCA for comparison with WW^T
        print("\nComputing centroid PCA for comparison...")
        centroids_centered = self.centroids - self.centroids.mean(dim=0)
        U, S, Vh = torch.linalg.svd(centroids_centered, full_matrices=False)
        self.pca_eigenvalues = (S ** 2).numpy()  # Variance = singular values squared
        self.pca_eigenvectors = Vh.T  # [128, 128]
        
        # Report PCA concentration
        total_pca = self.pca_eigenvalues.sum()
        cumsum_pca = np.cumsum(self.pca_eigenvalues)
        for k in [2, 5, 10]:
            pct = cumsum_pca[k-1] / total_pca * 100
            expected = k / 128 * 100
            print(f"  Centroid PCA top-{k}: {pct:.1f}% (vs {expected:.1f}% uniform) → {pct/expected:.1f}x")
        
        # Compute alignment between WW^T eigenvectors and PCA directions
        alignment_matrix = torch.abs(self.eigenvectors.T @ self.pca_eigenvectors)
        self.wwt_pca_alignment = alignment_matrix.numpy()
        top1_align = self.wwt_pca_alignment[0, 0]
        print(f"  WW^T_eig[0] · PCA[0] alignment: {top1_align:.3f}")
        
        # Compute mean-direction cosine for confound check (Section 5.3.2 comparison)
        print("\nComputing mean-direction cosine for confound check...")
        mean_centroid = self.centroids.mean(dim=0)
        mean_centroid_normalized = mean_centroid / mean_centroid.norm()
        centroid_norms = self.centroids.norm(dim=1)
        self.mean_cosine = ((self.centroids @ mean_centroid_normalized) / centroid_norms).numpy()
        print(f"  Mean-direction cosine: mean={self.mean_cosine.mean():.3f}, std={self.mean_cosine.std():.3f}")
    
    def _compute_alignment(self, k: int) -> np.ndarray:
        """
        Compute alignment of each centroid with top-k eigenspace.
        
        alignment_k(c) = ||proj_{top-k}(c)||² / ||c||²
                       = Σᵢ₌₁ᵏ (c · vᵢ)² / ||c||²
        
        Args:
            k: Number of top eigenvectors to use
            
        Returns:
            Array of alignment scores, one per centroid
        """
        top_k_vecs = self.eigenvectors[:, :k]  # [128, k]
        
        # Project centroids onto top-k subspace: [C, 128] @ [128, k] = [C, k]
        projections = self.centroids @ top_k_vecs
        
        # Squared norm of projection
        proj_norms_sq = (projections ** 2).sum(dim=1)  # [C]
        
        # Squared norm of centroids
        centroid_norms_sq = (self.centroids ** 2).sum(dim=1)  # [C]
        
        # Alignment = fraction of variance in top-k subspace
        alignment = proj_norms_sq / centroid_norms_sq.clamp(min=1e-8)
        
        return alignment.numpy()
    
    def _compute_all_alignments(self):
        """Compute alignment for all k values."""
        print("\nComputing centroid alignments...")
        self.alignment_scores = {}
        
        for k in self.SENSITIVITY_K_VALUES:
            self.alignment_scores[k] = self._compute_alignment(k)
            mean_align = self.alignment_scores[k].mean()
            print(f"  k={k}: mean alignment = {mean_align:.3f}")
    
    def _extend_cluster_frame(self):
        """Add alignment columns to cluster_frame."""
        for k, alignment in self.alignment_scores.items():
            col_name = f'projection_alignment_k{k}'
            self.cluster_frame[col_name] = alignment
        
        print(f"\nAdded {len(self.alignment_scores)} alignment columns to cluster_frame")
    
    def analyze(self) -> HypothesisResult:
        """Run correlation analysis between alignment and traffic."""
        df = self.cluster_frame
        k = self.DEFAULT_K
        align_col = f'projection_alignment_k{k}'
        
        # Filter to centroids with valid traffic data
        valid = df[df['traffic_share'].notna() & (df['traffic_share'] > 0)].copy()
        
        if len(valid) < 100:
            return self._empty_result("Insufficient data with traffic")
        
        # PRIMARY TEST: Spearman correlation
        rho, pval = spearmanr(valid[align_col], valid['traffic_share'])
        
        # Bootstrap CI for correlation
        def corr_func(indices):
            subset = valid.iloc[indices]
            return spearmanr(subset[align_col], subset['traffic_share'])[0]
        
        n = len(valid)
        bootstrap_rhos = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            indices = rng.choice(n, size=n, replace=True)
            bootstrap_rhos.append(corr_func(indices))
        
        ci_lower = np.percentile(bootstrap_rhos, 2.5)
        ci_upper = np.percentile(bootstrap_rhos, 97.5)
        
        # Sensitivity analysis: correlation for different k values
        sensitivity = {}
        for k_val in self.SENSITIVITY_K_VALUES:
            col = f'projection_alignment_k{k_val}'
            r, p = spearmanr(valid[col], valid['traffic_share'])
            sensitivity[k_val] = {'rho': r, 'pval': p}
        
        # Stratified analysis: top vs bottom alignment quartiles
        valid['align_quartile'] = pd.qcut(
            valid[align_col], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high']
        )
        
        q1 = valid[valid['align_quartile'] == 'Q1_low']
        q2 = valid[valid['align_quartile'] == 'Q2']
        q3 = valid[valid['align_quartile'] == 'Q3']
        q4 = valid[valid['align_quartile'] == 'Q4_high']
        
        q1_traffic = q1['traffic_share'].sum()
        q2_traffic = q2['traffic_share'].sum()
        q3_traffic = q3['traffic_share'].sum()
        q4_traffic = q4['traffic_share'].sum()
        total_traffic = valid['traffic_share'].sum()
        
        # Compute spectrum statistics
        total_trace = self.eigenvalues.sum()
        cumsum = np.cumsum(self.eigenvalues)
        variance_explained = {k: cumsum[k-1] / total_trace for k in self.SENSITIVITY_K_VALUES}
        
        # Compute PCA comparison statistics
        total_pca = self.pca_eigenvalues.sum()
        cumsum_pca = np.cumsum(self.pca_eigenvalues)
        pca_variance_explained = {k: cumsum_pca[k-1] / total_pca for k in self.SENSITIVITY_K_VALUES}
        
        # Compute concentration ratios (vs uniform baseline)
        dim = 128
        wwt_concentration = {k: (cumsum[k-1] / total_trace) / (k / dim) for k in [3, 5, 10]}
        pca_concentration = {k: (cumsum_pca[k-1] / total_pca) / (k / dim) for k in [3, 5, 10]}
        
        # CONFOUND CHECK: correlation between alignment_10 and mean-direction cosine
        # This tests whether projection-spectrum alignment is redundant with the 
        # mean-direction alignment already reported in Section 5.3.2
        valid['mean_cosine'] = self.mean_cosine[valid.index]
        confound_rho, confound_pval = spearmanr(valid[align_col], valid['mean_cosine'])
        
        # Also compute partial correlation: alignment → traffic, controlling for mean_cosine
        partial_rho = None
        partial_pval = None
        try:
            import pingouin as pg
            partial_result = pg.partial_corr(
                data=valid,
                x=align_col,
                y='traffic_share',
                covar='mean_cosine',
                method='spearman'
            )
            partial_rho = partial_result['r'].values[0]
            partial_pval = partial_result['p-val'].values[0]
            print(f"\n  Confound check (alignment_10 vs mean_cosine): ρ = {confound_rho:.3f}")
            print(f"  Partial corr (alignment → traffic | mean_cosine): ρ = {partial_rho:.3f}, p = {partial_pval:.2e}")
        except ImportError:
            print(f"\n  Confound check (alignment_10 vs mean_cosine): ρ = {confound_rho:.3f}")
            print("  (pingouin not installed, skipping partial correlation)")
        except Exception as e:
            print(f"\n  Confound check failed: {e}")
        
        # Build result
        statistics = {
            'primary_k': k,
            'correlation_rho': rho,
            'correlation_pval': pval,
            'correlation_ci': (ci_lower, ci_upper),
            'sensitivity_by_k': sensitivity,
            'variance_explained_by_k': variance_explained,
            'pca_variance_explained_by_k': pca_variance_explained,
            'wwt_concentration_ratio': wwt_concentration,
            'pca_concentration_ratio': pca_concentration,
            'wwt_pca_top1_alignment': self.wwt_pca_alignment[0, 0],
            'q1_alignment_mean': q1[align_col].mean(),
            'q4_alignment_mean': q4[align_col].mean(),
            'q1_traffic_share': q1_traffic / total_traffic,
            'q4_traffic_share': q4_traffic / total_traffic,
            'q2_traffic_share': q2_traffic / total_traffic,
            'q3_traffic_share': q3_traffic / total_traffic,
            'traffic_concentration_ratio': q4_traffic / q1_traffic if q1_traffic > 0 else np.inf,
            'n_centroids_analyzed': len(valid),
            'eigenvalue_top1': self.eigenvalues[0],
            'eigenvalue_top10_sum': self.eigenvalues[:10].sum(),
            'eigenvalue_total_trace': total_trace,
            # Confound check statistics
            'confound_alignment_vs_mean_cosine_rho': confound_rho,
            'confound_alignment_vs_mean_cosine_pval': confound_pval,
            'partial_corr_alignment_traffic_given_mean': partial_rho,
            'partial_corr_pval': partial_pval,
        }
        
        # Determine if hypothesis is supported
        # Criteria: significant positive correlation
        supported = (rho > 0.1) and (pval < 0.01)
        
        self.results = HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=supported,
            effect_size=rho,
            effect_size_ci=(ci_lower, ci_upper),
            p_value=pval,
            statistics=statistics,
            config_name=self.config.name,
            n_observations=len(valid),
            timestamp=datetime.now().isoformat()
        )
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print analysis summary."""
        r = self.results
        s = r.statistics
        
        print("\n" + "=" * 70)
        print(f"HYPOTHESIS {r.hypothesis_id}: {r.hypothesis_name}")
        print("=" * 70)
        print(f"\nClaim: {r.claim}")
        print(f"\nRESULT: {'SUPPORTED' if r.supported else 'NOT SUPPORTED'}")
        
        print(f"\n--- Spectrum Concentration Comparison ---")
        print("  WW^T (projection matrix):")
        for k in [3, 5, 10]:
            if k in s.get('wwt_concentration_ratio', {}):
                ratio = s['wwt_concentration_ratio'][k]
                # Get variance from the computed cumsum, not from variance_explained_by_k which may not have k=3
                pct = ratio * (k / 128) * 100
                print(f"    Top-{k}: {pct:.1f}% ({ratio:.1f}× uniform)")
        print("  Centroid PCA (data):")
        for k in [3, 5, 10]:
            if k in s.get('pca_concentration_ratio', {}):
                ratio = s['pca_concentration_ratio'][k]
                pct = ratio * (k / 128) * 100
                print(f"    Top-{k}: {pct:.1f}% ({ratio:.1f}× uniform)")
        
        print(f"\n--- WW^T ↔ PCA Alignment ---")
        print(f"  Top-1 alignment: {s.get('wwt_pca_top1_alignment', 0):.3f}")
        
        print(f"\n--- Primary Correlation (k={s['primary_k']}) ---")
        print(f"  Spearman ρ = {s['correlation_rho']:.4f}")
        print(f"  95% CI: [{s['correlation_ci'][0]:.4f}, {s['correlation_ci'][1]:.4f}]")
        print(f"  p-value = {s['correlation_pval']:.2e}")
        
        print(f"\n--- Traffic by Alignment Quartile ---")
        print(f"  Q1 (low alignment):  {s['q1_traffic_share']*100:.1f}%")
        print(f"  Q2:                  {s['q2_traffic_share']*100:.1f}%")
        print(f"  Q3:                  {s['q3_traffic_share']*100:.1f}%")
        print(f"  Q4 (high alignment): {s['q4_traffic_share']*100:.1f}%")
        print(f"  Ratio Q4/Q1: {s['traffic_concentration_ratio']:.2f}×")
        
        print(f"\n--- Confound Check: Independence from Mean-Direction ---")
        confound_rho = s.get('confound_alignment_vs_mean_cosine_rho')
        confound_pval = s.get('confound_alignment_vs_mean_cosine_pval')
        partial_rho = s.get('partial_corr_alignment_traffic_given_mean')
        partial_pval = s.get('partial_corr_pval')
        
        if confound_rho is not None:
            print(f"  corr(alignment_10, mean_cosine): ρ = {confound_rho:.4f}")
            print(f"  p-value = {confound_pval:.2e}")
        else:
            print(f"  corr(alignment_10, mean_cosine): N/A")
        
        if partial_rho is not None:
            print(f"  Partial corr(alignment → traffic | mean_cosine): ρ = {partial_rho:.4f}")
            print(f"  p-value = {partial_pval:.2e}")
        else:
            print(f"  Partial corr: (pingouin not installed)")
        print("=" * 70)
    
    def visualize(self):
        """Generate the two key figures."""
        self._plot_spectrum_comparison()
        self._plot_alignment_scatter_simple()  # Eigenvector 1 alignment vs traffic
    
    def _plot_alignment_scatter_simple(self):
        """
        Traffic figure: Many binned means showing eigenvector 1 alignment vs traffic.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        df = self.cluster_frame
        valid = df[df['traffic_share'].notna() & (df['traffic_share'] > 0)].copy()
        traffic = valid['traffic_share'].values * 1e5
        
        valid_indices = valid.index.values
        centroids_valid = self.centroids[valid_indices]
        
        # Eigenvector 1 alignment
        v_1 = self.eigenvectors[:, 0]
        align_1 = (centroids_valid @ v_1).numpy() ** 2
        
        # Bin into many bins for smooth curve
        n_bins = 50
        bin_edges = np.percentile(align_1, np.linspace(0, 100, n_bins + 1))
        bin_means = []
        bin_centers = []
        
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (align_1 >= bin_edges[i]) & (align_1 <= bin_edges[i+1])
            else:
                mask = (align_1 >= bin_edges[i]) & (align_1 < bin_edges[i+1])
            if mask.sum() > 0:
                bin_means.append(traffic[mask].mean())
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        
        bin_means = np.array(bin_means)
        bin_centers = np.array(bin_centers)
        
        # Correlation
        rho, _ = spearmanr(align_1, traffic)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Points
        ax.scatter(bin_centers, bin_means, s=40, c='#E74C3C', alpha=0.7, edgecolors='darkred', linewidth=0.5, zorder=3)
        
        # Smoothed trend line (polynomial fit)
        z = np.polyfit(bin_centers, bin_means, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        ax.plot(x_smooth, p(x_smooth), '-', color='#C0392B', linewidth=2.5, alpha=0.8, zorder=2)
        
        ax.set_xlabel('Alignment to Eigenvector 1', fontsize=11)
        ax.set_ylabel('Mean Traffic (×10⁻⁵)', fontsize=11)
        ax.grid(True, alpha=0.3, zorder=1)
        
        plt.tight_layout()
        save_figure(fig, f'{self.HYPOTHESIS_ID}_ev1_traffic', str(self.output_dir))
        plt.close(fig)
        
        print(f"\nEigenvector 1: ρ = {rho:.3f}")
    
    def _plot_alignment_scatter_top3(self):
        """
        Traffic figure: Many binned means of top-3 eigenvector alignment vs traffic.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        df = self.cluster_frame
        valid = df[df['traffic_share'].notna() & (df['traffic_share'] > 0)].copy()
        traffic = valid['traffic_share'].values * 1e5
        
        valid_indices = valid.index.values
        centroids_valid = self.centroids[valid_indices]
        
        # Top-3 alignment
        top3_vecs = self.eigenvectors[:, :3]
        projections = centroids_valid @ top3_vecs
        align_top3 = (projections ** 2).sum(dim=1).numpy()
        
        # Bin into many bins
        n_bins = 50
        bin_edges = np.percentile(align_top3, np.linspace(0, 100, n_bins + 1))
        bin_means = []
        bin_centers = []
        
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (align_top3 >= bin_edges[i]) & (align_top3 <= bin_edges[i+1])
            else:
                mask = (align_top3 >= bin_edges[i]) & (align_top3 < bin_edges[i+1])
            if mask.sum() > 0:
                bin_means.append(traffic[mask].mean())
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        
        bin_means = np.array(bin_means)
        bin_centers = np.array(bin_centers)
        
        # Correlation
        rho, _ = spearmanr(align_top3, traffic)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Points
        ax.scatter(bin_centers, bin_means, s=40, c='#3498DB', alpha=0.7, edgecolors='darkblue', linewidth=0.5, zorder=3)
        
        # Smoothed trend line
        z = np.polyfit(bin_centers, bin_means, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        ax.plot(x_smooth, p(x_smooth), '-', color='#2980B9', linewidth=2.5, alpha=0.8, zorder=2)
        
        ax.set_xlabel('Alignment to Top-3 Eigenvectors', fontsize=11)
        ax.set_ylabel('Mean Traffic (×10⁻⁵)', fontsize=11)
        ax.grid(True, alpha=0.3, zorder=1)
        
        plt.tight_layout()
        save_figure(fig, f'{self.HYPOTHESIS_ID}_top3_traffic', str(self.output_dir))
        plt.close(fig)
        
        print(f"Top-3 eigenvectors: ρ = {rho:.3f}")
    
    def _plot_option_b_lift_curve(self):
        """
        Option B: Lift/Lorenz curve for eigenvector 1.
        
        Sort centroids by alignment to eigenvector 1, plot cumulative traffic 
        vs cumulative centroids. Shows how much "lift" alignment gives.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        df = self.cluster_frame
        valid = df[df['traffic_share'].notna() & (df['traffic_share'] > 0)].copy()
        traffic = valid['traffic_share'].values
        
        valid_indices = valid.index.values
        centroids_valid = self.centroids[valid_indices]
        
        # Eigenvector 1 alignment
        v_1 = self.eigenvectors[:, 0]
        alignment_1 = (centroids_valid @ v_1).numpy() ** 2
        
        # Sort by alignment (descending)
        sorted_indices = np.argsort(-alignment_1)
        sorted_traffic = traffic[sorted_indices]
        
        # Cumulative sums (normalized to %)
        n = len(sorted_traffic)
        cum_centroids = np.arange(1, n + 1) / n * 100
        cum_traffic = np.cumsum(sorted_traffic) / sorted_traffic.sum() * 100
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Actual lift curve
        ax.plot(cum_centroids, cum_traffic, 'b-', linewidth=2.5, label='Eigenvector 1 alignment')
        
        # Diagonal (random baseline)
        ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, alpha=0.5, label='Random (no lift)')
        
        # Fill the lift area
        ax.fill_between(cum_centroids, cum_centroids, cum_traffic, alpha=0.2, color='blue')
        
        # Mark key points
        for pct in [10, 25, 50]:
            idx = int(n * pct / 100) - 1
            captured = cum_traffic[idx]
            ax.scatter([pct], [captured], s=80, c='red', zorder=5, edgecolors='darkred')
            ax.annotate(f'Top {pct}%\ncaptures {captured:.0f}%',
                       xy=(pct, captured),
                       xytext=(pct + 8, captured - 5),
                       fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        ax.set_xlabel('% of Centroids (sorted by alignment)', fontsize=12)
        ax.set_ylabel('% of Traffic Captured', fontsize=12)
        ax.set_title('Option B: Lift Curve for Eigenvector 1', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Compute Gini coefficient (area between curve and diagonal)
        gini = (cum_traffic.sum() - cum_centroids.sum()) / cum_centroids.sum()
        ax.text(60, 20, f'Gini = {gini:.3f}', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_figure(fig, f'{self.HYPOTHESIS_ID}_option_b_lift_curve', str(self.output_dir))
        plt.close(fig)
        
        print(f"\nOption B - Lift curve for eigenvector 1:")
        for pct in [10, 25, 50]:
            idx = int(n * pct / 100) - 1
            print(f"  Top {pct}% by alignment capture {cum_traffic[idx]:.1f}% of traffic")
        print(f"  Gini coefficient: {gini:.3f}")
    
    def _plot_option_c_q1_vs_q4(self):
        """
        Option C: Q1 vs Q4 traffic comparison per eigenvector.
        
        For each eigenvector, show side-by-side bars of mean traffic 
        for bottom quartile (Q1) vs top quartile (Q4) by alignment.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        df = self.cluster_frame
        valid = df[df['traffic_share'].notna() & (df['traffic_share'] > 0)].copy()
        traffic = valid['traffic_share'].values
        
        valid_indices = valid.index.values
        centroids_valid = self.centroids[valid_indices]
        
        n_eigenvectors = 8
        q1_means = []
        q4_means = []
        ratios = []
        
        for i in range(n_eigenvectors):
            v_i = self.eigenvectors[:, i]
            alignment_i = (centroids_valid @ v_i).numpy() ** 2
            
            # Quartile thresholds
            q1_thresh = np.percentile(alignment_i, 25)
            q4_thresh = np.percentile(alignment_i, 75)
            
            q1_mask = alignment_i <= q1_thresh
            q4_mask = alignment_i >= q4_thresh
            
            q1_mean = traffic[q1_mask].mean() * 10000  # Scale for readability
            q4_mean = traffic[q4_mask].mean() * 10000
            
            q1_means.append(q1_mean)
            q4_means.append(q4_mean)
            ratios.append(q4_mean / q1_mean if q1_mean > 0 else 0)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(1, n_eigenvectors + 1)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, q1_means, width, label='Q1 (low alignment)', 
                       color='#3498DB', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, q4_means, width, label='Q4 (high alignment)', 
                       color='#E74C3C', edgecolor='black', linewidth=0.5)
        
        # Add ratio annotations
        for i, ratio in enumerate(ratios):
            y_pos = max(q1_means[i], q4_means[i]) + 0.3
            ax.text(x[i], y_pos, f'{ratio:.2f}×', ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Eigenvector Index', fontsize=12)
        ax.set_ylabel('Mean Traffic (×10⁻⁴)', fontsize=12)
        ax.set_title('Option C: Traffic by Alignment Quartile per Eigenvector', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, f'{self.HYPOTHESIS_ID}_option_c_q1_vs_q4', str(self.output_dir))
        plt.close(fig)
        
        print(f"\nOption C - Q4/Q1 traffic ratio by eigenvector:")
        for i, ratio in enumerate(ratios):
            print(f"  Eigenvector {i+1}: {ratio:.2f}×")
    
    def _plot_spectrum_comparison(self):
        """Figure 2: Cumulative variance - WW^T vs Centroid PCA (conduit argument)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Compute cumulative variance for both
        total_wwt = self.eigenvalues.sum()
        cumvar_wwt = np.cumsum(self.eigenvalues) / total_wwt * 100
        
        total_pca = self.pca_eigenvalues.sum()
        cumvar_pca = np.cumsum(self.pca_eigenvalues) / total_pca * 100
        
        # Uniform baseline
        k_range = np.arange(1, 129)
        uniform_line = k_range / 128 * 100
        
        # Plot
        ax.plot(k_range, cumvar_pca, 'r-', linewidth=2.5, label='Centroid PCA (data)', zorder=3)
        ax.plot(k_range, cumvar_wwt, 'b-', linewidth=2.5, label='WW$^T$ (projection)', zorder=3)
        ax.plot(k_range, uniform_line, 'k--', linewidth=1.5, alpha=0.5, label='Uniform baseline')
        
        # Highlight key k values with markers only
        for k in [5, 10]:
            ax.scatter([k], [cumvar_pca[k-1]], s=80, c='red', zorder=5, edgecolors='darkred')
            ax.scatter([k], [cumvar_wwt[k-1]], s=80, c='blue', zorder=5, edgecolors='darkblue')
        
        ax.set_xlabel('Number of Components (k)', fontsize=12)
        ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax.set_title('Spectrum Concentration: Projection vs Data', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 70)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, f'{self.HYPOTHESIS_ID}_spectrum_comparison', str(self.output_dir))
        plt.close(fig)
    
    def _empty_result(self, reason: str) -> HypothesisResult:
        """Return empty result when analysis cannot proceed."""
        return HypothesisResult(
            hypothesis_id=self.HYPOTHESIS_ID,
            hypothesis_name=self.HYPOTHESIS_NAME,
            claim=self.CLAIM,
            supported=False,
            effect_size=0.0,
            effect_size_ci=(0.0, 0.0),
            p_value=1.0,
            statistics={'error': reason},
            config_name=self.config.name,
            n_observations=0,
            timestamp=datetime.now().isoformat()
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_h_projection_spectrum(config_name: str = "dev") -> HypothesisResult:
    """Run H_PROJ hypothesis test."""
    from hypothesis.configs import load_config
    
    config = load_config(config_name)
    test = H_ProjectionSpectrum(config)
    test.setup()
    result = test.analyze()
    test.visualize()
    result.save(str(test.output_dir))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Projection Spectrum Hypothesis Test")
    parser.add_argument("--config", type=str, default="dev", 
                       choices=["smoke", "dev", "prod"],
                       help="Configuration to use")
    args = parser.parse_args()
    
    result = run_h_projection_spectrum(args.config)
    print(f"\nHypothesis {'SUPPORTED' if result.supported else 'NOT SUPPORTED'}")
