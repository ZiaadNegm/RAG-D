"""
Visualization Utilities for Hypothesis Testing

Provides standard plotting functions for:
- Distributions and histograms
- Scatter plots with regression lines
- Stratified comparisons
- Correlation heatmaps
- Pareto/CDF curves
- Hypothesis-specific visualizations

All plots are designed for publication quality and consistent styling.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

# Lazy imports for matplotlib (can be slow)
_plt = None
_sns = None
_sns_available = True


def _get_plt():
    """Lazy import matplotlib."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend by default
        import matplotlib.pyplot as plt
        _plt = plt
        # Set default style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('ggplot')  # Fallback style
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight'
        })
    return _plt


def _get_sns():
    """Lazy import seaborn (optional)."""
    global _sns, _sns_available
    if _sns is None and _sns_available:
        try:
            import seaborn as sns
            _sns = sns
            sns.set_palette("husl")
        except ImportError:
            _sns_available = False
            warnings.warn("seaborn not available, using matplotlib fallbacks")
    return _sns


# =============================================================================
# Color Palettes and Styling
# =============================================================================

HYPOTHESIS_PALETTE = {
    'primary': '#2E86AB',     # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'success': '#C73E1D',      # Red (for high values)
    'neutral': '#6B717E',      # Gray
    'highlight': '#FFD23F',    # Yellow
}

QUARTILE_COLORS = {
    'Q1': '#3498db',  # Blue (low)
    'Q2': '#2ecc71',  # Green
    'Q3': '#f39c12',  # Orange
    'Q4': '#e74c3c',  # Red (high)
}


def get_quartile_palette(labels: List[str] = None) -> Dict[str, str]:
    """Get color palette for quartile bins."""
    if labels is None:
        return QUARTILE_COLORS
    
    base_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    return {label: base_colors[i % len(base_colors)] for i, label in enumerate(labels)}


# =============================================================================
# Distribution Plots
# =============================================================================

def plot_distribution(
    data: pd.Series,
    title: str = None,
    xlabel: str = None,
    log_scale: bool = False,
    bins: int = 50,
    kde: bool = True,
    percentile_lines: List[float] = None,
    save_path: str = None,
    ax=None
):
    """
    Plot distribution with optional KDE and percentile lines.
    
    Args:
        data: Series of values
        title: Plot title
        xlabel: X-axis label
        log_scale: Use log scale for x-axis
        bins: Number of histogram bins
        kde: Add KDE overlay
        percentile_lines: List of percentiles to mark (e.g., [25, 50, 75, 95])
        save_path: Path to save figure
        ax: Matplotlib axes (creates new figure if None)
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = data.dropna()
    
    if log_scale:
        plot_data = np.log10(plot_data[plot_data > 0] + 1)
        if xlabel:
            xlabel = f"log₁₀({xlabel} + 1)"
    
    # Histogram with KDE
    if sns is not None:
        sns.histplot(plot_data, bins=bins, kde=kde, ax=ax, color=HYPOTHESIS_PALETTE['primary'])
    else:
        ax.hist(plot_data, bins=bins, color=HYPOTHESIS_PALETTE['primary'], alpha=0.7, edgecolor='white')
        if kde:
            from scipy import stats
            density = stats.gaussian_kde(plot_data)
            x_range = np.linspace(plot_data.min(), plot_data.max(), 100)
            ax2 = ax.twinx()
            ax2.plot(x_range, density(x_range), color=HYPOTHESIS_PALETTE['secondary'], linewidth=2)
            ax2.set_ylabel('')
            ax2.set_yticks([])
    
    # Add percentile lines
    if percentile_lines:
        for p in percentile_lines:
            val = np.percentile(plot_data, p)
            ax.axvline(val, color=HYPOTHESIS_PALETTE['secondary'], linestyle='--', alpha=0.7)
            ax.text(val, ax.get_ylim()[1] * 0.9, f'p{p}', rotation=90, va='top')
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


def plot_dual_distribution(
    data1: pd.Series,
    data2: pd.Series,
    label1: str = "Group 1",
    label2: str = "Group 2",
    title: str = None,
    xlabel: str = None,
    save_path: str = None
):
    """
    Plot two distributions overlaid for comparison.
    
    Args:
        data1, data2: Series of values
        label1, label2: Legend labels
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if sns is not None:
        sns.kdeplot(data1.dropna(), ax=ax, label=label1, color=HYPOTHESIS_PALETTE['primary'], fill=True, alpha=0.3)
        sns.kdeplot(data2.dropna(), ax=ax, label=label2, color=HYPOTHESIS_PALETTE['secondary'], fill=True, alpha=0.3)
    else:
        from scipy import stats
        for data, label, color in [(data1, label1, HYPOTHESIS_PALETTE['primary']), 
                                    (data2, label2, HYPOTHESIS_PALETTE['secondary'])]:
            clean_data = data.dropna()
            if len(clean_data) > 1:
                density = stats.gaussian_kde(clean_data)
                x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                ax.fill_between(x_range, density(x_range), alpha=0.3, color=color, label=label)
                ax.plot(x_range, density(x_range), color=color, linewidth=2)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


# =============================================================================
# Scatter Plots with Regression
# =============================================================================

def plot_scatter_with_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    alpha: float = 0.5,
    show_regression: bool = True,
    annotate_corr: bool = True,
    log_x: bool = False,
    log_y: bool = False,
    save_path: str = None,
    ax=None
):
    """
    Scatter plot with optional regression line and correlation annotation.
    
    Args:
        df: DataFrame with data
        x, y: Column names for x and y axes
        hue: Optional column for color coding
        title, xlabel, ylabel: Labels
        alpha: Point transparency
        show_regression: Add regression line
        annotate_corr: Add Spearman correlation annotation
        log_x, log_y: Use log scale
        save_path: Path to save figure
        ax: Matplotlib axes
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    from scipy.stats import spearmanr
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    plot_df = df[[x, y] + ([hue] if hue else [])].dropna()
    
    # Scatter
    if hue:
        if sns is not None:
            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, alpha=alpha, ax=ax)
        else:
            unique_hues = plot_df[hue].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_hues)))
            for h, c in zip(unique_hues, colors):
                mask = plot_df[hue] == h
                ax.scatter(plot_df.loc[mask, x], plot_df.loc[mask, y], alpha=alpha, label=h, c=[c])
            ax.legend()
    else:
        ax.scatter(plot_df[x], plot_df[y], alpha=alpha, color=HYPOTHESIS_PALETTE['primary'])
    
    # Regression line
    if show_regression and not hue:
        if sns is not None:
            sns.regplot(data=plot_df, x=x, y=y, scatter=False, ax=ax, 
                       color=HYPOTHESIS_PALETTE['secondary'], line_kws={'linewidth': 2})
        else:
            # Manual regression line
            from scipy import stats
            slope, intercept, r, p, se = stats.linregress(plot_df[x], plot_df[y])
            x_line = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=HYPOTHESIS_PALETTE['secondary'], linewidth=2)
    
    # Correlation annotation
    if annotate_corr:
        corr, pval = spearmanr(plot_df[x], plot_df[y])
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        ax.annotate(f'ρ = {corr:.3f}{sig}\nn = {len(plot_df):,}',
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: List[str],
    hue: str = None,
    title: str = None,
    save_path: str = None
):
    """
    Create scatter matrix for multiple variables.
    
    Args:
        df: DataFrame with data
        columns: Columns to include
        hue: Optional column for color coding
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        PairGrid object
    """
    plt = _get_plt()
    sns = _get_sns()
    
    g = sns.pairplot(df[columns + ([hue] if hue else [])].dropna(),
                     hue=hue,
                     diag_kind='kde',
                     plot_kws={'alpha': 0.5},
                     corner=True)
    
    if title:
        g.fig.suptitle(title, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    
    return g


# =============================================================================
# Stratified Comparison Plots
# =============================================================================

def plot_stratified_bars(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    show_error_bars: bool = True,
    order: List[str] = None,
    palette: Dict[str, str] = None,
    save_path: str = None,
    ax=None
):
    """
    Bar plot of means by group with error bars.
    
    Args:
        df: DataFrame with data
        group_col: Column defining groups
        value_col: Column with values
        title, xlabel, ylabel: Labels
        show_error_bars: Show 95% CI error bars
        order: Order of groups on x-axis
        palette: Color palette dict
        save_path: Path to save figure
        ax: Matplotlib axes
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if palette is None:
        palette = get_quartile_palette(df[group_col].unique())
    
    sns.barplot(
        data=df, x=group_col, y=value_col,
        order=order, palette=palette,
        errorbar='ci' if show_error_bars else None,
        ax=ax
    )
    
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    
    # Rotate x labels if needed
    ax.tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


def plot_stratified_violin(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    order: List[str] = None,
    palette: Dict[str, str] = None,
    save_path: str = None,
    ax=None
):
    """
    Violin plot showing full distributions by group.
    
    Args:
        df: DataFrame with data
        group_col: Column defining groups
        value_col: Column with values
        title, xlabel, ylabel: Labels
        order: Order of groups on x-axis
        palette: Color palette dict
        save_path: Path to save figure
        ax: Matplotlib axes
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if palette is None:
        palette = get_quartile_palette(df[group_col].unique())
    
    sns.violinplot(
        data=df, x=group_col, y=value_col,
        order=order, palette=palette,
        ax=ax, inner='box'
    )
    
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    
    ax.tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


def plot_stratified_box(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    order: List[str] = None,
    show_outliers: bool = True,
    save_path: str = None,
    ax=None
):
    """
    Box plot by group.
    
    Args:
        df: DataFrame with data
        group_col: Column defining groups
        value_col: Column with values
        title, xlabel, ylabel: Labels
        order: Order of groups on x-axis
        show_outliers: Show outlier points
        save_path: Path to save figure
        ax: Matplotlib axes
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=df, x=group_col, y=value_col,
        order=order, showfliers=show_outliers,
        ax=ax
    )
    
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    
    ax.tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


# =============================================================================
# Correlation Heatmaps
# =============================================================================

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = None,
    mask_upper: bool = True,
    annotate: bool = True,
    cmap: str = 'RdBu_r',
    vmin: float = -1,
    vmax: float = 1,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot correlation matrix as heatmap.
    
    Args:
        corr_matrix: Correlation DataFrame
        title: Plot title
        mask_upper: Mask upper triangle
        annotate: Show correlation values
        cmap: Colormap
        vmin, vmax: Color scale limits
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    sns = _get_sns()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annotate,
        fmt='.2f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 9}
    )
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


# =============================================================================
# CDF and Pareto Curves
# =============================================================================

def plot_cdf(
    data: pd.Series,
    title: str = None,
    xlabel: str = None,
    log_x: bool = False,
    percentile_markers: List[float] = None,
    save_path: str = None,
    ax=None
):
    """
    Plot cumulative distribution function.
    
    Args:
        data: Series of values
        title: Plot title
        xlabel: X-axis label
        log_x: Use log scale
        percentile_markers: Percentiles to mark on plot
        save_path: Path to save figure
        ax: Matplotlib axes
        
    Returns:
        matplotlib axes
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_data = np.sort(data.dropna())
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax.plot(sorted_data, cdf, color=HYPOTHESIS_PALETTE['primary'], linewidth=2)
    
    if percentile_markers:
        for p in percentile_markers:
            val = np.percentile(sorted_data, p)
            ax.axvline(val, color=HYPOTHESIS_PALETTE['secondary'], linestyle='--', alpha=0.5)
            ax.axhline(p/100, color=HYPOTHESIS_PALETTE['secondary'], linestyle='--', alpha=0.5)
            ax.plot(val, p/100, 'o', color=HYPOTHESIS_PALETTE['tertiary'], markersize=8)
    
    if log_x:
        ax.set_xscale('log')
    
    ax.set_xlabel(xlabel or 'Value')
    ax.set_ylabel('Cumulative Probability')
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    return ax


def plot_pareto(
    df: pd.DataFrame,
    value_col: str,
    cumsum_col: str = None,
    title: str = None,
    xlabel: str = "Rank (sorted by value)",
    ylabel: str = None,
    highlight_percentiles: List[float] = [80, 95],
    save_path: str = None
):
    """
    Plot Pareto chart showing concentration.
    
    Args:
        df: DataFrame with data
        value_col: Column with values to analyze
        cumsum_col: Pre-computed cumulative sum column (optional)
        title: Plot title
        xlabel, ylabel: Labels
        highlight_percentiles: Percentiles to mark (e.g., [80] for 80-20 rule)
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    plt = _get_plt()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by value descending
    sorted_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    
    # Compute cumulative share
    total = sorted_df[value_col].sum()
    sorted_df['cumulative_pct'] = sorted_df[value_col].cumsum() / total * 100
    sorted_df['rank_pct'] = (np.arange(len(sorted_df)) + 1) / len(sorted_df) * 100
    
    # Left plot: Bar chart of individual contributions
    ax1.bar(range(min(100, len(sorted_df))), 
            sorted_df[value_col].head(100),
            color=HYPOTHESIS_PALETTE['primary'])
    ax1.set_xlabel('Rank')
    ax1.set_ylabel(ylabel or value_col)
    ax1.set_title('Top 100 by Value')
    
    # Right plot: Cumulative curve
    ax2.plot(sorted_df['rank_pct'], sorted_df['cumulative_pct'],
             color=HYPOTHESIS_PALETTE['primary'], linewidth=2)
    
    # Add diagonal (perfect equality)
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect equality')
    
    # Mark percentiles
    for p in highlight_percentiles:
        # Find rank at which we reach p% of value
        idx = (sorted_df['cumulative_pct'] >= p).idxmax()
        rank_at_p = sorted_df.loc[idx, 'rank_pct']
        ax2.axhline(p, color=HYPOTHESIS_PALETTE['secondary'], linestyle=':', alpha=0.5)
        ax2.axvline(rank_at_p, color=HYPOTHESIS_PALETTE['secondary'], linestyle=':', alpha=0.5)
        ax2.annotate(f'{p}% from top {rank_at_p:.1f}%',
                    xy=(rank_at_p, p),
                    xytext=(rank_at_p + 5, p - 10),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Cumulative % of Total')
    ax2.set_title('Pareto Curve (Concentration)')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    if title:
        fig.suptitle(title, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# =============================================================================
# Multi-Panel Hypothesis Plots
# =============================================================================

def plot_hypothesis_summary(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bin_col: str,
    title: str = None,
    save_path: str = None
):
    """
    Create multi-panel summary for a hypothesis test.
    
    Includes:
    - Top left: Scatter of x vs y
    - Top right: Stratified bars by bin
    - Bottom left: Distribution of x
    - Bottom right: Distribution of y
    
    Args:
        df: DataFrame with data
        x_col: Predictor column (e.g., dispersion)
        y_col: Outcome column (e.g., miss_rate)
        bin_col: Binned predictor column (e.g., dispersion_bin)
        title: Overall title
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    plt = _get_plt()
    sns = _get_sns()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top left: Scatter with regression
    plot_scatter_with_regression(df, x_col, y_col, ax=axes[0, 0])
    axes[0, 0].set_title(f'{x_col} vs {y_col}')
    
    # Top right: Stratified bars
    plot_stratified_bars(df, bin_col, y_col, ax=axes[0, 1])
    axes[0, 1].set_title(f'{y_col} by {bin_col}')
    
    # Bottom left: X distribution
    plot_distribution(df[x_col], ax=axes[1, 0])
    axes[1, 0].set_title(f'Distribution of {x_col}')
    axes[1, 0].set_xlabel(x_col)
    
    # Bottom right: Y distribution
    plot_distribution(df[y_col], ax=axes[1, 1])
    axes[1, 1].set_title(f'Distribution of {y_col}')
    axes[1, 1].set_xlabel(y_col)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(
    fig,
    name: str,
    output_dir: str,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 150
):
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib figure
        name: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save
        dpi: Resolution for raster formats
    """
    plt = _get_plt()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        path = output_path / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {path}")


def create_figure_grid(
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int] = None,
    sharex: bool = False,
    sharey: bool = False
):
    """
    Create figure with grid of subplots.
    
    Args:
        nrows, ncols: Grid dimensions
        figsize: Figure size (auto-calculated if None)
        sharex, sharey: Share axes
        
    Returns:
        Tuple of (figure, axes array)
    """
    plt = _get_plt()
    
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    return plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
