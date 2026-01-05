"""
Statistical Utilities for Hypothesis Testing

Provides reusable statistical functions for:
- Stratification and binning
- Correlation analysis
- Statistical tests (t-test, Mann-Whitney, chi-square)
- Effect size calculations
- Confidence intervals and bootstrap
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau


# =============================================================================
# Stratification and Binning
# =============================================================================

def create_quartile_bins(
    series: pd.Series,
    labels: Optional[List[str]] = None,
    prefix: str = "Q"
) -> pd.Series:
    """
    Create quartile bins for a numeric series.
    
    Args:
        series: Numeric series to bin
        labels: Optional custom labels
        prefix: Prefix for auto-generated labels
        
    Returns:
        Categorical series with bin assignments
    """
    if labels is None:
        labels = [f"{prefix}1", f"{prefix}2", f"{prefix}3", f"{prefix}4"]
    
    return pd.qcut(
        series.fillna(series.median()),
        q=4,
        labels=labels,
        duplicates='drop'
    )


def create_percentile_bins(
    series: pd.Series,
    percentiles: List[float] = [0, 25, 50, 75, 90, 95, 99, 100],
    labels: Optional[List[str]] = None
) -> pd.Series:
    """
    Create bins based on custom percentile thresholds.
    
    Args:
        series: Numeric series to bin
        percentiles: List of percentile boundaries
        labels: Optional custom labels (length = len(percentiles) - 1)
        
    Returns:
        Categorical series with bin assignments
    """
    bins = np.percentile(series.dropna(), percentiles)
    
    if labels is None:
        labels = [f"p{percentiles[i]}-p{percentiles[i+1]}" for i in range(len(percentiles)-1)]
    
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def stratify_by_column(
    df: pd.DataFrame,
    stratify_col: str,
    target_cols: List[str],
    aggfuncs: Dict[str, Union[str, Callable]] = None
) -> pd.DataFrame:
    """
    Compute statistics stratified by a categorical column.
    
    Args:
        df: DataFrame with data
        stratify_col: Column to stratify by (categorical or binned)
        target_cols: Columns to compute statistics for
        aggfuncs: Aggregation functions per column (default: mean, std, count)
        
    Returns:
        DataFrame with stratified statistics
    """
    if aggfuncs is None:
        aggfuncs = {col: ['mean', 'std', 'median', 'count'] for col in target_cols}
    
    result = df.groupby(stratify_col)[target_cols].agg(aggfuncs)
    
    # Flatten column names
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    
    return result.reset_index()


def compute_group_comparison(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    baseline_group: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare groups with statistical tests.
    
    Args:
        df: DataFrame with data
        group_col: Column defining groups
        value_col: Column with values to compare
        baseline_group: Reference group (uses first group if None)
        
    Returns:
        DataFrame with group statistics and comparison results
    """
    groups = df[group_col].unique()
    
    if baseline_group is None:
        baseline_group = groups[0]
    
    baseline_values = df[df[group_col] == baseline_group][value_col].dropna()
    
    results = []
    for group in groups:
        group_values = df[df[group_col] == group][value_col].dropna()
        
        row = {
            'group': group,
            'n': len(group_values),
            'mean': group_values.mean(),
            'std': group_values.std(),
            'median': group_values.median(),
        }
        
        # Compare to baseline
        if group != baseline_group and len(group_values) > 1 and len(baseline_values) > 1:
            # t-test
            t_stat, t_pval = stats.ttest_ind(group_values, baseline_values)
            row['t_statistic'] = t_stat
            row['t_pvalue'] = t_pval
            
            # Mann-Whitney U (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(group_values, baseline_values, alternative='two-sided')
            row['mannwhitney_u'] = u_stat
            row['mannwhitney_pvalue'] = u_pval
            
            # Effect size (Cohen's d)
            row['cohens_d'] = cohens_d(group_values, baseline_values)
            
            # Relative difference
            row['rel_diff_from_baseline'] = (group_values.mean() - baseline_values.mean()) / baseline_values.mean()
        else:
            row['t_statistic'] = np.nan
            row['t_pvalue'] = np.nan
            row['mannwhitney_u'] = np.nan
            row['mannwhitney_pvalue'] = np.nan
            row['cohens_d'] = np.nan
            row['rel_diff_from_baseline'] = 0.0
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# Correlation Analysis
# =============================================================================

def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'spearman'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix with p-values.
    
    Args:
        df: DataFrame with numeric columns
        columns: Columns to include (default: all numeric)
        method: 'pearson', 'spearman', or 'kendall'
        
    Returns:
        Tuple of (correlation matrix, p-value matrix)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n = len(columns)
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))
    
    corr_func = {
        'pearson': pearsonr,
        'spearman': spearmanr,
        'kendall': kendalltau
    }[method]
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif i < j:
                # Remove NaN pairs
                mask = df[[col1, col2]].notna().all(axis=1)
                if mask.sum() > 2:
                    corr, pval = corr_func(df.loc[mask, col1], df.loc[mask, col2])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    pval_matrix[i, j] = pval
                    pval_matrix[j, i] = pval
    
    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    pval_df = pd.DataFrame(pval_matrix, index=columns, columns=columns)
    
    return corr_df, pval_df


def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    control: List[str]
) -> Tuple[float, float]:
    """
    Compute partial correlation controlling for other variables.
    
    Args:
        df: DataFrame with data
        x, y: Variables to correlate
        control: Variables to control for
        
    Returns:
        Tuple of (partial correlation, p-value)
    """
    # Residualize x and y against controls
    from scipy.linalg import lstsq
    
    mask = df[[x, y] + control].notna().all(axis=1)
    data = df.loc[mask]
    
    if len(data) < len(control) + 3:
        return np.nan, np.nan
    
    # Design matrix for controls
    X_control = data[control].values
    X_control = np.column_stack([np.ones(len(X_control)), X_control])
    
    # Residualize x
    x_vals = data[x].values
    coef_x, _, _, _ = lstsq(X_control, x_vals)
    resid_x = x_vals - X_control @ coef_x
    
    # Residualize y
    y_vals = data[y].values
    coef_y, _, _, _ = lstsq(X_control, y_vals)
    resid_y = y_vals - X_control @ coef_y
    
    # Correlation of residuals
    return pearsonr(resid_x, resid_y)


# =============================================================================
# Statistical Tests
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1, group2: Arrays of values to compare
        
    Returns:
        Cohen's d (positive means group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_effect_sizes(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute pairwise effect sizes between all groups.
    
    Args:
        df: DataFrame with data
        group_col: Column defining groups
        value_col: Column with values to compare
        groups: Optional subset of groups to compare
        
    Returns:
        DataFrame with pairwise Cohen's d values
    """
    if groups is None:
        groups = df[group_col].unique()
    
    results = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            v1 = df[df[group_col] == g1][value_col].dropna()
            v2 = df[df[group_col] == g2][value_col].dropna()
            
            if len(v1) > 1 and len(v2) > 1:
                d = cohens_d(v1.values, v2.values)
                results.append({
                    'group1': g1,
                    'group2': g2,
                    'cohens_d': d,
                    'n1': len(v1),
                    'n2': len(v2),
                    'mean_diff': v1.mean() - v2.mean()
                })
    
    return pd.DataFrame(results)


def chi_square_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str
) -> Dict[str, Any]:
    """
    Chi-square test for independence between categorical variables.
    
    Args:
        df: DataFrame with data
        group_col: Column defining groups
        outcome_col: Column with outcomes (categorical)
        
    Returns:
        Dict with chi2 statistic, p-value, and contingency table
    """
    contingency = pd.crosstab(df[group_col], df[outcome_col])
    chi2, pval, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        'chi2': chi2,
        'pvalue': pval,
        'dof': dof,
        'contingency_table': contingency,
        'expected': expected
    }


def proportion_test(
    successes: int,
    n: int,
    null_proportion: float = 0.5,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Binomial proportion test.
    
    Args:
        successes: Number of successes
        n: Total trials
        null_proportion: Null hypothesis proportion
        alternative: 'two-sided', 'greater', or 'less'
        
    Returns:
        Tuple of (z-statistic, p-value)
    """
    from statsmodels.stats.proportion import proportions_ztest
    return proportions_ztest(successes, n, null_proportion, alternative=alternative)


# =============================================================================
# Confidence Intervals and Bootstrap
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Array of values
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95)
        random_state: Random seed
        
    Returns:
        Tuple of (point estimate, lower CI, upper CI)
    """
    rng = np.random.RandomState(random_state)
    
    n = len(data)
    boot_stats = []
    
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(statistic(boot_sample))
    
    boot_stats = np.array(boot_stats)
    alpha = 1 - ci_level
    
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    point_estimate = statistic(data)
    
    return point_estimate, lower, upper


def bootstrap_comparison(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Bootstrap comparison of two groups.
    
    Args:
        group1, group2: Arrays of values
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        random_state: Random seed
        
    Returns:
        Dict with point estimates, CIs, and p-value for difference
    """
    rng = np.random.RandomState(random_state)
    
    n1, n2 = len(group1), len(group2)
    boot_diffs = []
    
    for _ in range(n_bootstrap):
        boot1 = rng.choice(group1, size=n1, replace=True)
        boot2 = rng.choice(group2, size=n2, replace=True)
        boot_diffs.append(statistic(boot1) - statistic(boot2))
    
    boot_diffs = np.array(boot_diffs)
    alpha = 1 - ci_level
    
    observed_diff = statistic(group1) - statistic(group2)
    
    # p-value: proportion of bootstrap samples with opposite sign
    if observed_diff >= 0:
        pval = 2 * np.mean(boot_diffs <= 0)
    else:
        pval = 2 * np.mean(boot_diffs >= 0)
    
    return {
        'observed_diff': observed_diff,
        'ci_lower': np.percentile(boot_diffs, 100 * alpha / 2),
        'ci_upper': np.percentile(boot_diffs, 100 * (1 - alpha / 2)),
        'pvalue': min(pval, 1.0),
        'group1_stat': statistic(group1),
        'group2_stat': statistic(group2)
    }


# =============================================================================
# Regression Utilities
# =============================================================================

def simple_regression_summary(
    df: pd.DataFrame,
    x: str,
    y: str,
    robust: bool = True
) -> Dict[str, Any]:
    """
    Fit simple linear regression and return summary statistics.
    
    Args:
        df: DataFrame with data
        x: Predictor column
        y: Response column
        robust: Use robust standard errors (HC3)
        
    Returns:
        Dict with regression results
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        warnings.warn("statsmodels not available, using scipy fallback")
        # Fallback to scipy
        mask = df[[x, y]].notna().all(axis=1)
        slope, intercept, r, pval, se = stats.linregress(df.loc[mask, x], df.loc[mask, y])
        return {
            'intercept': intercept,
            'slope': slope,
            'r_squared': r**2,
            'pvalue': pval,
            'std_err': se
        }
    
    mask = df[[x, y]].notna().all(axis=1)
    X = sm.add_constant(df.loc[mask, x])
    y_vals = df.loc[mask, y]
    
    model = sm.OLS(y_vals, X).fit(cov_type='HC3' if robust else 'nonrobust')
    
    return {
        'intercept': model.params['const'],
        'slope': model.params[x],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'pvalue': model.pvalues[x],
        'std_err': model.bse[x],
        'conf_int_lower': model.conf_int().loc[x, 0],
        'conf_int_upper': model.conf_int().loc[x, 1],
        'n_obs': int(model.nobs),
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue
    }


# =============================================================================
# Summary Statistics
# =============================================================================

def describe_extended(
    series: pd.Series,
    percentiles: List[float] = [1, 5, 10, 25, 50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """
    Extended descriptive statistics including tail percentiles.
    
    Args:
        series: Numeric series
        percentiles: Percentiles to compute
        
    Returns:
        Dict with statistics
    """
    s = series.dropna()
    
    result = {
        'n': len(s),
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        'max': s.max(),
        'skew': s.skew(),
        'kurtosis': s.kurtosis(),
        'iqr': s.quantile(0.75) - s.quantile(0.25)
    }
    
    for p in percentiles:
        result[f'p{p}'] = s.quantile(p / 100)
    
    return result


def compute_gini(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for a distribution.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1]. 0 = perfect equality, 1 = max inequality
    """
    if len(values) == 0:
        return 0.0
    n = len(values)
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    return (2 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n


def compute_entropy(counts: np.ndarray, normalize: bool = False) -> float:
    """
    Compute Shannon entropy of a distribution.
    
    Args:
        counts: Array of counts (will be normalized to probabilities)
        normalize: If True, return normalized entropy (0-1 scale)
        
    Returns:
        Entropy in nats. If normalize=True, returns H / log(n).
    """
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    
    if normalize and len(counts) > 1:
        max_entropy = np.log(len(counts))
        return entropy / max_entropy
    return entropy
