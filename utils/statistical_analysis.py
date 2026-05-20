# coding=utf-8
"""
Statistical Analysis Module for Model Comparison

Provides tools for:
- Paired and independent t-tests
- Effect size calculation (Cohen's d)
- Confidence interval computation
- P-value calculation from T-distribution
"""

import numpy as np
import math


def _student_t_cdf(t, df):
    """
    Compute the cumulative distribution function (CDF) of Student's t-distribution.
    This is a pure Python implementation to avoid scipy dependency.

    Parameters:
        t: t-statistic value
        df: degrees of freedom

    Returns:
        Cumulative probability P(T <= t)
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    # For large df, approximate with standard normal
    if df > 100:
        # Approximation of standard normal CDF
        sign = 1 if t >= 0 else -1
        t = abs(t)
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        k = 1.0 / (1.0 + p * t)
        result = 1.0 - (((((a5 * k + a4) * k) + a3) * k + a2) * k + a1) * k * math.exp(-t * t / 2.0)
        return 0.5 + 0.5 * sign * result

    # For small df, use beta function approach
    # CDF(t) = 0.5 + t * Gamma((df+1)/2) * hypergeometric2F1(0.5, (df+1)/2; 3/2; -t^2/df) / (sqrt(pi*df) * Gamma(df/2))

    # Use numerical integration for accuracy
    # The PDF of t-distribution: f(t) = Gamma((df+1)/2) / (sqrt(pi*df) * Gamma(df/2)) * (1 + t^2/df)^(-(df+1)/2)

    from scipy import integrate

    def t_pdf(x):
        log_gamma_df_half = log_gamma(df / 2)
        log_gamma_df_plus1_half = log_gamma((df + 1) / 2)

        log_coeff = log_gamma_df_plus1_half - 0.5 * math.log(math.pi * df) - log_gamma_df_half
        coeff = math.exp(log_coeff)
        return coeff * (1 + x * x / df) ** (-(df + 1) / 2)

    # Integrate from -infinity to t
    if t <= -10:
        return 0.0  # Effectively zero
    elif t >= 10:
        return 1.0  # Effectively one
    else:
        # Use scipy for the integration
        integral, _ = integrate.quad(t_pdf, -100, t, limit=100)
        return integral


def _log_gamma(x):
    """
    Compute the natural logarithm of the gamma function using Lanczos approximation.
    """
    if x < 0:
        raise ValueError("Gamma function undefined for negative arguments")

    if x < 0.5:
        # Use reflection formula: Gamma(z) * Gamma(1-z) = pi / sin(pi*z)
        # log(Gamma(x)) = log(pi) - log(Gamma(1-x)) - log(sin(pi*x))
        return math.log(math.pi) - _log_gamma(1 - x) - math.log(math.sin(math.pi * x))

    # Lanczos approximation coefficients for g=7
    # These provide good accuracy
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]

    g = 7
    x -= 1
    result = c[0]

    for i in range(1, len(c)):
        result += c[i] / (x + i)

    t = x + g + 0.5
    log_2pi = math.log(2 * math.pi)
    log_result = 0.5 * log_2pi + (x + 0.5) * math.log(t) - t + math.log(result)

    return log_result


def log_gamma(x):
    """
    Wrapper for log_gamma with input validation.
    """
    if x <= 0:
        raise ValueError("Gamma function undefined for non-positive arguments")
    return _log_gamma(x)


def _student_t_ppf(p, df):
    """
    Compute the percent point function (inverse CDF) of Student's t-distribution.
    Find t such that P(T <= t) = p.

    Parameters:
        p: probability (0 < p < 1)
        df: degrees of freedom

    Returns:
        t-statistic value
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be in (0, 1)")

    if p < 0.5:
        return -_student_t_ppf(1 - p, df)

    # Binary search for the critical value
    low = 0
    high = 100

    # For very high p, we need a larger upper bound
    while _student_t_cdf(high, df) < p:
        high *= 2
        if high > 1e6:
            raise ValueError("Unable to find critical value")

    for _ in range(100):
        mid = (low + high) / 2
        cdf_mid = _student_t_cdf(mid, df)
        if cdf_mid < p:
            low = mid
        else:
            high = mid
        if high - low < 1e-10:
            break

    return (low + high) / 2


def paired_t_test(classical_accs, quantum_accs, alpha=0.05):
    """
    Paired sample t-test for comparing two models on the same dataset.

    This test is appropriate when:
    - Both models are evaluated on the same data splits
    - The measurements are paired (same seeds, same data splits)
    - We want to test if there's a significant difference

    Parameters:
        classical_accs: List of n accuracy values from classical model
        quantum_accs: List of n accuracy values from quantum model
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
            - t_statistic: T test statistic
            - p_value: Two-tailed p-value
            - degrees_of_freedom: n - 1
            - is_significant: True if p < alpha
            - critical_value: Critical t-value for two-tailed test
            - mean_difference: Mean of (quantum - classical)
            - std_difference: Std of differences
            - alpha: Significance level used
    """
    classical_accs = np.array(classical_accs, dtype=np.float64)
    quantum_accs = np.array(quantum_accs, dtype=np.float64)

    if len(classical_accs) != len(quantum_accs):
        raise ValueError("Classical and quantum accuracy lists must have the same length")

    n = len(classical_accs)
    if n < 2:
        raise ValueError("Need at least 2 samples for t-test")

    degrees_of_freedom = n - 1

    # Calculate differences: d_i = quantum_i - classical_i
    diffs = quantum_accs - classical_accs

    # Calculate mean and std of differences
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)  # Sample std (n-1 denominator)

    # Calculate t-statistic: t = mean(d) / (std(d) / sqrt(n))
    if std_diff == 0:
        # No variance in differences
        t_statistic = 0.0 if mean_diff == 0 else (np.inf if mean_diff > 0 else -np.inf)
    else:
        t_statistic = mean_diff / (std_diff / math.sqrt(n))

    # Calculate two-tailed p-value
    # P(|T| > |t|) = 2 * P(T > |t|)
    abs_t = abs(t_statistic)

    try:
        # Try using scipy if available for more accurate results
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs_t, df=degrees_of_freedom))
    except ImportError:
        # Fall back to our implementation
        cdf_val = _student_t_cdf(abs_t, degrees_of_freedom)
        p_value = 2 * (1 - cdf_val)

    # Calculate critical value for two-tailed test
    # For two-tailed at alpha, we find t_crit such that P(T > t_crit) = alpha/2
    # So P(T <= t_crit) = 1 - alpha/2
    try:
        from scipy import stats
        critical_value = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    except ImportError:
        critical_value = _student_t_ppf(1 - alpha / 2, degrees_of_freedom)

    is_significant = p_value < alpha

    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'degrees_of_freedom': degrees_of_freedom,
        'is_significant': is_significant,
        'critical_value': critical_value,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'alpha': alpha
    }


def independent_t_test(group1, group2, alpha=0.05):
    """
    Independent two-sample t-test (Welch's t-test).

    This test is appropriate when:
    - The two groups are independent (different random seeds or different data splits)
    - Variances may be unequal (Welch's t-test)

    Parameters:
        group1: List of accuracy values from first model
        group2: List of accuracy values from second model
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
            - t_statistic: T test statistic
            - p_value: Two-tailed p-value
            - degrees_of_freedom: Welch-Satterthwaite degrees of freedom
            - is_significant: True if p < alpha
            - critical_value: Critical t-value for two-tailed test
            - mean1, mean2: Group means
            - std1, std2: Group stds
            - n1, n2: Group sizes
            - alpha: Significance level used
    """
    group1 = np.array(group1, dtype=np.float64)
    group2 = np.array(group2, dtype=np.float64)

    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 samples in each group")

    # Calculate means and variances
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Welch's t-statistic
    # t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
    pooled_std = math.sqrt(var1 / n1 + var2 / n2)

    if pooled_std == 0:
        t_statistic = 0.0 if mean1 == mean2 else (np.inf if mean1 > mean2 else -np.inf)
    else:
        t_statistic = (mean1 - mean2) / pooled_std

    # Welch-Satterthwaite degrees of freedom
    # df = (var1/n1 + var2/n2)^2 / (var1^2/(n1^2*(n1-1)) + var2^2/(n2^2*(n2-1)))
    if var1 == 0 and var2 == 0:
        degrees_of_freedom = n1 + n2 - 2
    else:
        numerator = (var1 / n1 + var2 / n2) ** 2
        denominator = var1**2 / (n1**2 * (n1 - 1)) + var2**2 / (n2**2 * (n2 - 1))
        degrees_of_freedom = numerator / denominator if denominator > 0 else (n1 + n2 - 2)

    # Calculate p-value
    abs_t = abs(t_statistic)
    try:
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs_t, df=degrees_of_freedom))
    except ImportError:
        cdf_val = _student_t_cdf(abs_t, degrees_of_freedom)
        p_value = 2 * (1 - cdf_val)

    # Calculate critical value
    try:
        from scipy import stats
        critical_value = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    except ImportError:
        critical_value = _student_t_ppf(1 - alpha / 2, degrees_of_freedom)

    is_significant = p_value < alpha

    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'degrees_of_freedom': degrees_of_freedom,
        'is_significant': is_significant,
        'critical_value': critical_value,
        'mean1': mean1,
        'mean2': mean2,
        'std1': np.sqrt(var1),
        'std2': np.sqrt(var2),
        'n1': n1,
        'n2': n2,
        'alpha': alpha
    }


def compute_effect_size(group1, group2, test_type='pooled'):
    """
    Compute Cohen's d effect size.

    Effect size interpretation (Cohen's guidelines):
    - |d| < 0.2: Negligible/Small effect
    - 0.2 <= |d| < 0.5: Small effect
    - 0.5 <= |d| < 0.8: Medium effect
    - 0.8 <= |d| < 1.2: Large effect
    - |d| >= 1.2: Very large effect

    Parameters:
        group1: First group of values
        group2: Second group of values
        test_type: 'pooled' for pooled std, 'paired' for std of differences

    Returns:
        Dictionary containing:
            - cohen_d: Cohen's d effect size
            - interpretation: String interpretation of effect size
            - mean1, mean2: Group means
    """
    group1 = np.array(group1, dtype=np.float64)
    group2 = np.array(group2, dtype=np.float64)

    mean1, mean2 = np.mean(group1), np.mean(group2)

    if test_type == 'pooled':
        # Pooled standard deviation for independent samples
        # s_pooled = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1.0

        cohen_d = (mean1 - mean2) / pooled_std

    elif test_type == 'paired':
        # For paired test, use std of differences
        diffs = group1 - group2
        std_diffs = np.std(diffs, ddof=1)
        mean_diff = np.mean(diffs)

        cohen_d = mean_diff / std_diffs if std_diffs > 0 else 0.0

    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Interpret effect size
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        interpretation = "Negligible"
    elif abs_d < 0.5:
        interpretation = "Small"
    elif abs_d < 0.8:
        interpretation = "Medium"
    elif abs_d < 1.2:
        interpretation = "Large"
    else:
        interpretation = "Very Large"

    return {
        'cohen_d': cohen_d,
        'interpretation': interpretation,
        'mean1': mean1,
        'mean2': mean2
    }


def compute_confidence_interval(values, alpha=0.05):
    """
    Compute the confidence interval for the mean of a sample.

    Uses t-distribution for small samples.

    Parameters:
        values: List or array of sample values
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dictionary containing:
            - mean: Sample mean
            - std: Sample standard deviation
            - n: Sample size
            - lower_bound: Lower bound of CI
            - upper_bound: Upper bound of CI
            - confidence_level: 1 - alpha (e.g., 0.95 for 95% CI)
    """
    values = np.array(values, dtype=np.float64)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 samples to compute confidence interval")

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    degrees_of_freedom = n - 1

    # t-critical value for two-tailed test
    try:
        from scipy import stats
        t_crit = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    except ImportError:
        t_crit = _student_t_ppf(1 - alpha / 2, degrees_of_freedom)

    # Standard error
    se = std / math.sqrt(n)

    # Margin of error
    margin = t_crit * se

    lower_bound = mean - margin
    upper_bound = mean + margin

    return {
        'mean': mean,
        'std': std,
        'n': n,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': 1 - alpha,
        'alpha': alpha
    }


def perform_statistical_test(classical_accs, quantum_accs, alpha=0.05, test_type='paired'):
    """
    Perform comprehensive statistical test and print formatted results.

    Parameters:
        classical_accs: List of classical model accuracies
        quantum_accs: List of quantum model accuracies
        alpha: Significance level
        test_type: 'paired' or 'independent'

    Returns:
        Dictionary with all test results
    """
    classical_arr = np.array(classical_accs)
    quantum_arr = np.array(quantum_accs)

    # Choose appropriate test
    if test_type == 'paired':
        test_result = paired_t_test(classical_accs, quantum_accs, alpha)
        effect_size_result = compute_effect_size(quantum_arr, classical_arr, test_type='paired')
    else:
        test_result = independent_t_test(classical_accs, quantum_accs, alpha)
        effect_size_result = compute_effect_size(quantum_arr, classical_arr, test_type='pooled')

    # Compute confidence intervals
    classical_ci = compute_confidence_interval(classical_accs, alpha)
    quantum_ci = compute_confidence_interval(quantum_accs, alpha)

    # Combine results
    result = {
        'test_type': test_type,
        'alpha': alpha,
        'classical': {
            'accuracies': classical_accs,
            'mean': classical_ci['mean'],
            'std': classical_ci['std'],
            'min': float(np.min(classical_arr)),
            'max': float(np.max(classical_arr)),
            'median': float(np.median(classical_arr)),
            'ci': classical_ci
        },
        'quantum': {
            'accuracies': quantum_accs,
            'mean': quantum_ci['mean'],
            'std': quantum_ci['std'],
            'min': float(np.min(quantum_arr)),
            'max': float(np.max(quantum_arr)),
            'median': float(np.median(quantum_arr)),
            'ci': quantum_ci
        },
        'test': test_result,
        'effect_size': effect_size_result
    }

    return result


def print_statistical_summary(result):
    """
    Print a formatted summary of statistical test results.

    Parameters:
        result: Result dictionary from perform_statistical_test
    """
    alpha = result['alpha']
    classical = result['classical']
    quantum = result['quantum']
    test = result['test']
    effect_size = result['effect_size']

    print("\n" + "="*80)
    print(f"STATISTICAL TEST SUMMARY ({result['test_type'].upper()} T-TEST)")
    print("="*80)

    # Sample sizes
    print(f"\nSample Sizes: n = {len(classical['accuracies'])}")

    # Classical model results
    print(f"\n{'-'*80}")
    print("CLASSICAL MODEL RESULTS")
    print(f"{'-'*80}")
    print(f"  Mean:     {classical['mean']:.6f} ({classical['mean']*100:.2f}%)")
    print(f"  Std Dev:  {classical['std']:.6f}")
    print(f"  Median:   {classical['median']:.6f} ({classical['median']*100:.2f}%)")
    print(f"  Min:      {classical['min']:.6f} ({classical['min']*100:.2f}%)")
    print(f"  Max:      {classical['max']:.6f} ({classical['max']*100:.2f}%)")
    print(f"  {int((1-alpha)*100)}% CI:   [{classical['ci']['lower_bound']:.6f}, {classical['ci']['upper_bound']:.6f}]")

    # Quantum model results
    print(f"\n{'-'*80}")
    print("QUANTUM MODEL RESULTS")
    print(f"{'-'*80}")
    print(f"  Mean:     {quantum['mean']:.6f} ({quantum['mean']*100:.2f}%)")
    print(f"  Std Dev:  {quantum['std']:.6f}")
    print(f"  Median:   {quantum['median']:.6f} ({quantum['median']*100:.2f}%)")
    print(f"  Min:      {quantum['min']:.6f} ({quantum['min']*100:.2f}%)")
    print(f"  Max:      {quantum['max']:.6f} ({quantum['max']*100:.2f}%)")
    print(f"  {int((1-alpha)*100)}% CI:   [{quantum['ci']['lower_bound']:.6f}, {quantum['ci']['upper_bound']:.6f}]")

    # Difference
    mean_diff = quantum['mean'] - classical['mean']
    print(f"\n{'-'*80}")
    print("DIFFERENCE (Quantum - Classical)")
    print(f"{'-'*80}")
    print(f"  Mean Difference: {mean_diff:+.6f} ({mean_diff*100:+.2f}%)")

    # T-test results
    print(f"\n{'-'*80}")
    print(f"T-TEST RESULTS (paired t-test, alpha={alpha})")
    print(f"{'-'*80}")
    print(f"  T-statistic:           {test['t_statistic']:.6f}")
    print(f"  Degrees of freedom:    {test['degrees_of_freedom']}")
    print(f"  Critical value:         {test['critical_value']:.6f}")
    print(f"  P-value (two-tailed):   {test['p_value']:.6e}")

    if test['is_significant']:
        print(f"\n  *** SIGNIFICANT DIFFERENCE (p < {alpha}) ***")
    else:
        print(f"\n  *** NOT SIGNIFICANT (p = {test['p_value']:.4f} >= {alpha}) ***")

    # Effect size
    print(f"\n{'-'*80}")
    print("EFFECT SIZE (Cohen's d)")
    print(f"{'-'*80}")
    print(f"  Cohen's d:              {effect_size['cohen_d']:.6f}")
    print(f"  Interpretation:         {effect_size['interpretation']}")
    print(f"\n  Effect size guidelines:")
    print(f"    |d| < 0.2:  Negligible")
    print(f"    0.2 <= |d| < 0.5: Small")
    print(f"    0.5 <= |d| < 0.8: Medium")
    print(f"    0.8 <= |d| < 1.2: Large")
    print(f"    |d| >= 1.2:   Very Large")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    # Test the module with sample data
    np.random.seed(42)
    classical = np.array([0.85, 0.87, 0.84, 0.86, 0.83, 0.88, 0.85, 0.87, 0.84, 0.86])
    quantum = np.array([0.87, 0.89, 0.86, 0.88, 0.85, 0.90, 0.87, 0.89, 0.86, 0.88])

    result = perform_statistical_test(classical, quantum, alpha=0.05)
    print_statistical_summary(result)
