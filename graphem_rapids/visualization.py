"""
Visualization utilities for Graphem.
"""

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats


def report_corr(name, radii, centrality, alpha=0.025):
    """
    Calculate and report the correlation between radial distances and a centrality measure.
    
    Parameters:
        name: str
            Name of the centrality measure
        radii: array-like
            Radial distances from origin
        centrality: array-like
            Centrality values
        alpha: float
            Alpha level for confidence interval
    
    Returns:
        tuple: (correlation coefficient, p-value)
    """
    # Calculate Spearman's rank correlation
    rho, p_value = stats.spearmanr(radii, centrality)
    
    # Calculate bootstrap confidence interval
    n = len(radii)
    reps = 1000
    bootstrap_corrs = []
    
    for _ in range(reps):
        indices = np.random.choice(n, n, replace=True)
        r_sample = radii[indices]
        c_sample = centrality[indices]
        rho_boot, _ = stats.spearmanr(r_sample, c_sample)
        bootstrap_corrs.append(rho_boot)
    
    # Calculate confidence interval
    ci_low = np.percentile(bootstrap_corrs, 100 * alpha)
    ci_high = np.percentile(bootstrap_corrs, 100 * (1 - alpha))
    
    print(f"{name:15s}: rho = {rho:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}]), p = {p_value:.6f}")
    
    return rho, p_value


def report_full_correlation_matrix(radii, deg, btw, eig, pr, clo, nload, alpha=0.025):
    """
    Calculate and report correlations between radial distances and various centrality measures.
    
    Parameters:
        radii: array-like
            Radial distances from origin
        deg, btw, eig, pr, clo, edge_btw: array-like
            Various centrality measures
        alpha: float
            Alpha level for confidence interval
    
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    # Create a DataFrame with all measures
    df = pd.DataFrame({
        'Radius': radii,
        'Degree': deg,
        'Betweenness': btw,
        'Eigenvector': eig,
        'PageRank': pr,
        'Closeness': clo,
        'Node Load': nload
    })
    
    # Calculate correlation matrix
    corr_matrix = df.corr(method='spearman')
    
    # Print correlations with radii
    print("Correlations with radial distance:")
    report_corr("Degree", radii, deg, alpha)
    report_corr("Betweenness", radii, btw, alpha)
    report_corr("Eigenvector", radii, eig, alpha)
    report_corr("PageRank", radii, pr, alpha)
    report_corr("Closeness", radii, clo, alpha)
    report_corr("Node Load", radii, nload, alpha)
    
    return corr_matrix


def plot_radial_vs_centrality(radii, centralities, names):
    """
    Plot scatter plots of radial distances vs. various centrality measures.
    
    Parameters:
        radii: array-like
            Radial distances from origin
        centralities: list of array-like
            List of centrality measures
        names: list of str
            Names of the centrality measures
    """
    # Create figure with subplots
    fig = px.scatter(
        pd.DataFrame({
            'Radial Distance': np.tile(radii, len(names)),
            'Centrality Value': np.concatenate(centralities),
            'Centrality Measure': np.repeat(names, len(radii))
        }),
        x='Radial Distance',
        y='Centrality Value',
        facet_col='Centrality Measure',
        facet_col_wrap=3,
        trendline='ols',
        title='Correlation between Radial Distance and Centrality Measures'
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000
    )
    
    # Show figure
    fig.show()


def display_benchmark_results(benchmark_results):
    """
    Display benchmark results in a nicely formatted table.
    
    Parameters:
        benchmark_results: list of dict
            List of benchmark result dictionaries
    """
    # Convert to DataFrame
    df = pd.DataFrame(benchmark_results)
    
    # Reorder columns for better readability
    columns = [
        'graph_type', 'n', 'm', 'dim', 'seed_method',
        'influence', 'normalized_influence', 'time',
        'layout_time', 'selection_time', 'evaluation_time'
    ]
    df = df[[col for col in columns if col in df.columns]]
    
    # Display the DataFrame
    return df
