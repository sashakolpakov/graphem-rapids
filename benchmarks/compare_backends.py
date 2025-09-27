#!/usr/bin/env python
"""
Backend performance comparison script for GraphEm Rapids.

This script compares the performance of different computational backends
(PyTorch CPU, PyTorch CUDA, and cuVS) across various graph types and sizes.
"""

import time
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from graphem_rapids.generators import (
    erdos_renyi_graph,
    generate_random_regular,
    generate_scale_free,
    generate_ba,
    generate_ws
)
from graphem_rapids.benchmark import run_benchmark
from graphem_rapids import get_backend_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_backends():
    """Get list of available backends based on system capabilities."""
    info = get_backend_info()
    backends = []

    # PyTorch CPU is always available
    backends.append(('pytorch_cpu', 'PyTorch CPU'))

    # PyTorch CUDA if available
    if info['cuda_available']:
        backends.append(('pytorch_cuda', 'PyTorch CUDA'))

    # cuVS if available
    if info['cuvs_available']:
        backends.append(('cuvs', 'RAPIDS cuVS'))

    return backends


def run_backend_comparison(graph_configs, backends, num_iterations=30, output_dir="backend_comparison_results"):
    """
    Run performance comparison across different backends.

    Parameters
    ----------
    graph_configs : list
        List of (generator, params, name) tuples for graphs to test
    backends : list
        List of (backend_name, display_name) tuples for backends to test
    num_iterations : int
        Number of layout iterations for each test
    output_dir : str
        Directory to save results

    Returns
    -------
    pd.DataFrame
        Results dataframe with timing and performance metrics
    """
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    total_tests = len(graph_configs) * len(backends)
    test_count = 0

    for generator, params, graph_name in graph_configs:
        logger.info("\nTesting graph: %s", graph_name)

        for backend_name, backend_display in backends:
            test_count += 1
            logger.info("  Backend: %s (%d/%d)", backend_display, test_count, total_tests)

            try:
                # Configure backend-specific parameters
                backend_kwargs = {}
                if backend_name == 'pytorch_cpu':
                    backend_kwargs['device'] = 'cpu'
                elif backend_name == 'pytorch_cuda':
                    backend_kwargs['device'] = 'cuda'
                # cuVS uses its own backend handling

                # Run benchmark
                start_time = time.time()
                result = run_benchmark(
                    generator,
                    params,
                    dim=3,
                    L_min=10.0,
                    k_attr=0.5,
                    k_inter=0.1,
                    knn_k=15,
                    sample_size=512,
                    batch_size=1024,
                    num_iterations=num_iterations,
                    backend=backend_name,
                    **backend_kwargs
                )
                total_time = time.time() - start_time

                # Record results
                result_entry = {
                    'graph_name': graph_name,
                    'backend': backend_display,
                    'backend_code': backend_name,
                    'n_vertices': result['n'],
                    'n_edges': result['m'],
                    'density': result['density'],
                    'avg_degree': result['avg_degree'],
                    'layout_time': result['layout_time'],
                    'total_time': total_time,
                    'throughput_vertices_per_sec': result['n'] / result['layout_time'],
                    'throughput_edges_per_sec': result['m'] / result['layout_time'],
                    'success': True,
                    'error': None
                }

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("    Error: %s", e)
                result_entry = {
                    'graph_name': graph_name,
                    'backend': backend_display,
                    'backend_code': backend_name,
                    'n_vertices': params.get('n', 0),
                    'n_edges': 0,
                    'density': 0,
                    'avg_degree': 0,
                    'layout_time': np.inf,
                    'total_time': np.inf,
                    'throughput_vertices_per_sec': 0,
                    'throughput_edges_per_sec': 0,
                    'success': False,
                    'error': str(e)
                }

            results.append(result_entry)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(output_path / "backend_comparison.csv", index=False)
    logger.info("Results saved to %s", output_path / 'backend_comparison.csv')

    return df


def create_comparison_plots(df, output_dir="backend_comparison_results"):
    """Create visualization plots for backend comparison results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Filter successful runs only
    df_success = df[df['success']].copy()

    if df_success.empty:
        logger.warning("No successful runs to plot")
        return

    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Layout time comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_success, x='graph_name', y='layout_time', hue='backend')
    plt.yscale('log')
    plt.title('Layout Time Comparison Across Backends')
    plt.xlabel('Graph Type')
    plt.ylabel('Layout Time (seconds, log scale)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / "layout_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Throughput comparison (vertices per second)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_success, x='graph_name', y='throughput_vertices_per_sec', hue='backend')
    plt.yscale('log')
    plt.title('Vertex Processing Throughput Comparison')
    plt.xlabel('Graph Type')
    plt.ylabel('Vertices per Second (log scale)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / "vertex_throughput_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Scalability analysis (if we have multiple graph sizes)
    if len(df_success['n_vertices'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        for backend in df_success['backend'].unique():
            backend_data = df_success[df_success['backend'] == backend]
            plt.loglog(backend_data['n_vertices'], backend_data['layout_time'],
                      'o-', label=backend, alpha=0.7, markersize=6)

        plt.xlabel('Number of Vertices (log scale)')
        plt.ylabel('Layout Time (seconds, log scale)')
        plt.title('Scalability Comparison: Layout Time vs Graph Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "scalability_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Performance speedup table
    pivot_table = df_success.pivot_table(
        index='graph_name',
        columns='backend',
        values='layout_time',
        aggfunc='mean'
    )

    # Calculate speedup relative to CPU (if available)
    if 'PyTorch CPU' in pivot_table.columns:
        speedup_table = pivot_table.div(pivot_table['PyTorch CPU'], axis=0)
        speedup_table.to_csv(output_path / "speedup_table.csv")

        # Plot speedup heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(speedup_table, annot=True, fmt='.2f', cmap='RdYlBu_r', center=1,
                   cbar_kws={'label': 'Speedup Factor (lower is faster)'})
        plt.title('Backend Performance Speedup (relative to PyTorch CPU)')
        plt.xlabel('Backend')
        plt.ylabel('Graph Type')
        plt.tight_layout()
        plt.savefig(output_path / "speedup_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    logger.info("Plots saved to %s/", output_path)


def print_summary_report(df):
    """Print a summary report of backend comparison results."""
    df_success = df[df['success']].copy()

    print("\n" + "="*80)
    print("BACKEND COMPARISON SUMMARY REPORT")
    print("="*80)

    print(f"\nTotal tests run: {len(df)}")
    print(f"Successful tests: {len(df_success)} ({len(df_success)/len(df)*100:.1f}%)")
    print(f"Failed tests: {len(df) - len(df_success)}")

    if not df_success.empty:
        print("\nAverage Layout Times by Backend:")
        avg_times = df_success.groupby('backend')['layout_time'].agg(['mean', 'std', 'min', 'max'])
        print(avg_times.round(3))

        print("\nThroughput (Vertices/sec) by Backend:")
        throughput = df_success.groupby('backend')['throughput_vertices_per_sec'].agg(['mean', 'std'])
        print(throughput.round(1))

        # Find best performing backend for each graph
        print("\nBest Backend for Each Graph Type:")
        best_backends = df_success.loc[df_success.groupby('graph_name')['layout_time'].idxmin()]
        for _, row in best_backends.iterrows():
            print(f"  {row['graph_name']}: {row['backend']} ({row['layout_time']:.3f}s)")

    # Report failed tests
    df_failed = df[~df['success']]
    if not df_failed.empty:
        print(f"\nFailed Tests ({len(df_failed)}):")
        for _, row in df_failed.iterrows():
            print(f"  {row['graph_name']} + {row['backend']}: {row['error']}")

    print("\n" + "="*80)


def main():
    """Main function for backend comparison."""
    parser = argparse.ArgumentParser(description="Compare GraphEm Rapids backend performance")
    parser.add_argument("--num-iterations", "-n", type=int, default=30,
                       help="Number of layout iterations per test")
    parser.add_argument("--output-dir", "-o", default="backend_comparison_results",
                       help="Output directory for results and plots")
    parser.add_argument("--small-test", action="store_true",
                       help="Run smaller test graphs for quick comparison")
    args = parser.parse_args()

    # Get available backends
    backends = get_available_backends()
    logger.info("Available backends: %s", [b[1] for b in backends])

    # Define test graphs
    if args.small_test:
        graph_configs = [
            (erdos_renyi_graph, {'n': 100, 'p': 0.05, 'seed': 42}, 'Erdős-Rényi (n=100)'),
            (generate_random_regular, {'n': 100, 'd': 4, 'seed': 42}, 'Random Regular (n=100)'),
            (generate_ba, {'n': 100, 'm': 2, 'seed': 42}, 'Barabási-Albert (n=100)')
        ]
    else:
        graph_configs = [
            (erdos_renyi_graph, {'n': 200, 'p': 0.02, 'seed': 42}, 'Erdős-Rényi (n=200)'),
            (erdos_renyi_graph, {'n': 500, 'p': 0.01, 'seed': 42}, 'Erdős-Rényi (n=500)'),
            (generate_random_regular, {'n': 200, 'd': 4, 'seed': 42}, 'Random Regular (n=200)'),
            (generate_random_regular, {'n': 500, 'd': 4, 'seed': 42}, 'Random Regular (n=500)'),
            (generate_scale_free, {'n': 200, 'seed': 42}, 'Scale-Free (n=200)'),
            (generate_ba, {'n': 200, 'm': 2, 'seed': 42}, 'Barabási-Albert (n=200)'),
            (generate_ws, {'n': 200, 'k': 6, 'p': 0.1, 'seed': 42}, 'Watts-Strogatz (n=200)')
        ]

    # Run comparison
    logger.info("Starting backend comparison...")
    df_results = run_backend_comparison(graph_configs, backends, args.num_iterations, args.output_dir)

    # Generate plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(df_results, args.output_dir)

    # Print summary
    print_summary_report(df_results)

    logger.info("Backend comparison complete! Results saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()