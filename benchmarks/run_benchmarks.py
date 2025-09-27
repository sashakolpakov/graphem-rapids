#!/usr/bin/env python
"""
Comprehensive benchmark runner for the Graphem library.

This script runs all available tests and benchmarks in the library and
generates result tables in multiple formats (Markdown, LaTeX).
"""

import os
from copy import copy
from pathlib import Path
import cProfile
import pstats
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import stats
from line_profiler import LineProfiler

# PyTorch backend for GraphEm Rapids
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import GraphEm Rapids modules
from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.generators import (
    erdos_renyi_graph,
    generate_random_regular,
    generate_scale_free,
    generate_geometric,
    generate_caveman,
    generate_ws,
    generate_ba,
    generate_sbm
)
from graphem_rapids.benchmark import run_benchmark

from graphem_rapids.datasets import load_dataset

from graphem_rapids.influence import (
    graphem_seed_selection,
    ndlib_estimated_influence,
    greedy_seed_selection
)


class BenchmarkRunner:
    """Main class for running benchmarks and generating reports."""
    
    def __init__(self, output_dir="results", formats=None, subsample_size=None):
        """
        Initialize the benchmark runner.
        
        Parameters:
            output_dir: str
                Directory to save results
            formats: list
                Output formats (markdown, latex, html)
            subsample_size: int
                Subsample vertices for large graphs (default: no subsampling)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if formats is None:
            self.formats = ["markdown", "latex"]
        else:
            self.formats = formats
            
        self.subsample_size = subsample_size
        
        # Set up timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # List to collect all results
        self.results = {}
        
    def run_all_benchmarks(self):
        """Run all available benchmarks."""
        print(f"\n{'='*75}")
        print("Running all Graphem benchmarks")
        print(f"{'='*75}")
        print(f"Results will be saved to: {self.run_dir}")
        
        # Record start time
        start_time = time.time()
        
        # Run each benchmark type
        print("\nRunning benchmark suite:")
        benchmarks = [
            ("Graph Generator Benchmarks", self.run_generator_benchmarks),
            ("Real-world Dataset Benchmarks", self.run_dataset_benchmarks),
            ("Influence Maximization Benchmarks", self.run_influence_benchmarks)
        ]
        
        # Create progress bar for overall benchmarks
        with tqdm(benchmarks, desc="Overall Progress", unit="benchmark", ncols=100) as pbar:
            for benchmark_name, benchmark_func in pbar:
                pbar.set_description(f"Running {benchmark_name}")
                benchmark_func()
        
        # Generate summary report
        total_time = time.time() - start_time
        print(f"\nAll benchmarks completed in {total_time:.2f} seconds")
        self.generate_summary_report(total_time)
        
    def run_generator_benchmarks(self):
        """Run benchmarks on various graph generators."""
        print(f"\n{'-'*75}")
        print("Running graph generator benchmarks")
        print(f"{'-'*75}")
        
        # Define graph configurations to test
        graph_configs = [
            (erdos_renyi_graph, {'n': 500, 'p': 0.03, 'seed': 42}, 'Erdős–Rényi'),
            (generate_random_regular, {'n': 500, 'd': 3, 'seed': 42}, 'Random Regular (d=3)'),
            (generate_random_regular, {'n': 500, 'd': 5, 'seed': 42}, 'Random Regular (d=5)'),
            (generate_ws, {'n': 500, 'k': 4, 'p': 0.1, 'seed': 42}, 'Watts-Strogatz Small-World'),
            (generate_ba, {'n': 500, 'm': 2, 'seed': 42}, 'Barabási-Albert'),
            (generate_sbm, {'n_per_block': 125, 'num_blocks': 4, 'p_in': 0.3, 'p_out': 0.01, 'seed': 42}, 'Stochastic Block Model'),
            (generate_scale_free, {'n': 500, 'seed': 42}, 'Scale-Free'),
            (generate_geometric, {'n': 500, 'radius': 0.15, 'seed': 42}, 'Random Geometric'),
            (generate_caveman, {'l': 20, 'k': 25}, 'Caveman')
        ]
        
        # Run benchmarks
        results = []
        
        # Use tqdm to create a progress bar
        for generator, params, name in tqdm(graph_configs, desc="Generator Benchmarks", unit="graph"):
            tqdm.write(f"\nBenchmarking {name}...")
            
            # Run benchmark
            result = run_benchmark(
                generator, 
                params,
                dim=3,
                num_iterations=30
            )
            
            # Add graph type name
            result['graph_type'] = name
            
            # Add to results list
            results.append({
                'graph_type': name,
                'vertices': result['n'],
                'edges': result['m'],
                'density': result['density'],
                'avg_degree': result['avg_degree'],
                'layout_time': result['layout_time'],
                'total_time': result['total_time']
            })
            
            # For correlation analysis, compute and save key correlations
            corr_results = {}
            for measure in ['degree', 'betweenness', 'eigenvector', 'pagerank', 'closeness', 'node_load']:
                if measure in result and 'radii' in result:

                    # Check if the measure is constant (which happens with degree for regular graphs)
                    values = result[measure]
                    if np.all(values == values[0]):
                        # All values are the same - correlation is undefined
                        corr_results[f'{measure}_corr'] = np.nan
                        corr_results[f'{measure}_p'] = np.nan
                        tqdm.write(f"\n{measure} centrality is constant - correlation is not applicable")
                    else:
                        # Calculate correlation normally
                        corr, p_val = stats.spearmanr(result['radii'], values)
                        corr_results[f'{measure}_corr'] = corr
                        corr_results[f'{measure}_p'] = p_val
            
            # Update the result entry with correlations
            results[-1].update(corr_results)
        
        # Create a DataFrame
        df = pd.DataFrame(results)
        
        # Save the results
        self.save_results('generator_benchmarks', df)
        self.results['generator_benchmarks'] = df
    
    def run_dataset_benchmarks(self):
        """Run benchmarks on real-world datasets."""
        # Create output directory
        benchmark_dir = self.run_dir / "dataset_benchmarks"
        benchmark_dir.mkdir(exist_ok=True)
        
        # List of datasets to test
        datasets = [
            'snap-facebook_combined',
            'snap-ca-GrQc',
            'snap-ca-HepTh'
        ]
        
        # Use the configurable subsample_size if provided, otherwise no subsampling
        sample_size = self.subsample_size  # Will be None if not specified
        results = []
        
        # Use tqdm for progress tracking
        for dataset_name in tqdm(datasets, desc="Dataset Benchmarks", unit="dataset"):
            tqdm.write(f"\nBenchmarking {dataset_name}...")
            
            try:
                # Load the dataset
                vertices, edges = load_dataset(dataset_name)
                
                # Record original size
                original_size = len(vertices)
                original_edges = len(edges)

                n_vertices = copy(original_size)
                
                # Sample if needed and if subsample_size is provided
                if sample_size is not None and n_vertices > sample_size:
                    tqdm.write(f"\nSampling {sample_size} vertices from {n_vertices}...")
                    sampled_vertices = np.random.choice(vertices, sample_size, replace=False)
                    
                    # Filter edges that contain sampled vertices
                    sampled_edges = []
                    for u, v in edges:
                        if u in sampled_vertices and v in sampled_vertices:
                            sampled_edges.append((u, v))
                    
                    vertices = sampled_vertices
                    edges = np.array(sampled_edges)
                    n_vertices = sample_size
                    tqdm.write(f"\nAfter sampling: {n_vertices} vertices, {len(edges)} edges")
                else:
                    if sample_size is None:
                        tqdm.write(f"\nUsing full dataset: {n_vertices} vertices, {len(edges)} edges")

                # Create NetworkX graph for analysis
                G = nx.Graph()
                G.add_nodes_from(vertices)
                G.add_edges_from(edges)
                G = nx.convert_node_labels_to_integers(G,
                                                       first_label=0,
                                                       ordering='default',
                                                       label_attribute=None)
                
                # Basic graph properties
                density = 2 * len(edges) / (n_vertices * (n_vertices - 1))
                avg_degree = 2 * len(edges) / n_vertices
                
                # Get largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                lcc_size = len(largest_cc)
                lcc_fraction = lcc_size / n_vertices

                # Analyze largest connected component
                G_cc = G
                if len(largest_cc) < n_vertices:
                    print(f"Extracting largest connected component with {len(largest_cc):,} vertices...")
                    G_cc = G.subgraph(largest_cc).copy()

                    # Re-index nodes to be consecutive integers
                    G_cc = nx.convert_node_labels_to_integers(G_cc)

                    # Extract edges from the largest component
                    edges = G_cc.edges
                    n_vertices = len(largest_cc)
                
                # Run the embedding
                tqdm.write(f"\nRunning layout for {dataset_name}...")
                start_time = time.time()
                # Create and run embedder
                embedder = GraphEmbedderPyTorch(
                    edges=edges,
                    n_vertices=n_vertices,
                    dimension=3,
                    L_min=4.0,
                    k_attr=0.5,
                    k_inter=0.1,
                    knn_k=min(15, n_vertices // 10),
                    sample_size=min(512, len(edges)),
                    batch_size=min(1024, len(edges)),
                    verbose=False
                )
                
                # Add progress tracking for layout iterations
                iterations = 30
                for i in tqdm(range(iterations), desc="Layout Iterations", leave=False):
                    embedder.update_positions()
                    if (i + 1) % 10 == 0:
                        tqdm.write(f"\nCompleted {i+1}/{iterations} iterations")
                
                layout_time = time.time() - start_time
                
                # Calculate centrality measures
                tqdm.write("\nCalculating centrality measures...")
                
                # Run with progress
                with tqdm(total=2, desc="Centrality Measures", leave=False) as pbar:
                    positions = np.array(embedder.positions)
                    radii = np.linalg.norm(positions, axis=1)
                    degree = np.array([d for _, d in G_cc.degree()])
                    pbar.update(1)
                    
                    # Try to calculate other centrality measures if graph is small enough

                    btw_corr, eig_corr, pr_corr = np.nan, np.nan, np.nan

                    if n_vertices < 5000:
                        try:
                            btw = np.array(list(nx.betweenness_centrality(G_cc).values()))
                            btw_corr, _ = stats.spearmanr(radii, btw)
                        except nx.NetworkXError as e:
                            btw_corr = np.nan
                            print(e)
                            
                        try:
                            eig = np.array(list(nx.eigenvector_centrality_numpy(G_cc).values()))
                            eig_corr, _ = stats.spearmanr(radii, eig)
                        except nx.NetworkXError as e:
                            eig_corr = np.nan
                            print(e)
                            
                        try:
                            pr = np.array(list(nx.pagerank(G_cc).values()))
                            pr_corr, _ = stats.spearmanr(radii, pr)
                        except nx.NetworkXError as e:
                            pr_corr = np.nan
                            print(e)

                    pbar.update(1)
                
                # Store results
                results.append({
                    'dataset': dataset_name,
                    'original_vertices': original_size,
                    'original_edges': original_edges,
                    'sampled_vertices': n_vertices,
                    'sampled_edges': len(edges),
                    'density': density,
                    'avg_degree': avg_degree,
                    'lcc_size': lcc_size,
                    'lcc_fraction': lcc_fraction,
                    'layout_time': layout_time,
                    'degree_correlation': stats.spearmanr(radii, degree)[0],
                    'betweenness_correlation': btw_corr,
                    'eigenvector_correlation': eig_corr,
                    'pagerank_correlation': pr_corr
                })
                
            except Exception as e:
                tqdm.write(f"\nError processing {dataset_name}: {str(e)}")
                # Add dummy entry with error
                results.append({
                    'dataset': dataset_name,
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        self.save_results('dataset_benchmarks', df)
        self.results['dataset_benchmarks'] = df
    
    def run_influence_benchmarks(self):
        """Run influence maximization benchmarks."""
        print(f"\n{'-'*75}")
        print("Running influence maximization benchmarks")
        print(f"{'-'*75}")
        
        # Define graph configurations to test
        graph_configs = [
            (erdos_renyi_graph, {'n': 200, 'p': 0.05, 'seed': 42}, 'Erdős–Rényi'),
            (generate_random_regular, {'n': 200, 'd': 4, 'seed': 42}, 'Random Regular'),
            (generate_ws, {'n': 200, 'k': 4, 'p': 0.1, 'seed': 42}, 'Watts-Strogatz'),
            (generate_ba, {'n': 200, 'm': 3, 'seed': 42}, 'Barabási-Albert'),
            (generate_sbm, {'n_per_block': 50, 'num_blocks': 4, 'p_in': 0.2, 'p_out': 0.01, 'seed': 42}, 'SBM')
        ]
        
        results = []
        
        # Use tqdm for progress tracking
        for generator, params, name in tqdm(graph_configs, desc="Influence Maximization", unit="graph"):
            tqdm.write(f"\nTesting influence maximization on {name}...")
            
            try:
                # Generate graph
                edges = generator(**params)
                
                # Determine number of vertices
                if len(edges) > 0:
                    n = int(max(np.max(edges) + 1, params.get('n', 0)))
                else:
                    n = params.get('n', 0)
                
                # Create NetworkX graph
                G = nx.Graph()
                G.add_nodes_from(range(n))
                G.add_edges_from(edges)
                
                # Parameters for influence maximization
                k = 10  # Number of seed nodes
                p = 0.1  # Propagation probability
                
                # Graphem-based seed selection
                tqdm.write("\nRunning GraphEm seed selection...")
                embedder = GraphEmbedderPyTorch(
                    edges=edges,
                    n_vertices=n,
                    dimension=3,
                    L_min=10.0,
                    k_attr=0.5,
                    k_inter=0.1,
                    knn_k=15,
                    sample_size=min(512, len(edges)),
                    batch_size=min(1024, n),
                    verbose=False
                )
                
                # Run layout with progress bar
                iterations = 20
                for _ in tqdm(range(iterations), desc="Layout for GraphEm", leave=False):
                    embedder.update_positions()
                
                graphem_start = time.time()
                graphem_seeds = graphem_seed_selection(embedder, k, num_iterations=0)  # Already ran layout above
                graphem_time = time.time() - graphem_start
                
                # Greedy seed selection (just a small subset for benchmark)
                tqdm.write("\nRunning Greedy seed selection...")
                greedy_start = time.time()
                with tqdm(total=k, desc="Greedy Selection", leave=False) as pbar:
                    # We can't modify the greedy_seed_selection function directly, but we'll update afterward
                    greedy_seeds, _ = greedy_seed_selection(G, k, p, iterations_count=50)
                    # This just moves the progress bar to completion
                    pbar.update(k)
                greedy_time = time.time() - greedy_start
                
                # Evaluate influence
                tqdm.write("\nEvaluating influence...")
                
                # Run with progress
                with tqdm(total=2, desc="Influence Evaluation", leave=False) as pbar:
                    graphem_influence, _ = ndlib_estimated_influence(G, graphem_seeds, p, iterations_count=100)
                    pbar.update(1)
                    greedy_influence, _ = ndlib_estimated_influence(G, greedy_seeds, p, iterations_count=100)
                    pbar.update(1)
                
                # Compute random baseline
                random_influences = []
                with tqdm(total=5, desc="Random Baseline", leave=False) as pbar:
                    for _ in range(5):
                        random_seeds = np.random.choice(n, k, replace=False)
                        random_influence, _ = ndlib_estimated_influence(G, random_seeds, p, iterations_count=100)
                        random_influences.append(random_influence)
                        pbar.update(1)
                random_influence = np.mean(random_influences)
                
                # Store results
                results.append({
                    'graph_type': name,
                    'vertices': n,
                    'edges': len(edges),
                    'avg_degree': 2 * len(edges) / n,
                    'graphem_influence': graphem_influence,
                    'greedy_influence': greedy_influence,
                    'random_influence': random_influence,
                    'graphem_time': graphem_time,
                    'greedy_time': greedy_time,
                    'graphem_norm_influence': graphem_influence / n,
                    'greedy_norm_influence': greedy_influence / n,
                    'random_norm_influence': random_influence / n,
                    'graphem_efficiency': graphem_influence / (graphem_time or 1),
                    'greedy_efficiency': greedy_influence / (greedy_time or 1)
                })
                
            except Exception as e:
                tqdm.write(f"\nError in influence maximization for {name}: {str(e)}")
                results.append({
                    'graph_type': name,
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        self.save_results('influence_benchmarks', df)
        self.results['influence_benchmarks'] = df
    
    def save_results(self, name, df):
        """
        Save results to files in various formats.
        
        Parameters:
            name: str
                Name of the result set
            df: pandas.DataFrame
                The DataFrame with results
        """
        # Create a directory for this result set
        result_dir = self.run_dir / name
        result_dir.mkdir(exist_ok=True)
        
        # Save raw data as CSV
        df.to_csv(result_dir / f"{name}.csv", index=False)
        
        # Save in each requested format with progress indication
        print(f"\nGenerating {len(self.formats)} result formats for {name}...")
        for fmt in tqdm(self.formats, desc="Formats", unit="format"):
            if fmt == "markdown":
                self._save_markdown(df, result_dir / f"{name}.md", name)
            elif fmt == "latex":
                self._save_latex(df, result_dir / f"{name}.tex", name)
            elif fmt == "html":
                self._save_html(df, result_dir / f"{name}.html", name)
    
    def _save_markdown(self, df, filepath, title):
        """Save DataFrame as Markdown table."""
        # Create a copy to avoid modifying the original
        df_display = df.copy()
        
        # Format numeric columns
        for col in df_display.select_dtypes(include=['float']).columns:
            if "time" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}s" if not pd.isna(x) else "N/A")
            elif "correlation" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
            elif "edges" in col or "vertices" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "N/A")
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        # Create Markdown content
        content = [f"# {title.replace('_', ' ').title()}", ""]
        
        # Add dataset descriptions if this is the dataset benchmarks
        if "dataset" in title:
            content.append("## Dataset Information\n")
            content.append("The datasets used in this benchmark are from the following sources:\n")
            content.append("- **SNAP**: Stanford Network Analysis Project datasets (prefixed with 'snap-')")
            content.append("  - **facebook_combined**: Facebook social network")
            content.append("  - **ca-GrQc**: Collaboration network of Arxiv General Relativity")
            content.append("  - **ca-HepTh**: Collaboration network of Arxiv High Energy Physics Theory")
            content.append("- **Network Repository**: Various network datasets (prefixed with 'netrepo-')")
            content.append("- **Semantic Scholar**: Academic citation networks (prefixed with 'semanticscholar-')\n")
            
            content.append("## Column Descriptions\n")
            content.append("- **dataset**: Name of the dataset")
            content.append("- **original_vertices**: Number of vertices in the original dataset")
            content.append("- **original_edges**: Number of edges in the original dataset")
            content.append("- **sampled_vertices**: Number of vertices after sampling (if applied)")
            content.append("- **sampled_edges**: Number of edges after sampling (if applied)")
            content.append("- **density**: Edge density of the graph (2|E|/(|V|(|V|-1)))")
            content.append("- **avg_degree**: Average degree of vertices (2|E|/|V|)")
            content.append("- **lcc_size**: Size of the largest connected component")
            content.append("- **lcc_fraction**: Fraction of graph in the largest connected component")
            content.append("- **layout_time**: Time (in seconds) to compute the graph layout")
            content.append("- **degree_correlation**: Spearman correlation between radial distance and degree centrality")
            content.append("- **betweenness_correlation**: Spearman correlation between radial distance and betweenness centrality")
            content.append("- **eigenvector_correlation**: Spearman correlation between radial distance and eigenvector centrality")
            content.append("- **pagerank_correlation**: Spearman correlation between radial distance and PageRank\n")
        
        # Add generator descriptions if this is the generator benchmarks
        elif "generator" in title:
            content.append("## Graph Generator Information\n")
            content.append("The graph generators used in this benchmark:\n")
            content.append("- **Erdős–Rényi**: Random graph where each edge has equal probability")
            content.append("- **Random Regular**: Graph where all vertices have the same degree")
            content.append("- **Watts-Strogatz**: Small-world network with high clustering and short path lengths")
            content.append("- **Barabási-Albert**: Scale-free network generated by preferential attachment")
            content.append("- **Stochastic Block Model (SBM)**: Community-structured graph with dense within-community connections")
            content.append("- **Scale-Free**: Graph with power-law degree distribution")
            content.append("- **Random Geometric**: Graph where vertices are connected if within a distance threshold")
            content.append("- **Caveman**: Graph consisting of connected dense clusters\n")
            
            content.append("## Column Descriptions\n")
            content.append("- **graph_type**: Type of graph generator used")
            content.append("- **vertices**: Number of vertices in the graph")
            content.append("- **edges**: Number of edges in the graph")
            content.append("- **density**: Edge density of the graph (2|E|/(|V|(|V|-1)))")
            content.append("- **avg_degree**: Average degree of vertices (2|E|/|V|)")
            content.append("- **layout_time**: Time (in seconds) to compute the graph layout")
            content.append("- **degree_corr**: Spearman correlation between radial distance and degree centrality")
            content.append("- **betweenness_corr**: Spearman correlation between radial distance and betweenness centrality")
            content.append("- **eigenvector_corr**: Spearman correlation between radial distance and eigenvector centrality")
            content.append("- **pagerank_corr**: Spearman correlation between radial distance and PageRank\n")
        
        # Add influence descriptions if this is the influence benchmarks
        elif "influence" in title:
            content.append("## Influence Maximization Information\n")
            content.append("This benchmark compares different seed selection strategies for influence maximization:\n")
            content.append("- **GraphEm**: Our method that selects seeds based on the graph embedding")
            content.append("- **Greedy**: The greedy algorithm that iteratively selects the best node")
            content.append("- **Random**: Randomly selected seeds as a baseline\n")
            
            content.append("## Column Descriptions\n")
            content.append("- **graph_type**: Type of graph generator used")
            content.append("- **vertices**: Number of vertices in the graph")
            content.append("- **edges**: Number of edges in the graph")
            content.append("- **avg_degree**: Average degree of vertices (2|E|/|V|)")
            content.append("- **graphem_influence**: Number of nodes influenced by GraphEm seeds")
            content.append("- **greedy_influence**: Number of nodes influenced by greedy selection seeds")
            content.append("- **random_influence**: Number of nodes influenced by random seeds")
            content.append("- **graphem_time**: Time (in seconds) to select seeds using GraphEm")
            content.append("- **greedy_time**: Time (in seconds) to select seeds using greedy algorithm")
            content.append("- **graphem_norm_influence**: GraphEm influence normalized by graph size")
            content.append("- **greedy_norm_influence**: Greedy influence normalized by graph size")
            content.append("- **random_norm_influence**: Random influence normalized by graph size")
            content.append("- **graphem_efficiency**: GraphEm influence per second")
            content.append("- **greedy_efficiency**: Greedy influence per second\n")
        
        # Add the table
        content.append(df_display.to_markdown(index=False))
        
        # Add timestamp
        content.append(f"\n\n*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
        
        print(f"Saved Markdown results to {filepath}")
    
    def _save_latex(self, df, filepath, title):
        """Save DataFrame as LaTeX table."""
        # Create a copy to avoid modifying the original
        df_display = df.copy()
        
        # Format numeric columns
        for col in df_display.select_dtypes(include=['float']).columns:
            if "time" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            elif "correlation" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        # Generate LaTeX
        latex_table = df_display.to_latex(
            index=False,
            caption=title.replace('_', ' ').title(),
            label=f"tab:{title}",
            float_format="%.4f"
        )
        
        # Add document structure for standalone compilation
        latex_content = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{array}",
            "\\usepackage{caption}",
            "\\usepackage{longtable}",
            "\\usepackage{hyperref}",
            "\\begin{document}",
            f"\\section*{{{title.replace('_', ' ').title()}}}",
            "Generated by Graphem Benchmark Runner on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
        
        # Add dataset descriptions if this is the dataset benchmarks
        if "dataset" in title:
            latex_content.extend([
                "\\subsection*{Dataset Information}",
                "The datasets used in this benchmark are from the following sources:",
                "\\begin{itemize}",
                "\\item \\textbf{SNAP}: Stanford Network Analysis Project datasets (prefixed with 'snap-')",
                "  \\begin{itemize}",
                "  \\item \\textbf{facebook\\_combined}: Facebook social network",
                "  \\item \\textbf{ca-GrQc}: Collaboration network of Arxiv General Relativity",
                "  \\item \\textbf{ca-HepTh}: Collaboration network of Arxiv High Energy Physics Theory",
                "  \\end{itemize}",
                "\\item \\textbf{Network Repository}: Various network datasets (prefixed with 'netrepo-')",
                "\\item \\textbf{Semantic Scholar}: Academic citation networks (prefixed with 'semanticscholar-')",
                "\\end{itemize}",
                
                "\\subsection*{Column Descriptions}",
                "\\begin{itemize}",
                "\\item \\textbf{dataset}: Name of the dataset",
                "\\item \\textbf{original\\_vertices}: Number of vertices in the original dataset",
                "\\item \\textbf{original\\_edges}: Number of edges in the original dataset",
                "\\item \\textbf{sampled\\_vertices}: Number of vertices after sampling (if applied)",
                "\\item \\textbf{sampled\\_edges}: Number of edges after sampling (if applied)",
                "\\item \\textbf{density}: Edge density of the graph (2|E|/(|V|(|V|-1)))",
                "\\item \\textbf{avg\\_degree}: Average degree of vertices (2|E|/|V|)",
                "\\item \\textbf{lcc\\_size}: Size of the largest connected component",
                "\\item \\textbf{lcc\\_fraction}: Fraction of graph in the largest connected component",
                "\\item \\textbf{layout\\_time}: Time (in seconds) to compute the graph layout",
                "\\item \\textbf{degree\\_correlation}: Spearman correlation between radial distance and degree centrality",
                "\\item \\textbf{betweenness\\_correlation}: Spearman correlation between radial distance and betweenness centrality",
                "\\item \\textbf{eigenvector\\_correlation}: Spearman correlation between radial distance and eigenvector centrality",
                "\\item \\textbf{pagerank\\_correlation}: Spearman correlation between radial distance and PageRank",
                "\\end{itemize}"
            ])
        
        # Add generator descriptions if this is the generator benchmarks
        elif "generator" in title:
            latex_content.extend([
                "\\subsection*{Graph Generator Information}",
                "The graph generators used in this benchmark:",
                "\\begin{itemize}",
                "\\item \\textbf{Erdős–Rényi}: Random graph where each edge has equal probability",
                "\\item \\textbf{Random Regular}: Graph where all vertices have the same degree",
                "\\item \\textbf{Watts-Strogatz}: Small-world network with high clustering and short path lengths",
                "\\item \\textbf{Barabási-Albert}: Scale-free network generated by preferential attachment",
                "\\item \\textbf{Stochastic Block Model (SBM)}: Community-structured graph with dense within-community connections",
                "\\item \\textbf{Scale-Free}: Graph with power-law degree distribution",
                "\\item \\textbf{Random Geometric}: Graph where vertices are connected if within a distance threshold",
                "\\item \\textbf{Caveman}: Graph consisting of connected dense clusters",
                "\\end{itemize}",
                
                "\\subsection*{Column Descriptions}",
                "\\begin{itemize}",
                "\\item \\textbf{graph\\_type}: Type of graph generator used",
                "\\item \\textbf{vertices}: Number of vertices in the graph",
                "\\item \\textbf{edges}: Number of edges in the graph",
                "\\item \\textbf{density}: Edge density of the graph (2|E|/(|V|(|V|-1)))",
                "\\item \\textbf{avg\\_degree}: Average degree of vertices (2|E|/|V|)",
                "\\item \\textbf{layout\\_time}: Time (in seconds) to compute the graph layout",
                "\\item \\textbf{degree\\_corr}: Spearman correlation between radial distance and degree centrality",
                "\\item \\textbf{betweenness\\_corr}: Spearman correlation between radial distance and betweenness centrality",
                "\\item \\textbf{eigenvector\\_corr}: Spearman correlation between radial distance and eigenvector centrality",
                "\\item \\textbf{pagerank\\_corr}: Spearman correlation between radial distance and PageRank",
                "\\end{itemize}"
            ])
        
        # Add influence descriptions if this is the influence benchmarks
        elif "influence" in title:
            latex_content.extend([
                "\\subsection*{Influence Maximization Information}",
                "This benchmark compares different seed selection strategies for influence maximization:",
                "\\begin{itemize}",
                "\\item \\textbf{GraphEm}: Our method that selects seeds based on the graph embedding",
                "\\item \\textbf{Greedy}: The greedy algorithm that iteratively selects the best node",
                "\\item \\textbf{Random}: Randomly selected seeds as a baseline",
                "\\end{itemize}",
                
                "\\subsection*{Column Descriptions}",
                "\\begin{itemize}",
                "\\item \\textbf{graph\\_type}: Type of graph generator used",
                "\\item \\textbf{vertices}: Number of vertices in the graph",
                "\\item \\textbf{edges}: Number of edges in the graph",
                "\\item \\textbf{avg\\_degree}: Average degree of vertices (2|E|/|V|)",
                "\\item \\textbf{graphem\\_influence}: Number of nodes influenced by GraphEm seeds",
                "\\item \\textbf{greedy\\_influence}: Number of nodes influenced by greedy selection seeds",
                "\\item \\textbf{random\\_influence}: Number of nodes influenced by random seeds",
                "\\item \\textbf{graphem\\_time}: Time (in seconds) to select seeds using GraphEm",
                "\\item \\textbf{greedy\\_time}: Time (in seconds) to select seeds using greedy algorithm",
                "\\item \\textbf{graphem\\_norm\\_influence}: GraphEm influence normalized by graph size",
                "\\item \\textbf{greedy\\_norm\\_influence}: Greedy influence normalized by graph size",
                "\\item \\textbf{random\\_norm\\_influence}: Random influence normalized by graph size",
                "\\item \\textbf{graphem\\_efficiency}: GraphEm influence per second",
                "\\item \\textbf{greedy\\_efficiency}: Greedy influence per second",
                "\\end{itemize}"
            ])
            
        # Add the table and end document
        latex_content.extend([
            latex_table,
            "\\end{document}"
        ])
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex_content))
        
        print(f"Saved LaTeX results to {filepath}")
    
    def _save_html(self, df, filepath, title):
        """Save DataFrame as HTML table."""
        # Create a copy to avoid modifying the original
        df_display = df.copy()
        
        # Format numeric columns
        for col in df_display.select_dtypes(include=['float']).columns:
            if "time" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}s" if not pd.isna(x) else "N/A")
            elif "correlation" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
            elif "edges" in col or "vertices" in col:
                df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "N/A")
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title.replace('_', ' ').title()}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }",
            "table { border-collapse: collapse; width: 100%; margin-top: 20px; margin-bottom: 20px; }",
            "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }",
            "th { background-color: #f2f2f2; }",
            "tr:hover { background-color: #f5f5f5; }",
            ".timestamp { color: #666; font-size: 0.8em; margin-top: 20px; }",
            "h1 { color: #333; }",
            "h2 { color: #555; margin-top: 30px; }",
            "ul { margin-bottom: 20px; }",
            "li { margin-bottom: 5px; }",
            ".description { margin-bottom: 30px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title.replace('_', ' ').title()}</h1>"
        ]
        
        # Add dataset descriptions if this is the dataset benchmarks
        if "dataset" in title:
            html_content.extend([
                "<div class='description'>",
                "<h2>Dataset Information</h2>",
                "<p>The datasets used in this benchmark are from the following sources:</p>",
                "<ul>",
                "<li><strong>SNAP</strong>: Stanford Network Analysis Project datasets (prefixed with 'snap-')",
                "  <ul>",
                "  <li><strong>facebook_combined</strong>: Facebook social network</li>",
                "  <li><strong>ca-GrQc</strong>: Collaboration network of Arxiv General Relativity</li>",
                "  <li><strong>ca-HepTh</strong>: Collaboration network of Arxiv High Energy Physics Theory</li>",
                "  </ul>",
                "</li>",
                "<li><strong>Network Repository</strong>: Various network datasets (prefixed with 'netrepo-')</li>",
                "<li><strong>Semantic Scholar</strong>: Academic citation networks (prefixed with 'semanticscholar-')</li>",
                "</ul>",
                
                "<h2>Column Descriptions</h2>",
                "<ul>",
                "<li><strong>dataset</strong>: Name of the dataset</li>",
                "<li><strong>original_vertices</strong>: Number of vertices in the original dataset</li>",
                "<li><strong>original_edges</strong>: Number of edges in the original dataset</li>",
                "<li><strong>sampled_vertices</strong>: Number of vertices after sampling (if applied)</li>",
                "<li><strong>sampled_edges</strong>: Number of edges after sampling (if applied)</li>",
                "<li><strong>density</strong>: Edge density of the graph (2|E|/(|V|(|V|-1)))</li>",
                "<li><strong>avg_degree</strong>: Average degree of vertices (2|E|/|V|)</li>",
                "<li><strong>lcc_size</strong>: Size of the largest connected component</li>",
                "<li><strong>lcc_fraction</strong>: Fraction of graph in the largest connected component</li>",
                "<li><strong>layout_time</strong>: Time (in seconds) to compute the graph layout</li>",
                "<li><strong>degree_correlation</strong>: Spearman correlation between radial distance and degree centrality</li>",
                "<li><strong>betweenness_correlation</strong>: Spearman correlation between radial distance and betweenness centrality</li>",
                "<li><strong>eigenvector_correlation</strong>: Spearman correlation between radial distance and eigenvector centrality</li>",
                "<li><strong>pagerank_correlation</strong>: Spearman correlation between radial distance and PageRank</li>",
                "</ul>",
                "</div>"
            ])
        
        # Add generator descriptions if this is the generator benchmarks
        elif "generator" in title:
            html_content.extend([
                "<div class='description'>",
                "<h2>Graph Generator Information</h2>",
                "<p>The graph generators used in this benchmark:</p>",
                "<ul>",
                "<li><strong>Erdős–Rényi</strong>: Random graph where each edge has equal probability</li>",
                "<li><strong>Random Regular</strong>: Graph where all vertices have the same degree</li>",
                "<li><strong>Watts-Strogatz</strong>: Small-world network with high clustering and short path lengths</li>",
                "<li><strong>Barabási-Albert</strong>: Scale-free network generated by preferential attachment</li>",
                "<li><strong>Stochastic Block Model (SBM)</strong>: Community-structured graph with dense within-community connections</li>",
                "<li><strong>Scale-Free</strong>: Graph with power-law degree distribution</li>",
                "<li><strong>Random Geometric</strong>: Graph where vertices are connected if within a distance threshold</li>",
                "<li><strong>Caveman</strong>: Graph consisting of connected dense clusters</li>",
                "</ul>",
                
                "<h2>Column Descriptions</h2>",
                "<ul>",
                "<li><strong>graph_type</strong>: Type of graph generator used</li>",
                "<li><strong>vertices</strong>: Number of vertices in the graph</li>",
                "<li><strong>edges</strong>: Number of edges in the graph</li>",
                "<li><strong>density</strong>: Edge density of the graph (2|E|/(|V|(|V|-1)))</li>",
                "<li><strong>avg_degree</strong>: Average degree of vertices (2|E|/|V|)</li>",
                "<li><strong>layout_time</strong>: Time (in seconds) to compute the graph layout</li>",
                "<li><strong>degree_corr</strong>: Spearman correlation between radial distance and degree centrality</li>",
                "<li><strong>betweenness_corr</strong>: Spearman correlation between radial distance and betweenness centrality</li>",
                "<li><strong>eigenvector_corr</strong>: Spearman correlation between radial distance and eigenvector centrality</li>",
                "<li><strong>pagerank_corr</strong>: Spearman correlation between radial distance and PageRank</li>",
                "</ul>",
                "</div>"
            ])
        
        # Add influence descriptions if this is the influence benchmarks
        elif "influence" in title:
            html_content.extend([
                "<div class='description'>",
                "<h2>Influence Maximization Information</h2>",
                "<p>This benchmark compares different seed selection strategies for influence maximization:</p>",
                "<ul>",
                "<li><strong>GraphEm</strong>: Our method that selects seeds based on the graph embedding</li>",
                "<li><strong>Greedy</strong>: The greedy algorithm that iteratively selects the best node</li>",
                "<li><strong>Random</strong>: Randomly selected seeds as a baseline</li>",
                "</ul>",
                
                "<h2>Column Descriptions</h2>",
                "<ul>",
                "<li><strong>graph_type</strong>: Type of graph generator used</li>",
                "<li><strong>vertices</strong>: Number of vertices in the graph</li>",
                "<li><strong>edges</strong>: Number of edges in the graph</li>",
                "<li><strong>avg_degree</strong>: Average degree of vertices (2|E|/|V|)</li>",
                "<li><strong>graphem_influence</strong>: Number of nodes influenced by GraphEm seeds</li>",
                "<li><strong>greedy_influence</strong>: Number of nodes influenced by greedy selection seeds</li>",
                "<li><strong>random_influence</strong>: Number of nodes influenced by random seeds</li>",
                "<li><strong>graphem_time</strong>: Time (in seconds) to select seeds using GraphEm</li>",
                "<li><strong>greedy_time</strong>: Time (in seconds) to select seeds using greedy algorithm</li>",
                "<li><strong>graphem_norm_influence</strong>: GraphEm influence normalized by graph size</li>",
                "<li><strong>greedy_norm_influence</strong>: Greedy influence normalized by graph size</li>",
                "<li><strong>random_norm_influence</strong>: Random influence normalized by graph size</li>",
                "<li><strong>graphem_efficiency</strong>: GraphEm influence per second</li>",
                "<li><strong>greedy_efficiency</strong>: Greedy influence per second</li>",
                "</ul>",
                "</div>"
            ])
        
        # Format DataFrame as HTML table
        html_table = df_display.to_html(index=False, classes="table table-striped")
        html_content.append(html_table)
        
        # Add timestamp
        html_content.extend([
            f"<p class='timestamp'>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "</body>",
            "</html>"
        ])
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        
        print(f"Saved HTML results to {filepath}")
    
    def generate_summary_report(self, total_time):
        """
        Generate a summary report of all benchmarks.
        
        Parameters:
            total_time: float
                Total execution time in seconds
        """
        # Create summary file
        summary_path = self.run_dir / "summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Graphem Benchmark Summary\n\n")
            f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total execution time:** {total_time:.2f} seconds\n\n")
            
            # List all benchmarks run
            f.write("## Benchmarks Executed\n\n")
            for result_name in self.results:
                f.write(f"- {result_name.replace('_', ' ').title()}\n")
            
            # Add links to detailed results
            f.write("\n## Result Files\n\n")
            for result_name in self.results:
                f.write(f"### {result_name.replace('_', ' ').title()}\n\n")
                for fmt in self.formats:
                    rel_path = f"{result_name}/{result_name}.{fmt}"
                    f.write(f"- [{fmt.capitalize()} Report]({rel_path})\n")
                f.write(f"- [Raw Data (CSV)]({result_name}/{result_name}.csv)\n\n")
        
        print(f"Summary report saved to {summary_path}")


def main(argv=None):
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Run Graphem benchmarks")
    
    parser.add_argument(
        "--output", "-o", 
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--formats", "-f",
        nargs="+",
        choices=["markdown", "latex", "html"],
        default=["markdown", "latex"],
        help="Output formats (default: markdown latex)"
    )
    
    parser.add_argument(
        "--subsample", "-s",
        type=int,
        default=None,
        help="Subsample vertices for large graphs (default: no subsampling)"
    )
    
    parser.add_argument(
        "--profile", "-p",
        action="store_true",
        help="Run with cProfile profiling (outputs profile.prof for visualization with snakeviz)"
    )
    
    parser.add_argument(
        "--line-profile", "-l",
        action="store_true",
        help="Run with line profiling (detailed line-by-line performance analysis)"
    )
    
    parser.add_argument(
        "--torch-profile", "-t",
        action="store_true",
        help="Run with PyTorch profiling for GPU operations (outputs to profile_torch directory)"
    )

    # If imported and called (argv=None & not __main__), use defaults
    if argv is None and __name__ != "__main__":
        args = parser.parse_args([])
    else:
        args = parser.parse_args(argv)
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=args.output, formats=args.formats, subsample_size=args.subsample)
    
    # Handle different profiling options
    if args.profile:
        print("\n--- Running with cProfile ---")
        # Use cProfile
        profile_path = "profile.prof"
        cProfile.runctx('runner.run_all_benchmarks()', globals(), locals(), filename=profile_path)
        
        # Print summary to console
        print(f"\n--- Profile saved to {profile_path} ---")
        print("Top 20 functions by cumulative time:")
        p = pstats.Stats(profile_path)
        p.strip_dirs().sort_stats('cumulative').print_stats(20)
        print(f"\nTo visualize: snakeviz {profile_path}")
    
    elif args.line_profile:
        print("\n--- Running with line_profiler ---")
        # Use line_profiler for detailed line-by-line analysis
        profiler = LineProfiler()
        
        # Profile the most important methods
        profiler.add_function(GraphEmbedderPyTorch.run_layout)
        profiler.add_function(runner.run_generator_benchmarks)
        profiler.add_function(runner.run_dataset_benchmarks)
        
        # Run with profiling
        profiler.runctx('runner.run_all_benchmarks()', globals(), locals())
        
        # Output results
        profiler.print_stats()
    
    elif args.torch_profile and TORCH_AVAILABLE:
        print("\n--- Running with PyTorch profiler ---")
        # Use PyTorch profiler for GPU operations
        profile_dir = "profile_torch"
        os.makedirs(profile_dir, exist_ok=True)

        # Start PyTorch profiling
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            with_stack=True
        ) as prof:
            # Run benchmarks
            runner.run_all_benchmarks()

        print(f"\n--- PyTorch profile saved to {profile_dir} ---")
        print("To visualize: tensorboard --logdir=profile_torch")
    
    else:
        # Run without profiling
        runner.run_all_benchmarks()


if __name__ == "__main__":
    main()
