#!/usr/bin/env python
"""
Test script for working with real-world datasets in Graphem.

This script demonstrates how to download, load, and analyze real-world graph datasets
from various sources including SNAP, Network Repository, and Semantic Scholar.
"""

import time
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px

from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.visualization import report_full_correlation_matrix
from graphem_rapids.datasets import (
    list_available_datasets,
    load_dataset,
)


def print_available_datasets():
    """
    Print information about all available datasets.
    """
    print(f"\n{'='*75}")
    print("Available Real-World Datasets")
    print(f"{'='*75}")
    
    datasets = list_available_datasets()
    
    # Group by source
    by_source = {}
    for dataset_id, info in datasets.items():
        source = info['source']
        if source not in by_source:
            by_source[source] = []
        by_source[source].append((dataset_id, info))
    
    # Print by source
    for source, dataset_list in by_source.items():
        print(f"{'-'*25}")
        print(f"\n{source} Datasets:")
        print(f"{'-'*25}")
        
        for dataset_id, info in dataset_list:
            nodes = info.get('nodes', 'Unknown')
            edges = info.get('edges', 'Unknown')
            print(f"- {dataset_id}: {info['description']}")
            if nodes != 'Unknown' and edges != 'Unknown':
                print(f"  ({nodes:,} nodes, {edges:,} edges)")
        
    print("\nTo use a dataset, call load_dataset('dataset-name') or load_dataset_as_networkx('dataset-name')")


def analyze_dataset(dataset_name, sample_size=None, dim=3, num_iterations=30):
    """
    Download, load, and analyze a dataset.
    
    Parameters:
        dataset_name: str
            Name of the dataset to analyze
        sample_size: int, optional
            If set, sample the graph to this number of nodes for visualization
        dim: int
            Dimension of the embedding
        num_iterations: int
            Number of layout iterations
    """
    print(f"\n{'='*75}")
    print(f"Analyzing dataset: {dataset_name}")
    print(f"{'='*75}")
    
    # Load the dataset
    print(f"Loading dataset {dataset_name}...")
    start_time = time.time()
    vertices, edges = load_dataset(dataset_name)
    n_vertices = len(vertices)
    load_time = time.time() - start_time
    
    print(f"Loaded dataset with {n_vertices:,} vertices and {adjacency.nnz//2:,} edges in {load_time:.2f}s")
    
    # Sample the graph if needed
    if sample_size is not None and sample_size < n_vertices:
        print(f"Sampling {sample_size:,} vertices from the graph...")
        sampled_vertices = np.random.choice(vertices, sample_size, replace=False)
        
        # Filter edges that contain sampled vertices
        sampled_edges = []
        for u, v in edges:
            if u in sampled_vertices and v in sampled_vertices:
                sampled_edges.append((u, v))
        
        vertices = sampled_vertices
        edges = np.array(sampled_edges)
        n_vertices = sample_size
        
        print(f"Sampled graph has {n_vertices:,} vertices and {adjacency.nnz//2:,} edges")
    
    # Create NetworkX graph for analysis
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G,
                                           first_label=0,
                                           ordering='default',
                                           label_attribute=None)
    
    # Analyze graph properties
    density = 2 * adjacency.nnz//2 / (n_vertices * (n_vertices - 1))
    avg_degree = 2 * adjacency.nnz//2 / n_vertices
    
    print("Graph statistics:")
    print(f"- Density: {density:.6f}")
    print(f"- Average degree: {avg_degree:.2f}")
    
    # Measure connected components
    components = list(nx.connected_components(G))
    print(f"- Number of connected components: {len(components):,}")
    print(f"- Largest component size: {len(max(components, key=len)):,} vertices")
    
    # Analyze largest connected component
    G_cc = G
    largest_cc = max(components, key=len)
    if len(largest_cc) < n_vertices:
        print(f"Extracting largest connected component with {len(largest_cc):,} vertices...")
        G_cc = G.subgraph(largest_cc).copy()
        
        # Re-index nodes to be consecutive integers
        G_cc = nx.convert_node_labels_to_integers(G_cc)
        
        # Extract edges from the largest component
        n_vertices = len(largest_cc)
    
    # Compute diameter if manageable
    if n_vertices < 10000:
        try:
            diameter = nx.diameter(G_cc)
            print(f"- Diameter: {diameter}")
        except nx.NetworkXError as e:
            print("- Diameter: N/A")
            print(e)
    else:
        print("- Diameter: Skipped (Graph too large)")
    
    # Compute average shortest path length if manageable
    if n_vertices < 10000:
        try:
            avg_path_length = nx.average_shortest_path_length(G_cc)
            print(f"- Average shortest path length: {avg_path_length:.2f}")
        except nx.NetworkXError as e:
            print("- Average shortest path length: N/A")
            print(e)
    else:
        print("- Average shortest path length: Skipped (Graph too large)")
    
    # Compute clustering coefficient
    avg_clustering = nx.average_clustering(G_cc)
    print(f"- Average clustering coefficient: {avg_clustering:.4f}")
    
    # Create and run embedder
    print(f"Creating embedding in dimension {dim}...")
    # Create and run embedder
    embedder = GraphEmbedderPyTorch(
        adjacency=G_cc.edges,
        n_vertices=G_cc.number_of_nodes(),
        n_components=dim,
        L_min=4.0,
        k_attr=0.5,
        k_inter=0.1,
        n_neighbors=min(15, G_cc.number_of_nodes() // 10),
        sample_size=min(512, G_cc.number_of_edges()),
        verbose=False
    )
    
    print(f"Running layout for {num_iterations} iterations...")
    layout_start = time.time()
    embedder.run_layout(num_iterations=num_iterations)
    layout_time = time.time() - layout_start
    print(f"Layout completed in {layout_time:.2f}s")
    
    # Calculate centrality measures
    print("Calculating centrality measures...")
    
    # Get positions and calculate radial distances
    positions = np.array(embedder.positions)
    radii = np.linalg.norm(positions, axis=1)
    
    # Calculate centrality measures
    degree = np.array([d for _, d in G_cc.degree()])
    
    # Only calculate betweenness for smaller graphs
    if n_vertices < 5000:
        print("Calculating betweenness centrality...")
        betweenness = np.array(list(nx.betweenness_centrality(G_cc).values()))
    else:
        print("Skipping betweenness centrality (graph too large)")
        betweenness = np.zeros(n_vertices)
    
    print("Calculating eigenvector centrality...")
    try:
        eigenvector = np.array(list(nx.eigenvector_centrality_numpy(G_cc).values()))
    except nx.NetworkXError as e:
        print("Error calculating eigenvector centrality, using zeros")
        print(e)
        eigenvector = np.zeros(n_vertices)
    
    print("Calculating PageRank...")
    pagerank = np.array(list(nx.pagerank(G_cc).values()))
    
    print("Calculating closeness centrality...")
    if n_vertices < 5000:
        closeness = np.array(list(nx.closeness_centrality(G_cc).values()))
    else:
        print("Skipping closeness centrality (graph too large)")
        closeness = np.zeros(n_vertices)

    # Only calculate node load for smaller graphs
    if n_vertices < 5000:
        print("Calculating node load centrality...")
        node_load = np.array(list(nx.load_centrality(G_cc).values()))
    else:
        print("Skipping node load centrality (graph too large)")
        node_load = np.zeros(n_vertices)
    
    # Display correlation with radial distances
    print("\nCorrelation between embedding radii and centrality measures:")
    report_full_correlation_matrix(
        radii, 
        degree,
        betweenness,
        eigenvector,
        pagerank,
        closeness,
        node_load
    )
    
    # Display the graph
    print("Displaying graph layout...")
    embedder.display_layout(edge_width=1, node_size=5)
    
    # Color nodes by degree centrality
    print("Displaying graph layout with nodes colored by degree centrality...")
    normalized_degree = (degree - np.min(degree)) / (np.max(degree) - np.min(degree) + 1e-10)
    embedder.display_layout(edge_width=1, node_size=5, node_colors=normalized_degree)
    
    return embedder, G_cc


def compare_datasets(dataset_names, sample_size=1000, dim=3, num_iterations=30):
    """
    Compare multiple datasets by analyzing their properties.
    
    Parameters:
        dataset_names: list
            The list of dataset names to compare
        sample_size: int
            Sample size for each dataset
        dim: int
            Dimension of the embedding
        num_iterations: int
            Number of layout iterations
    """
    print(f"\n{'='*75}")
    print("Comparing Multiple Datasets")
    print(f"{'='*75}")
    
    results = []
    
    for dataset_name in dataset_names:
        print(f"\n{'-'*25}")
        print(f"Dataset: {dataset_name}")
        print(f"{'-'*25}")
        
        # Load the dataset
        print(f"Loading dataset {dataset_name}...")
        start_time = time.time()
        vertices, edges = load_dataset(dataset_name)
        n_vertices = len(vertices)
        load_time = time.time() - start_time
        
        print(f"Loaded dataset with {n_vertices:,} vertices and {adjacency.nnz//2:,} edges in {load_time:.2f}s")
        
        # Sample the graph
        print(f"Sampling {sample_size:,} vertices from the graph...")
        sampled_vertices = np.random.choice(vertices, sample_size, replace=False)
        
        # Filter edges that contain sampled vertices
        sampled_edges = []
        for u, v in edges:
            if u in sampled_vertices and v in sampled_vertices:
                sampled_edges.append((u, v))

        vertices = sampled_vertices
        edges = np.array(sampled_edges)
        n_vertices = sample_size
        
        print(f"Sampled graph has {n_vertices:,} vertices and {adjacency.nnz//2:,} edges")
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        G = nx.convert_node_labels_to_integers(G,
                                               first_label=0,
                                               ordering='default',
                                               label_attribute=None)
        
        # Analyze graph properties
        density = 2 * adjacency.nnz//2 / (n_vertices * (n_vertices - 1))
        avg_degree = 2 * adjacency.nnz//2 / n_vertices
        
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
        
        # Compute average shortest path length if manageable
        try:
            diameter = nx.diameter(G_cc)
        except nx.NetworkXError as e:
            print("- Diameter: N/A")
            print(e)
            diameter = float('nan')
        
        try:
            avg_path_length = nx.average_shortest_path_length(G_cc)
        except nx.NetworkXError as e:
            print("- Average shortest path length: N/A")
            print(e)
            avg_path_length = float('nan')
        
        # Compute clustering coefficient
        avg_clustering = nx.average_clustering(G_cc)
        
        # Create and run embedder
        embedder = GraphEmbedderPyTorch(
            adjacency=edges,
            n_vertices=n_vertices,
            n_components=dim,
            L_min=4.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=min(15, n_vertices // 10),
            sample_size=min(512, adjacency.nnz//2),
            verbose=False
        )
        
        layout_start = time.time()
        embedder.run_layout(num_iterations=num_iterations)
        layout_time = time.time() - layout_start
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'vertices': n_vertices,
            'edges': adjacency.nnz//2,
            'density': density,
            'avg_degree': avg_degree,
            'lcc_size': lcc_size,
            'lcc_fraction': lcc_fraction,
            'diameter': diameter,
            'avg_path_length': avg_path_length,
            'avg_clustering': avg_clustering,
            'layout_time': layout_time
        })
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    print("\nDataset Comparison Results:")
    print(df)
    
    # Create visualizations
    # Plot layout time vs edges
    fig = px.scatter(
        df, x='edges', y='layout_time', 
        hover_data=['dataset', 'vertices', 'avg_degree'],
        title='Layout Time vs. Number of Edges',
        labels={'edges': 'Number of Edges', 'layout_time': 'Layout Time (seconds)'}
    )
    fig.show()
    
    # Plot clustering coefficient vs average degree
    fig = px.scatter(
        df, x='avg_degree', y='avg_clustering', 
        hover_data=['dataset', 'vertices', 'edges'],
        title='Clustering Coefficient vs. Average Degree',
        labels={'avg_degree': 'Average Degree', 'avg_clustering': 'Average Clustering Coefficient'}
    )
    fig.show()
    
    return df


def main():
    """
    Main function to demonstrate working with real-world datasets.
    """
    # Print available datasets
    print_available_datasets()
    
    # Analyze a small social network dataset
    analyze_dataset('snap-facebook_combined', sample_size=None, dim=3, num_iterations=50)

    # Analyze a medium-sized dataset with sampling
    analyze_dataset('snap-ca-GrQc', sample_size=3500, dim=3, num_iterations=30)
    
    # Compare multiple datasets
    compare_datasets([
        'snap-facebook_combined',
        'snap-ca-GrQc',
        'snap-ca-HepTh',
        'snap-wiki-vote'
    ], sample_size=1500, dim=3, num_iterations=20)


if __name__ == "__main__":
    main()
