"""
Benchmark functionality for GraphEm Rapids.
"""

import time
import logging
import numpy as np
import networkx as nx
from scipy import stats

from .backends.embedder_pytorch import GraphEmbedderPyTorch
from .influence import graphem_seed_selection, ndlib_estimated_influence, greedy_seed_selection

logger = logging.getLogger(__name__)


def run_benchmark(graph_generator, graph_params, dim=3, L_min=10.0, k_attr=0.5, k_inter=0.1,
                 knn_k=15, sample_size=512, batch_size=1024, num_iterations=40, backend='pytorch', **kwargs):
    """
    Run a benchmark on the given graph using GraphEm Rapids.

    Parameters
    ----------
    graph_generator : callable
        Function to generate a graph
    graph_params : dict
        Parameters for the graph generator
    dim : int, default=3
        Embedding dimension
    L_min, k_attr, k_inter, knn_k : float
        GraphEmbedder parameters
    sample_size, batch_size : int
        Batch parameters for kNN search
    num_iterations : int, default=40
        Number of layout iterations
    backend : str, default='pytorch'
        Backend to use ('pytorch', 'cuvs', or 'auto')
    **kwargs
        Additional parameters passed to the embedder

    Returns
    -------
    dict
        Benchmark results including timings and graph metrics
    """
    logger.info("Running benchmark with %s...", graph_generator.__name__)

    # Generate the graph
    start_time = time.time()
    edges = graph_generator(**graph_params)

    # Count vertices and edges
    if len(edges) > 0:
        n = max(np.max(edges) + 1, graph_params.get('n', 0))
    else:
        n = graph_params.get('n', 0)
    m = len(edges)

    logger.info("Generated graph with %d vertices and %d edges", n, m)

    # Convert to NetworkX graph for centrality calculations
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(n))
    nx_graph.add_edges_from(edges)

    # Calculate centrality measures
    logger.info("Calculating centrality measures...")
    degree = np.array([d for _, d in nx_graph.degree()])

    betweenness = np.zeros(n)
    btw_dict = nx.betweenness_centrality(nx_graph)
    for i, val in btw_dict.items():
        betweenness[i] = val

    eigenvector = np.zeros(n)
    try:
        eig_dict = nx.eigenvector_centrality_numpy(nx_graph)
        for i, val in eig_dict.items():
            eigenvector[i] = val
    except (nx.NetworkXError, nx.AmbiguousSolution) as e:
        logger.warning("Eigenvector centrality calculation failed: %s", e)
        logger.warning("Setting eigenvector centrality to degree centrality as fallback")
        # Use degree centrality as a fallback
        deg_dict = nx.degree_centrality(nx_graph)
        for i, val in deg_dict.items():
            eigenvector[i] = val

    pagerank = np.zeros(n)
    pr_dict = nx.pagerank(nx_graph)
    for i, val in pr_dict.items():
        pagerank[i] = val

    closeness = np.zeros(n)
    close_dict = nx.closeness_centrality(nx_graph)
    for i, val in close_dict.items():
        closeness[i] = val

    node_load = np.zeros(n)
    node_load_dict = nx.load_centrality(nx_graph)
    for i, val in node_load_dict.items():
        node_load[i] = val

    # Create embedder - using PyTorch backend for now
    logger.info("Creating embedder...")
    embedder = GraphEmbedderPyTorch(
        edges=edges,
        n_vertices=n,
        dimension=dim,
        L_min=L_min,
        k_attr=k_attr,
        k_inter=k_inter,
        knn_k=knn_k,
        sample_size=min(sample_size, len(edges)),
        batch_size=min(batch_size, n),
        verbose=True,
        **kwargs
    )

    # Run layout and get embedding
    logger.info("Running layout for %d iterations...", num_iterations)
    layout_start = time.time()
    embedder.run_layout(num_iterations=num_iterations)
    layout_time = time.time() - layout_start

    # Get positions and calculate radial distances
    positions = np.array(embedder.positions.cpu().numpy() if hasattr(embedder.positions, 'cpu') else embedder.positions)
    radii = np.linalg.norm(positions, axis=1)

    # Return benchmark data
    result = {
        'n': n,
        'm': m,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0.0,
        'avg_degree': 2 * m / n if n > 0 else 0.0,
        'layout_time': layout_time,
        'graph_type': graph_generator.__name__,
        'dimension': dim,
        'backend': backend,
        'radii': radii,
        'positions': positions,
        'degree': degree,
        'betweenness': betweenness,
        'eigenvector': eigenvector,
        'pagerank': pagerank,
        'closeness': closeness,
        'node_load': node_load
    }

    total_time = time.time() - start_time
    result['total_time'] = total_time

    logger.info("Benchmark completed in %.2f seconds", total_time)
    return result


def benchmark_correlations(graph_generator, graph_params, dim=2, L_min=10.0, k_attr=0.5, k_inter=0.1,
                          knn_k=15, sample_size=512, batch_size=1024, num_iterations=40, backend='pytorch', **kwargs):
    """
    Run a benchmark to calculate correlations between embedding radii and centrality measures.

    Parameters
    ----------
    graph_generator : callable
        Function to generate a graph
    graph_params : dict
        Parameters for the graph generator
    Other parameters : same as run_benchmark

    Returns
    -------
    dict
        Benchmark results with correlations
    """
    # Run the benchmark to get basic metrics
    results = run_benchmark(
        graph_generator,
        graph_params,
        dim=dim,
        L_min=L_min,
        k_attr=k_attr,
        k_inter=k_inter,
        knn_k=knn_k,
        sample_size=sample_size,
        batch_size=batch_size,
        num_iterations=num_iterations,
        backend=backend,
        **kwargs
    )

    # Calculate correlations with radial distances
    radii = results['radii']
    correlations = {}

    # Degree correlation
    rho, p = stats.spearmanr(radii, results['degree'])
    correlations['degree'] = {'rho': rho, 'p': p}

    # Betweenness correlation
    rho, p = stats.spearmanr(radii, results['betweenness'])
    correlations['betweenness'] = {'rho': rho, 'p': p}

    # Eigenvector correlation
    rho, p = stats.spearmanr(radii, results['eigenvector'])
    correlations['eigenvector'] = {'rho': rho, 'p': p}

    # PageRank correlation
    rho, p = stats.spearmanr(radii, results['pagerank'])
    correlations['pagerank'] = {'rho': rho, 'p': p}

    # Closeness correlation
    rho, p = stats.spearmanr(radii, results['closeness'])
    correlations['closeness'] = {'rho': rho, 'p': p}

    # Node load correlation
    rho, p = stats.spearmanr(radii, results['node_load'])
    correlations['node_load'] = {'rho': rho, 'p': p}

    # Add correlations to results
    results['correlations'] = correlations

    return results


def run_influence_benchmark(graph_generator, graph_params, k=10, p=0.1, iterations=200,
                           dim=3, num_layout_iterations=20, layout_params=None, backend='pytorch'):
    """
    Run a benchmark comparing influence maximization methods.

    Parameters
    ----------
    graph_generator : callable
        Function to generate a graph
    graph_params : dict
        Parameters for the graph generator
    k : int, default=10
        Number of seed nodes to select
    p : float, default=0.1
        Propagation probability
    iterations : int, default=200
        Number of iterations for influence simulation
    dim : int, default=3
        Embedding dimension
    num_layout_iterations : int, default=20
        Number of iterations for layout algorithm
    layout_params : dict, optional
        Parameters for the layout algorithm
    backend : str, default='pytorch'
        Backend to use for embedding

    Returns
    -------
    dict
        Benchmark results comparing influence maximization methods
    """
    logger.info("Running influence benchmark with %s...", graph_generator.__name__)

    # Generate the graph
    start_time = time.time()
    edges = graph_generator(**graph_params)

    # Count vertices and edges
    if len(edges) > 0:
        n = max(np.max(edges) + 1, graph_params.get('n', 0))
    else:
        n = graph_params.get('n', 0)
    m = len(edges)

    logger.info("Generated graph with %d vertices and %d edges", n, m)

    # Convert to NetworkX graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(n))
    nx_graph.add_edges_from(edges)

    # Default layout parameters
    if layout_params is None:
        layout_params = {
            'L_min': 10.0,
            'k_attr': 0.5,
            'k_inter': 0.1,
            'knn_k': 15,
            'sample_size': 512,
            'batch_size': 1024
        }

    # Create embedder
    logger.info("Creating embedder...")
    embedder = GraphEmbedderPyTorch(
        edges=edges,
        n_vertices=n,
        dimension=dim,
        verbose=True,
        **layout_params
    )

    # Run GraphEm-based seed selection
    logger.info("Running GraphEm seed selection...")
    graphem_start = time.time()
    graphem_seeds = graphem_seed_selection(embedder, k, num_iterations=num_layout_iterations)
    graphem_time = time.time() - graphem_start

    # Run greedy seed selection
    logger.info("Running greedy seed selection...")
    greedy_start = time.time()
    greedy_seeds, greedy_iters = greedy_seed_selection(nx_graph, k, p, iterations)
    greedy_time = time.time() - greedy_start

    # Evaluate influence for GraphEm seeds
    logger.info("Evaluating GraphEm influence...")
    graphem_eval_start = time.time()
    graphem_influence, _ = ndlib_estimated_influence(nx_graph, graphem_seeds, p, iterations)
    graphem_eval_time = time.time() - graphem_eval_start

    # Evaluate influence for Greedy seeds
    logger.info("Evaluating Greedy influence...")
    greedy_eval_start = time.time()
    greedy_influence, _ = ndlib_estimated_influence(nx_graph, greedy_seeds, p, iterations)
    greedy_eval_time = time.time() - greedy_eval_start

    # Calculate random baseline
    logger.info("Evaluating random baseline...")
    random_influences = []
    for _ in range(10):
        random_seeds = np.random.choice(n, k, replace=False)
        random_influence, _ = ndlib_estimated_influence(nx_graph, random_seeds, p, iterations)
        random_influences.append(random_influence)
    random_influence = np.mean(random_influences)

    # Compile results
    results = {
        'graph_type': graph_generator.__name__,
        'n': n,
        'm': m,
        'backend': backend,
        'graphem_seeds': graphem_seeds,
        'greedy_seeds': greedy_seeds,
        'graphem_influence': graphem_influence,
        'greedy_influence': greedy_influence,
        'random_influence': random_influence,
        'graphem_time': graphem_time,
        'greedy_time': greedy_time,
        'graphem_eval_time': graphem_eval_time,
        'greedy_eval_time': greedy_eval_time,
        'greedy_iterations': greedy_iters,
        'graphem_norm_influence': graphem_influence / n,
        'greedy_norm_influence': greedy_influence / n,
        'random_norm_influence': random_influence / n
    }

    # Calculate efficiency ratios
    results['graphem_efficiency'] = results['graphem_norm_influence'] / graphem_time if graphem_time > 0 else 0
    results['greedy_efficiency'] = results['greedy_norm_influence'] / greedy_time if greedy_time > 0 else 0

    total_time = time.time() - start_time
    results['total_time'] = total_time

    logger.info("Influence benchmark completed")
    return results