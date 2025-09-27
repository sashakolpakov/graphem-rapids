"""
Influence maximization functionality for Graphem.
"""

import numpy as np
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep


def graphem_seed_selection(embedder, k, num_iterations=20):
    """
    Run the GraphEmbedder layout to get an embedding, then select
    k seeds by choosing the nodes with the highest radial distances.
    
    Parameters:
        embedder: GraphEmbedder
            The initialized graph embedder object
        k: int
            Number of seed nodes to select
        num_iterations: int
            Number of layout iterations to run
    
    Returns:
        seeds: list
            List of k vertices selected as seeds
    """
    # Run the layout algorithm
    embedder.run_layout(num_iterations=num_iterations)
    
    # Compute radial distances from the origin (0, 0, 0)
    positions = np.array(embedder.positions)
    radial_distances = np.linalg.norm(positions, axis=1)
    
    # Select the k nodes with the highest radial distances
    seeds = np.argsort(-radial_distances)[:k]
    
    return seeds.tolist()


def ndlib_estimated_influence(G, seeds, p=0.1, iterations_count=200):
    """
    Run NDlib's Independent Cascades model on graph G, starting with the given seeds,
    and return the estimated final influence (number of nodes in state 2) and
    the number of iterations executed.
    
    Parameters:
        G: networkx.Graph
            The graph to run influence propagation on
        seeds: list
            The list of seed nodes
        p: float
            Propagation probability
        iterations_count: int
            Maximum number of simulation iterations
    
    Returns:
        influence: float
            The estimated influence (average number of influenced nodes)
        iterations: int
            The number of iterations run
    """
    # Configure the Independent Cascades model
    model = ep.IndependentCascadesModel(G)
    config = mc.Configuration()
    
    # Set edge propagation probabilities
    for e in G.edges():
        config.add_edge_configuration("threshold", e, p)
    
    # Initialize the model with configuration
    model.set_initial_status(config)
    
    # Set initial seeds to infected state
    for seed in seeds:
        config.add_node_configuration("status", seed, 1)
    
    # Run the simulation
    iterations = model.iteration_bunch(iterations_count)
    
    # Get the number of nodes in state 2 (influenced) at the end
    final_status = iterations[-1]['status']
    influenced_count = sum(1 for node_state in final_status.values() if node_state == 2)
    
    return influenced_count, len(iterations)


def greedy_seed_selection(G, k, p=0.1, iterations_count=200):
    """
    Greedy seed selection using NDlib influence estimation.
    For each candidate node evaluation, it calls NDlib's simulation and accumulates
    the total number of iterations used across all evaluations.

    Returns:
        seeds: the selected seed set (list of nodes)
        total_iters: the total number of NDlib iterations run during selection.
    """
    seeds = []
    total_iters = 0
    n = G.number_of_nodes()
    
    # Create a copy of the graph for evaluations
    G_copy = G.copy()
    
    # Marginal gain of each node
    for _ in range(k):
        best_node = None
        best_influence = -1
        
        # Evaluate each node not already in the seed set
        for node in range(n):
            if node in seeds:
                continue
                
            # Evaluate influence with this node added to the seed set
            candidate_seeds = seeds + [node]
            influence, iters = ndlib_estimated_influence(G_copy, candidate_seeds, p, iterations_count)
            total_iters += iters
            
            # Update best node if influence is improved
            if influence > best_influence:
                best_influence = influence
                best_node = node
        
        # Add the best node to the seed set
        if best_node is not None:
            seeds.append(best_node)
    
    return seeds, total_iters
