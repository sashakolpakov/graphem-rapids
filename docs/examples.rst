Examples
========

Basic Embedding
---------------

::

    import graphem_rapids as gr

    # Generate and embed
    adjacency = gr.generate_er(n=1000, p=0.01, seed=42)
    embedder = gr.create_graphem(adjacency, n_components=3)
    embedder.run_layout(num_iterations=50)

    # Get results
    positions = embedder.get_positions()
    embedder.display_layout()

Community Detection Visualization
----------------------------------

::

    import graphem_rapids as gr

    # Generate graph with communities
    adjacency, labels = gr.generate_sbm(
        n_per_block=100,
        num_blocks=5,
        p_in=0.2,
        p_out=0.01,
        labels=True,
        seed=42
    )

    # Compute 2D embedding
    embedder = gr.create_graphem(adjacency, n_components=2)
    embedder.run_layout(num_iterations=50)

    # Visualize with community colors
    embedder.display_layout(node_colors=labels, node_size=5)

Influence Maximization
-----------------------

::

    import graphem_rapids as gr
    import networkx as nx

    # Generate graph
    adjacency = gr.generate_er(n=1000, p=0.01, seed=42)

    # Compute embedding
    embedder = gr.create_graphem(adjacency, n_components=3)
    embedder.run_layout(num_iterations=50)

    # Select influential nodes (fast, embedding-based)
    seeds = gr.graphem_seed_selection(embedder, k=10)

    # Evaluate influence spread
    G = nx.from_scipy_sparse_array(adjacency)
    influence, _ = gr.ndlib_estimated_influence(G, seeds, p=0.1, iterations_count=100)
    print(f"Influence: {influence:.1f} nodes ({influence/len(G)*100:.1f}%)")

    # Compare with greedy (slow, optimal)
    greedy_seeds, _ = gr.greedy_seed_selection(G, k=10, p=0.1)
    greedy_influence, _ = gr.ndlib_estimated_influence(G, greedy_seeds, p=0.1, iterations_count=100)
    print(f"Greedy influence: {greedy_influence:.1f} nodes ({greedy_influence/len(G)*100:.1f}%)")

Large-Scale Graph with cuVS
----------------------------

::

    import graphem_rapids as gr

    # Generate large graph
    adjacency = gr.generate_er(n=200000, p=0.00005, seed=42)

    # Force cuVS backend for large-scale
    embedder = gr.GraphEmbedderCuVS(
        adjacency,
        n_components=2,
        index_type='ivf_flat',
        sample_size=None,
        batch_size=4096,
        verbose=True
    )

    # Run layout
    embedder.run_layout(num_iterations=30)
    positions = embedder.get_positions()

Memory Management
-----------------

::

    import graphem_rapids as gr
    from graphem_rapids.utils.memory_management import MemoryManager, get_gpu_memory_info

    # Check GPU memory
    mem_info = get_gpu_memory_info()
    print(f"GPU: {mem_info['free']:.1f}GB free / {mem_info['total']:.1f}GB total")

    # Automatic cleanup
    with MemoryManager(cleanup_on_exit=True):
        adjacency = gr.generate_er(n=50000, p=0.0001)
        embedder = gr.create_graphem(adjacency, n_components=3)
        embedder.run_layout(num_iterations=40)
        positions = embedder.get_positions()
        # GPU memory automatically cleaned up

Benchmarking Correlations
--------------------------

::

    import graphem_rapids as gr

    # Run correlation benchmark
    results = gr.benchmark_correlations(
        gr.generate_ba,
        {"n": 1000, "m": 3, "seed": 42},
        n_components=2,
        L_min=40,
        num_iterations=50
    )

    # Print correlations with centrality measures
    print("Spearman correlations with radial distance:")
    for measure, corr in results['correlations'].items():
        print(f"{measure:15s}: ρ={corr['rho']:6.3f}, p={corr['p']:.2e}")

Custom Parameters
-----------------

::

    import graphem_rapids as gr

    adjacency = gr.generate_ws(n=1000, k=6, p=0.3, seed=42)

    # Fine-tune layout parameters
    embedder = gr.GraphEmbedderPyTorch(
        adjacency,
        n_components=2,
        device='cuda',
        L_min=2.0,          # Longer minimum spring length
        k_attr=0.1,         # Weaker attraction
        k_inter=1.0,        # Stronger intersection repulsion
        n_neighbors=15,     # More neighbors for intersection detection
        sample_size=512,
        batch_size=1024,
        memory_efficient=True
    )

    embedder.run_layout(num_iterations=100)
    embedder.display_layout()

Bipartite Graph Visualization
------------------------------

::

    import graphem_rapids as gr
    import numpy as np

    # Generate bipartite graph
    adjacency = gr.generate_complete_bipartite_graph(n_top=30, n_bottom=50)

    # Compute 2D embedding
    embedder = gr.create_graphem(adjacency, n_components=2)
    embedder.run_layout(num_iterations=50)

    # Color by partition
    n_top, n_bottom = 30, 50
    colors = np.concatenate([np.zeros(n_top), np.ones(n_bottom)])
    embedder.display_layout(node_colors=colors, node_size=8)

Delaunay Triangulation
-----------------------

::

    import graphem_rapids as gr

    # Generate planar graph from Delaunay triangulation
    adjacency = gr.generate_delaunay_triangulation(n=200, seed=42)

    # Embed in 2D to preserve planarity
    embedder = gr.create_graphem(adjacency, n_components=2)
    embedder.run_layout(num_iterations=50)

    # Visualize
    embedder.display_layout(edge_width=0.5, node_size=3)
