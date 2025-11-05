Graph Generators
================

All generators return scipy sparse adjacency matrices in CSR format.

Random Graphs
-------------

Erdős-Rényi
~~~~~~~~~~~

::

    adjacency = gr.erdos_renyi_graph(n=1000, p=0.01, seed=42)

Random edges with probability *p*.

Random Regular
~~~~~~~~~~~~~~

::

    adjacency = gr.generate_random_regular(n=100, d=3, seed=42)

Every vertex has exactly degree *d*.

Scale-Free Graphs
-----------------

Barabási-Albert
~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_ba(n=300, m=3, seed=42)

Preferential attachment model.

Scale-Free
~~~~~~~~~~

::

    adjacency = gr.generate_scale_free(n=100, seed=42)

Holme-Kim algorithm with configurable parameters.

Power-Law Cluster
~~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_power_cluster(n=1000, m=3, p=0.5, seed=42)

Power-law degree distribution with triangle formation.

Small-World Graphs
------------------

Watts-Strogatz
~~~~~~~~~~~~~~

::

    adjacency = gr.generate_ws(n=1000, k=6, p=0.3, seed=42)

Ring lattice with random rewiring.

Community Structures
--------------------

Stochastic Block Model
~~~~~~~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_sbm(
        n_per_block=75,
        num_blocks=4,
        p_in=0.15,   # Within-block edge probability
        p_out=0.01,  # Between-block edge probability
        seed=42
    )

With labels::

    adjacency, labels = gr.generate_sbm(
        n_per_block=75,
        num_blocks=4,
        labels=True,
        seed=42
    )

Caveman Graph
~~~~~~~~~~~~~

::

    adjacency = gr.generate_caveman(l=10, k=10)

*l* cliques of size *k*.

Relaxed Caveman
~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_relaxed_caveman(l=10, k=10, p=0.1, seed=42)

Caveman graph with rewiring probability *p*.

Bipartite Graphs
----------------

Random Bipartite
~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_bipartite_graph(
        n_top=50,
        n_bottom=100,
        p=0.2,       # Edge probability
        seed=42
    )

Complete Bipartite
~~~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_complete_bipartite_graph(n_top=50, n_bottom=100)

Every vertex in top set connects to every vertex in bottom set (K_{n,m}).

Geometric Graphs
----------------

Random Geometric
~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_geometric(n=100, radius=0.2, dim=2, seed=42)

Vertices in unit cube, edges within distance *radius*.

Delaunay Triangulation
~~~~~~~~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_delaunay_triangulation(n=100, seed=42)

Planar graph from Delaunay triangulation of random 2D points.

Road Network
~~~~~~~~~~~~

::

    adjacency = gr.generate_road_network(width=30, height=30)

2D grid graph.

Tree Structures
---------------

Balanced Tree
~~~~~~~~~~~~~

::

    adjacency = gr.generate_balanced_tree(r=2, h=10)

*r*-ary tree of height *h*.

Complete Example
----------------

::

    import graphem_rapids as gr
    import networkx as nx

    # Generate graph
    adjacency = gr.generate_sbm(
        n_per_block=100,
        num_blocks=5,
        p_in=0.2,
        p_out=0.01,
        seed=42
    )

    # Compute embedding
    embedder = gr.create_graphem(adjacency, n_components=2)
    embedder.run_layout(num_iterations=50)

    # Visualize with community colors
    adjacency_with_labels, labels = gr.generate_sbm(
        n_per_block=100,
        num_blocks=5,
        labels=True,
        seed=42
    )
    embedder.display_layout(node_colors=labels)
