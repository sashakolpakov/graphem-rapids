Graph Generators
================

All generators return symmetric ``scipy.sparse.csr_matrix`` adjacency matrices.
The canonical embedder separately validates that an input is unweighted,
loop-free, and duplicate-free before transferring graph data to CUDA.  Most
generators return a binary matrix directly; the general scale-free generator
retains NetworkX multiedge multiplicity and therefore needs the explicit
binarization shown below before use with ``GraphEmbedder``.

Random Graphs
-------------

Erdős-Rényi
~~~~~~~~~~~

::

    adjacency = gr.generate_er(n=1000, p=0.01, seed=42)

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

General Scale-Free
~~~~~~~~~~~~~~~~~~

::

    adjacency = gr.generate_scale_free(n=100, seed=42)
    adjacency.data[:] = 1

NetworkX scale-free model, converted to an undirected simple graph by dropping
edge direction, removing self-loops, and explicitly binarizing any retained
parallel-edge counts.

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

The current implementation does not forward ``seed`` to NetworkX's rewiring
routine.  Do not use this generator for a reproduction cell that requires a
seeded input until that source-level issue is corrected and qualified.

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

    # Compute the canonical CUDA embedding
    embedder = gr.GraphEmbedder(
        adjacency=adjacency,
        n_components=2,
        n_neighbors=15,
        sample_size=256,
        device="cuda",
    )
    embedder.run_layout(num_iterations=50)

    # Visualize with community colors
    adjacency_with_labels, labels = gr.generate_sbm(
        n_per_block=100,
        num_blocks=5,
        labels=True,
        seed=42
    )
    positions = embedder.get_positions()
    scores = embedder.get_scores()

``positions`` and ``labels`` share vertex order and can be passed to a plotting
library chosen by the caller.  ``GraphEmbedder`` does not provide a separate
visualization backend.
