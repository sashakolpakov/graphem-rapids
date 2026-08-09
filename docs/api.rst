API Reference
=============

Core Functions
--------------

.. autofunction:: graphem_rapids.create_graphem

.. autofunction:: graphem_rapids.get_backend_info

Embedders
---------

PyTorch Backend
~~~~~~~~~~~~~~~

.. autoclass:: graphem_rapids.GraphEmbedderPyTorch
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

RAPIDS cuVS Backend
~~~~~~~~~~~~~~~~~~~

.. autoclass:: graphem_rapids.GraphEmbedderCuVS
   :members:
   :undoc-members:
   :show-inheritance:

Graph Generators
----------------

Random Graphs
~~~~~~~~~~~~~

.. autofunction:: graphem_rapids.generate_er

.. autofunction:: graphem_rapids.generate_random_regular

Scale-Free and Small-World
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphem_rapids.generate_ba

.. autofunction:: graphem_rapids.generate_ws

.. autofunction:: graphem_rapids.generate_scale_free

.. autofunction:: graphem_rapids.generate_power_cluster

Community Structures
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphem_rapids.generate_sbm

.. autofunction:: graphem_rapids.generate_caveman

.. autofunction:: graphem_rapids.generate_relaxed_caveman

Bipartite Graphs
~~~~~~~~~~~~~~~~

.. autofunction:: graphem_rapids.generate_bipartite_graph

.. autofunction:: graphem_rapids.generate_complete_bipartite_graph

Geometric Graphs
~~~~~~~~~~~~~~~~

.. autofunction:: graphem_rapids.generate_geometric

.. autofunction:: graphem_rapids.generate_delaunay_triangulation

.. autofunction:: graphem_rapids.generate_road_network

Tree Structures
~~~~~~~~~~~~~~~

.. autofunction:: graphem_rapids.generate_balanced_tree

Influence Maximization
----------------------

.. autofunction:: graphem_rapids.graphem_seed_selection

.. autofunction:: graphem_rapids.degree_discount_seed_selection

.. autofunction:: graphem_rapids.estimate_independent_cascade

.. autofunction:: graphem_rapids.ndlib_estimated_influence

.. autofunction:: graphem_rapids.greedy_seed_selection

Benchmarking
------------

.. autofunction:: graphem_rapids.benchmark_correlations

.. autofunction:: graphem_rapids.run_benchmark

.. autofunction:: graphem_rapids.run_influence_benchmark

Visualization
-------------

.. autofunction:: graphem_rapids.report_corr

.. autofunction:: graphem_rapids.report_full_correlation_matrix

.. autofunction:: graphem_rapids.plot_radial_vs_centrality

.. autofunction:: graphem_rapids.display_benchmark_results

Datasets
--------

.. autofunction:: graphem_rapids.load_dataset

Utilities
---------

Backend Selection
~~~~~~~~~~~~~~~~~

.. autoclass:: graphem_rapids.utils.backend_selection.BackendConfig
   :members:

.. autofunction:: graphem_rapids.utils.backend_selection.get_optimal_backend

Memory Management
~~~~~~~~~~~~~~~~~

.. autoclass:: graphem_rapids.utils.memory_management.MemoryManager
   :members:

.. autofunction:: graphem_rapids.utils.memory_management.get_gpu_memory_info

.. autofunction:: graphem_rapids.utils.memory_management.get_optimal_chunk_size
