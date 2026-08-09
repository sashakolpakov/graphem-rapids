GraphEm Rapids Documentation
============================

High-performance graph embedding using PyTorch and RAPIDS cuVS. Force-directed layout with geometric intersection detection.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/arXiv-2506.07435-b31b1b.svg
   :target: https://arxiv.org/abs/2506.07435

Features
--------

* **Unified API**: SciPy sparse adjacency matrices or GPU-resident edge lists
* **Multiple Backends**: PyTorch and a CuPy/cuVS large-graph path
* **GPU Acceleration**: sparse initialization, bounded edge batching, and on-device top-k
* **Graph Generators**: Erdős-Rényi, scale-free, SBM, bipartite, Delaunay, and more
* **Influence Maximization**: embedding and degree-discount selection with repeated IC evaluation

Quick Start
-----------

Installation::

    pip install graphem-rapids              # PyTorch backend
    pip install graphem-rapids[cuda]        # + CUDA
    pip install graphem-rapids[rapids]      # + RAPIDS cuVS
    pip install graphem-rapids[all]         # Everything

Basic Usage::

    import graphem_rapids as gr

    # Generate graph
    adjacency = gr.generate_er(n=1000, p=0.01)

    # Create embedder (automatic backend selection)
    embedder = gr.create_graphem(adjacency, n_components=3)

    # Run layout
    embedder.run_layout(num_iterations=50)

    # Get positions and visualize
    positions = embedder.get_positions()
    embedder.display_layout()

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   api
   backends
   generators
   examples

Citation
--------

.. code-block:: bibtex

    @misc{kolpakov-rivin-2025fast,
      title={Fast Geometric Embedding for Node Influence Maximization},
      author={Kolpakov, Alexander and Rivin, Igor},
      year={2025},
      eprint={2506.07435},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2506.07435}
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
