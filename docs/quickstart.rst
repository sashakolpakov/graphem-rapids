Quick start
===========

Requirements
------------

The canonical qualified environment uses Python 3.11, CUDA 12.9, Torch 2.11,
CuPy 14.1.1, and cuVS 26.06.  Install the CUDA-matched Torch wheel before the
package:

.. code-block:: console

   $ python3.11 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install --upgrade pip
   $ python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
   $ python -m pip install -e .

Device policy
-------------

Production and benchmark runs pin ``device="cuda"``.  An explicit CUDA request
fails if Torch CUDA is unavailable and never downgrades.  ``device="cpu"`` and
an automatic CPU selection use the same Torch spectral tensor function, emit a
``RuntimeWarning``, and record the selection reason.  They are small-fixture
spectral diagnostics only: midpoint search and force refinement still require
CuPy, cupyx, cuVS, and CUDA.

Adjacency input
---------------

.. code-block:: python

   import graphem_rapids as gr

   adjacency = gr.generate_er(n=1_000, p=0.1, seed=0)
   embedder = gr.GraphEmbedder(
       adjacency=adjacency,
       n_components=3,
       L_min=40.0,
       k_attr=1.0,
       k_inter=1.0,
       n_neighbors=15,
       sample_size=2_048,
       midpoint_query_batch_size=64,
       seed=0,
       device="cuda",
   )
   embedder.run_layout(num_iterations=30)

   positions = embedder.get_positions()
   scores = embedder.get_scores()
   farthest = embedder.get_top_k(50)
   diagnostics = embedder.get_diagnostics()

Edge-list input
---------------

Supply exactly one of ``adjacency`` and ``edges``.  An edge list also requires
the total vertex count:

.. code-block:: python

   embedder = gr.GraphEmbedder(
       edges=edge_array,
       n_vertices=vertex_count,
       n_components=3,
       n_neighbors=15,
       sample_size=2_048,
       midpoint_query_batch_size=64,
       device="cuda",
   )

Edges must contain integer vertex IDs in ``[0, n_vertices)`` and must not
contain loops or duplicate undirected pairs.  For both input forms,
``sample_size`` cannot exceed the edge count and ``n_neighbors`` must be
strictly smaller than the edge count.  Spectral initialization additionally
requires ``n_vertices >= 3 * (n_components + 1)``.  Exact midpoint queries are
submitted in batches of at most 64.  ``midpoint_query_batch_size`` can select a
smaller positive batch but cannot exceed that canonical bound.

Failure and diagnostics
-----------------------

Construction fails on a missing GPU dependency, invalid graph, unavailable
requested CUDA device, nonconverged spectral solve, or failed numerical gate.
``get_diagnostics()`` records the requested and selected spectral devices,
solver protocol, eigenvalues, residual, query-edge hashes, midpoint-search
receipts, iteration count, and primitive timings.  The midpoint receipt includes
the configured/effective batch size, hard policy bound, search-call count,
submitted batch-size histogram, call-width and resolved-query-width
histograms, and the highest device-wide bytes observed at declared CUDA memory
checkpoints.  Benchmark qualification also records an external GPU-memory
high-water mark.
