Quick start
===========

GraphEm requires CUDA 12, Torch, CuPy, cupyx, and cuVS.  Production runs pin
``device="cuda"`` and fail if CUDA is unavailable.  The same Torch spectral
tensor function can be selected explicitly on CPU for correctness checks;
that selection emits a ``RuntimeWarning``.  Midpoint search and force
refinement run on the GPU.

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
       seed=0,
       device="cuda",
   )
   embedder.run_layout(num_iterations=30)
   farthest = embedder.get_top_k(50)
