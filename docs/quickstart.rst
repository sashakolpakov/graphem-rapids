Quick Start Guide
=================

Installation
------------

Basic Installation::

    pip install graphem-rapids

With CUDA support::

    pip install graphem-rapids[cuda]

With RAPIDS cuVS::

    pip install graphem-rapids[rapids]

All features::

    pip install graphem-rapids[all]

Basic Usage
-----------

Generate and Embed a Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import graphem_rapids as gr

    # Generate graph (returns sparse adjacency matrix)
    adjacency = gr.generate_er(n=1000, p=0.01, seed=42)

    # Create embedder with automatic backend selection
    embedder = gr.create_graphem(adjacency, n_components=3)

    # Run force-directed layout
    embedder.run_layout(num_iterations=50)

    # Get positions (numpy array)
    positions = embedder.get_positions()  # shape: (n_vertices, n_components)

    # Visualize (2D or 3D)
    embedder.display_layout()

Backend Selection
-----------------

Automatic (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

::

    embedder = gr.create_graphem(adjacency, n_components=3)

PyTorch Backend
~~~~~~~~~~~~~~~

Best for 1K-100K vertices::

    embedder = gr.GraphEmbedderPyTorch(
        adjacency, n_components=3, device='cuda',
        L_min=1.0, k_attr=0.2, k_inter=0.5,
        n_neighbors=10, batch_size=None
    )

RAPIDS cuVS Backend
~~~~~~~~~~~~~~~~~~~

Best for 100K+ vertices::

    embedder = gr.GraphEmbedderCuVS(
        adjacency, n_components=3,
        index_type='auto',  # 'brute_force', 'ivf_flat', 'ivf_pq'
        sample_size=1024, batch_size=None
    )

**Index Types:**
  * ``brute_force``: <100K vertices (exact KNN)
  * ``ivf_flat``: 100K-1M vertices (good balance)
  * ``ivf_pq``: >1M vertices (memory-efficient)

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

::

    export GRAPHEM_BACKEND=pytorch        # Force specific backend
    export GRAPHEM_PREFER_GPU=true        # Prefer GPU backends
    export GRAPHEM_MEMORY_LIMIT=8         # Memory limit in GB
    export GRAPHEM_VERBOSE=true           # Enable verbose logging

Programmatic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from graphem_rapids.utils.backend_selection import BackendConfig, get_optimal_backend

    config = BackendConfig(
        n_vertices=50000,
        force_backend='cuvs',
        memory_limit=16.0,
        prefer_gpu=True
    )
    backend = get_optimal_backend(config)
    embedder = gr.create_graphem(adjacency, backend=backend)

Check Available Backends
~~~~~~~~~~~~~~~~~~~~~~~~

::

    info = gr.get_backend_info()
    print(f"CUDA: {info['cuda_available']}")
    print(f"Recommended: {info['recommended_backend']}")
