Backend Selection
=================

GraphEm Rapids provides multiple computational backends optimized for different graph sizes.

Overview
--------

+------------------+-------------------+-------------------------+
| Backend          | Best For          | Features                |
+==================+===================+=========================+
| PyTorch          | 1K-100K vertices  | CUDA, memory-efficient  |
+------------------+-------------------+-------------------------+
| RAPIDS cuVS      | 100K+ vertices    | Optimized KNN, CuPy     |
+------------------+-------------------+-------------------------+

Automatic Selection
-------------------

The ``create_graphem()`` function automatically selects the optimal backend based on:

* Dataset size (number of vertices)
* Available hardware (CUDA, RAPIDS)
* Memory constraints
* User preferences

::

    import graphem_rapids as gr

    # Automatic selection (recommended)
    embedder = gr.create_graphem(adjacency, n_components=3)

PyTorch Backend
---------------

Best for medium-scale graphs (1K-100K vertices).

**Features:**

* CUDA acceleration with automatic CPU fallback
* Memory-efficient chunking
* Automatic batch size selection
* PyKeOps integration for KNN

**Usage:**

::

    embedder = gr.GraphEmbedderPyTorch(
        adjacency,
        n_components=3,
        device='cuda',           # 'cuda', 'cpu', or None (auto)
        L_min=1.0,               # Minimum spring length
        k_attr=0.2,              # Attraction constant
        k_inter=0.5,             # Intersection repulsion constant
        n_neighbors=10,          # KNN for intersection detection
        sample_size=256,         # Sample size for KNN
        batch_size=None,         # None (auto) or manual (e.g., 1024)
        memory_efficient=True,   # Enable memory optimizations
        verbose=True
    )

RAPIDS cuVS Backend
-------------------

Best for large-scale graphs (100K+ vertices).

**Features:**

* Optimized KNN with cuVS indices
* CuPy operations for GPU acceleration
* Multiple index types (brute force, IVF-Flat, IVF-PQ)
* Automatic index selection

**Usage:**

::

    embedder = gr.GraphEmbedderCuVS(
        adjacency,
        n_components=3,
        index_type='auto',       # 'auto', 'brute_force', 'ivf_flat', 'ivf_pq'
        L_min=1.0,
        k_attr=0.2,
        k_inter=0.5,
        n_neighbors=10,
        sample_size=1024,        # Larger samples for cuVS
        batch_size=None,
        verbose=True
    )

**Index Types:**

* ``brute_force``: Exact KNN, best for <100K vertices
* ``ivf_flat``: Good balance for 100K-1M vertices
* ``ivf_pq``: Memory-efficient for >1M vertices
* ``auto``: Automatic selection (recommended)

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

::

    export GRAPHEM_BACKEND=pytorch        # Force backend: 'pytorch', 'cuvs', 'auto'
    export GRAPHEM_PREFER_GPU=true        # Prefer GPU backends
    export GRAPHEM_MEMORY_LIMIT=8         # Memory limit in GB
    export GRAPHEM_VERBOSE=true           # Enable verbose logging
    export GRAPHEM_RAPIDS_QUIET=true      # Suppress RAPIDS startup messages

Programmatic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from graphem_rapids.utils.backend_selection import BackendConfig, get_optimal_backend

    config = BackendConfig(
        n_vertices=50000,
        n_components=3,
        force_backend='cuvs',    # Force specific backend
        memory_limit=16.0,       # GB
        prefer_gpu=True,
        verbose=True
    )

    backend = get_optimal_backend(config)
    embedder = gr.create_graphem(adjacency, backend=backend)

Check Backend Availability
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    info = gr.get_backend_info()
    print(f"CUDA available: {info['cuda_available']}")
    print(f"RAPIDS available: {info['rapids_available']}")
    print(f"Recommended: {info['recommended_backend']}")

Memory Management
-----------------

Automatic Batch Sizing
~~~~~~~~~~~~~~~~~~~~~~~

By default, batch sizes are automatically selected based on available memory::

    embedder = gr.GraphEmbedderPyTorch(adjacency, batch_size=None)

Manual Batch Sizing
~~~~~~~~~~~~~~~~~~~~

For fine-tuned control::

    embedder = gr.GraphEmbedderPyTorch(adjacency, batch_size=1024)

Programmatic Batch Sizing
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from graphem_rapids.utils.memory_management import get_optimal_chunk_size

    optimal = get_optimal_chunk_size(
        n_vertices=1000000,
        n_components=3,
        backend='pytorch'
    )
    embedder = gr.GraphEmbedderPyTorch(adjacency, batch_size=optimal)

GPU Memory Monitoring
~~~~~~~~~~~~~~~~~~~~~

::

    from graphem_rapids.utils.memory_management import get_gpu_memory_info

    mem_info = get_gpu_memory_info()
    print(f"Free: {mem_info['free']:.1f}GB / Total: {mem_info['total']:.1f}GB")
