GraphEm RAPIDS
==============

GraphEm RAPIDS provides one implementation of the corrected GraphEm algorithm:
a device-parameterized Torch normalized-Laplacian start followed by CuPy/cuVS
force refinement on CUDA.

Production runs use ``device="cuda"`` and fail if Torch CUDA is unavailable.
The device parameter applies only to the shared Torch spectral function;
``GraphEmbedder`` always requires the CUDA/RAPIDS stack for graph storage,
midpoint search, forces, and updates.  Diagnostic CPU spectral selection warns
and does not constitute a CPU layout backend.

The package validates graph invariants, spectral residuals, orthogonality,
midpoint identities, and finite layout state.  A failed check raises an error
rather than selecting another solver or algorithm.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   api
   generators
