API reference
=============

GraphEmbedder
-------------

``GraphEmbedder`` accepts exactly one undirected simple graph through either a
SciPy-compatible adjacency matrix or a CUDA/array-compatible edge list.  The
``device`` keyword controls the shared Torch spectral function; the remaining
layout path is always CUDA/RAPIDS.  Production calls should pass
``device="cuda"`` explicitly.

.. autoclass:: graphem_rapids.GraphEmbedder
   :members:
   :member-order: bysource

Graph generators
----------------

Generators return symmetric SciPy CSR adjacency matrices.  ``GraphEmbedder``
then enforces its binary simple-graph contract; see :doc:`generators` for the
scale-free multiedge caveat.

.. automodule:: graphem_rapids.generators
   :members:
   :member-order: bysource
