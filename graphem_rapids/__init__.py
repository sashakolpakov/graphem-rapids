"""GraphEm RAPIDS public API."""

from .embedder import GraphEmbedder
from .generators import (
    generate_ba,
    generate_balanced_tree,
    generate_bipartite_graph,
    generate_caveman,
    generate_complete_bipartite_graph,
    generate_delaunay_triangulation,
    generate_er,
    generate_geometric,
    generate_power_cluster,
    generate_random_regular,
    generate_relaxed_caveman,
    generate_road_network,
    generate_scale_free,
    generate_sbm,
    generate_ws,
)
from .datasets import load_dataset


__version__ = "0.3.0.dev0"

__all__ = [
    "GraphEmbedder",
    "generate_ba",
    "generate_balanced_tree",
    "generate_bipartite_graph",
    "generate_caveman",
    "generate_complete_bipartite_graph",
    "generate_delaunay_triangulation",
    "generate_er",
    "generate_geometric",
    "generate_power_cluster",
    "generate_random_regular",
    "generate_relaxed_caveman",
    "generate_road_network",
    "generate_scale_free",
    "generate_sbm",
    "generate_ws",
    "load_dataset",
]
