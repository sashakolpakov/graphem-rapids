"""Small-graph quality measurements with explicit method status."""

from __future__ import annotations

import hashlib
import numbers
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats

from .embedder import GraphEmbedder, MIDPOINT_QUERY_BATCH_SIZE_BOUND


def _graph_fingerprint(adjacency):
    upper = sp.triu(sp.csr_matrix(adjacency), k=1, format="coo")
    edges = np.column_stack((upper.row, upper.col)).astype("<i8", copy=False)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    return {
        "vertices": int(adjacency.shape[0]),
        "edges": int(edges.shape[0]),
        "edge_sha256": hashlib.sha256(edges.tobytes(order="C")).hexdigest(),
    }


def _measure_method(name, compute, vertex_count):
    started = time.perf_counter()
    try:
        mapping = compute()
        values = np.empty(vertex_count, dtype=np.float64)
        for vertex in range(vertex_count):
            values[vertex] = mapping[vertex] if hasattr(mapping, "__getitem__") else mapping
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} produced non-finite values")
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "seconds": time.perf_counter() - started,
        }
    return {
        "status": "passed",
        "seconds": time.perf_counter() - started,
        "values": values,
        "value_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
    }


def _centralities(graph):
    vertex_count = graph.number_of_nodes()
    methods = {
        "degree": lambda: dict(graph.degree()),
        "betweenness": lambda: nx.betweenness_centrality(graph),
        "eigenvector": lambda: nx.eigenvector_centrality_numpy(graph),
        "pagerank": lambda: nx.pagerank(graph),
        "closeness": lambda: nx.closeness_centrality(graph),
        "load": lambda: nx.load_centrality(graph),
    }
    return {
        name: _measure_method(name, function, vertex_count)
        for name, function in methods.items()
    }


def _topk_overlap(first_scores, second_scores, count):
    first = set(np.lexsort((np.arange(len(first_scores)), -first_scores))[:count])
    second = set(np.lexsort((np.arange(len(second_scores)), -second_scores))[:count])
    return len(first & second)


def run_benchmark(
    graph_generator,
    graph_params,
    *,
    n_components=3,
    L_min=10.0,
    k_attr=0.5,
    k_inter=0.1,
    n_neighbors=15,
    sample_size=512,
    midpoint_query_batch_size=MIDPOINT_QUERY_BATCH_SIZE_BOUND,
    num_iterations=40,
    seed=0,
    top_k=50,
):
    """Measure GraphEm and each centrality without substituting failed methods."""
    total_started = time.perf_counter()
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, numbers.Integral):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    adjacency = sp.csr_matrix(graph_generator(**graph_params))
    graph = nx.from_scipy_sparse_array(adjacency)
    centralities = _centralities(graph)

    embedder = GraphEmbedder(
        adjacency=adjacency,
        n_components=n_components,
        L_min=L_min,
        k_attr=k_attr,
        k_inter=k_inter,
        n_neighbors=n_neighbors,
        sample_size=sample_size,
        midpoint_query_batch_size=midpoint_query_batch_size,
        seed=seed,
    )
    layout_started = time.perf_counter()
    embedder.run_layout(num_iterations=num_iterations)
    layout_seconds = time.perf_counter() - layout_started
    positions = embedder.get_positions()
    radii = np.linalg.norm(positions, axis=1)
    count = min(int(top_k), len(radii))

    for record in centralities.values():
        if record["status"] != "passed":
            continue
        values = record.pop("values")
        coefficient, p_value = stats.spearmanr(radii, values)
        record["radius_spearman"] = float(coefficient)
        record["radius_spearman_p"] = float(p_value)
        record["farthest_topk_overlap"] = _topk_overlap(radii, values, count)
        record["nearest_topk_overlap"] = _topk_overlap(-radii, values, count)

    return {
        "schema": "graphem-quality-v1",
        "claim_ready": False,
        "graph": _graph_fingerprint(adjacency),
        "configuration": {
            "n_components": n_components,
            "L_min": L_min,
            "k_attr": k_attr,
            "k_inter": k_inter,
            "n_neighbors": n_neighbors,
            "sample_size": sample_size,
            "midpoint_query_batch_size": midpoint_query_batch_size,
            "num_iterations": num_iterations,
            "seed": seed,
            "top_k": count,
        },
        "graphem": {
            "status": "passed",
            "layout_seconds": layout_seconds,
            "radius_sha256": hashlib.sha256(
                radii.astype("<f8").tobytes()
            ).hexdigest(),
            "diagnostics": embedder.get_diagnostics(),
        },
        "centralities": centralities,
        "total_seconds": time.perf_counter() - total_started,
    }


def benchmark_correlations(*args, **kwargs):
    """Return the same explicit-status record as :func:`run_benchmark`."""
    return run_benchmark(*args, **kwargs)
