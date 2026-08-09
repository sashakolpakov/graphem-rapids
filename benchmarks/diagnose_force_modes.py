#!/usr/bin/env python3
"""Small CPU diagnostic for the paper/legacy spring-sign discrepancy."""

import argparse
import json
import time

import networkx as nx
import numpy as np
from scipy import stats

from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.influence import (
    degree_discount_seed_selection,
    estimate_independent_cascade,
)


def graph_cases(n_vertices, seed):
    return {
        "erdos_renyi": nx.fast_gnp_random_graph(n_vertices, 6.0 / n_vertices, seed=seed),
        "barabasi_albert": nx.barabasi_albert_graph(n_vertices, 3, seed=seed),
        "watts_strogatz": nx.watts_strogatz_graph(n_vertices, 6, 0.1, seed=seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cascade-trials", type=int, default=128)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    for graph_name, graph in graph_cases(args.vertices, args.seed).items():
        adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float32, format="csr")
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        pagerank = np.fromiter(nx.pagerank(graph).values(), dtype=np.float64)
        degree_nodes = np.argpartition(degree, -args.seed_count)[-args.seed_count:].tolist()
        pagerank_nodes = np.argpartition(
            pagerank, -args.seed_count
        )[-args.seed_count:].tolist()
        degree_spread = estimate_independent_cascade(
            adjacency,
            degree_nodes,
            p=0.1,
            n_simulations=args.cascade_trials,
            random_seed=args.seed,
            backend="cpu",
        )
        discount_nodes = degree_discount_seed_selection(
            adjacency, args.seed_count, p=0.1
        )
        discount_spread = estimate_independent_cascade(
            adjacency,
            discount_nodes,
            p=0.1,
            n_simulations=args.cascade_trials,
            random_seed=args.seed,
            backend="cpu",
        )
        pagerank_spread = estimate_independent_cascade(
            adjacency,
            pagerank_nodes,
            p=0.1,
            n_simulations=args.cascade_trials,
            random_seed=args.seed,
            backend="cpu",
        )

        for force_mode in ("legacy", "attractive"):
            started = time.perf_counter()
            embedder = GraphEmbedderPyTorch(
                adjacency,
                n_components=2,
                device="cpu",
                force_mode=force_mode,
                sample_size=256,
                n_neighbors=10,
                verbose=False,
                seed=args.seed,
            )
            embedder.run_layout(args.iterations)
            elapsed = time.perf_counter() - started
            scores = embedder.get_scores()
            nodes = embedder.topk_nodes(args.seed_count)
            diverse_nodes = embedder.diverse_topk_nodes(
                args.seed_count, diversity=0.2
            )
            spread = estimate_independent_cascade(
                adjacency,
                nodes,
                p=0.1,
                n_simulations=args.cascade_trials,
                random_seed=args.seed,
                backend="cpu",
            )
            diverse_spread = estimate_independent_cascade(
                adjacency,
                diverse_nodes,
                p=0.1,
                n_simulations=args.cascade_trials,
                random_seed=args.seed,
                backend="cpu",
            )
            records.append(
                {
                    "graph": graph_name,
                    "vertices": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "force_mode": force_mode,
                    "seconds": elapsed,
                    "degree_spearman": float(stats.spearmanr(scores, degree).statistic),
                    "pagerank_spearman": float(stats.spearmanr(scores, pagerank).statistic),
                    "cascade_mean": spread.mean,
                    "cascade_stderr": spread.stderr,
                    "diverse_cascade_mean": diverse_spread.mean,
                    "diverse_cascade_stderr": diverse_spread.stderr,
                    "degree_cascade_mean": degree_spread.mean,
                    "degree_cascade_stderr": degree_spread.stderr,
                    "degree_discount_cascade_mean": discount_spread.mean,
                    "degree_discount_cascade_stderr": discount_spread.stderr,
                    "pagerank_cascade_mean": pagerank_spread.mean,
                    "pagerank_cascade_stderr": pagerank_spread.stderr,
                }
            )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
