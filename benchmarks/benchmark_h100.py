#!/usr/bin/env python3
"""End-to-end large-graph benchmark for an H100-class CUDA GPU.

The benchmark deliberately includes graph generation, sparse construction,
initialization, layout, seed selection, and repeated Independent Cascade
evaluation. It emits one JSON record per graph size so interrupted scale sweeps
retain completed results.
"""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy import stats


def _parse_sizes(value):
    sizes = []
    for item in value.split(","):
        vertices, edges = item.split(":")
        sizes.append((int(vertices.replace("_", "")), int(edges.replace("_", ""))))
    return sizes


def _timed_gpu(cp, function, *args, **kwargs):
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    cp.cuda.get_current_stream().synchronize()
    return result, time.perf_counter() - started


class GpuMemorySampler:
    """Poll device allocation without requiring NVML."""

    def __init__(self, cp, interval=0.02):
        self.cp = cp
        self.interval = interval
        self.device_id = cp.cuda.Device().id
        self.maximum_used = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        self.cp.cuda.Device(self.device_id).use()
        while not self._stop.is_set():
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.maximum_used = max(self.maximum_used, int(total - free))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        free, total = self.cp.cuda.runtime.memGetInfo()
        self.maximum_used = max(self.maximum_used, int(total - free))


def generate_rmat(cp, n_vertices, n_edges, seed):
    """Generate canonical undirected R-MAT edges entirely on the GPU."""
    cp.random.seed(seed)
    n_bits = int(np.ceil(np.log2(max(n_vertices, 2))))
    source = cp.zeros(n_edges, dtype=cp.int64)
    target = cp.zeros(n_edges, dtype=cp.int64)
    for bit in range(n_bits):
        draws = cp.random.random(n_edges, dtype=cp.float32)
        source |= ((draws >= 0.76).astype(cp.int64) << bit)
        target |= (
            (((draws >= 0.57) & (draws < 0.76)) | (draws >= 0.95)).astype(cp.int64)
            << bit
        )
    valid = (source < n_vertices) & (target < n_vertices) & (source != target)
    source, target = source[valid], target[valid]
    lower = cp.minimum(source, target)
    upper = cp.maximum(source, target)
    keys = cp.unique(lower * np.int64(n_vertices) + upper)
    edge_dtype = cp.int32 if n_vertices < np.iinfo(np.int32).max else cp.int64
    return cp.column_stack((keys // n_vertices, keys % n_vertices)).astype(edge_dtype)


def make_undirected_csr(cp, cpx_sparse, edges, n_vertices):
    source, target = edges[:, 0], edges[:, 1]
    rows = cp.concatenate((source, target))
    columns = cp.concatenate((target, source))
    data = cp.ones(rows.shape[0], dtype=cp.float32)
    adjacency = cpx_sparse.csr_matrix(
        (data, (rows, columns)), shape=(n_vertices, n_vertices), dtype=cp.float32
    )
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    return adjacency


def topk(cp, scores, k):
    candidates = cp.argpartition(scores, scores.size - k)[-k:]
    return candidates[cp.argsort(scores[candidates])[::-1]]


def pagerank(cp, adjacency, iterations=30, damping=0.85):
    n_vertices = adjacency.shape[0]
    degree = cp.diff(adjacency.indptr).astype(cp.float32)
    rank = cp.full(n_vertices, 1.0 / n_vertices, dtype=cp.float32)
    dangling = degree == 0
    safe_degree = cp.where(dangling, 1.0, degree)
    for _ in range(iterations):
        contribution = rank / safe_degree
        rank = damping * (adjacency.T @ contribution)
        rank += (1.0 - damping + damping * cp.sum(contribution[dangling])) / n_vertices
    return rank


def optional_cugraph_baselines(cp, edges, n_vertices, degree, k, force_iterations):
    """Run k-core and standard ForceAtlas2 radius when cuGraph is installed."""
    import cudf  # pylint: disable=import-outside-toplevel
    import cugraph  # pylint: disable=import-outside-toplevel

    edge_frame = cudf.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
    graph = cugraph.Graph(directed=False)
    _, graph_seconds = _timed_gpu(
        cp,
        graph.from_cudf_edgelist,
        edge_frame,
        source="source",
        destination="target",
        renumber=False,
    )

    core_frame, core_seconds = _timed_gpu(cp, cugraph.core_number, graph)
    core_vertices = cp.asarray(core_frame["vertex"].values)
    core_values = cp.asarray(core_frame["core_number"].values, dtype=cp.float32)
    core_scores = cp.full(n_vertices, -1.0, dtype=cp.float32)
    core_scores[core_vertices] = core_values
    # Degree resolves the often very large equal-core ties deterministically.
    combined_core = core_scores + degree / (cp.max(degree) + 1.0) * 1e-3
    core_nodes, core_topk_seconds = _timed_gpu(cp, topk, cp, combined_core, k)

    layout_frame, force_seconds = _timed_gpu(
        cp,
        cugraph.force_atlas2,
        graph,
        max_iter=force_iterations,
        barnes_hut_optimize=True,
        verbose=False,
    )
    layout_vertices = cp.asarray(layout_frame["vertex"].values)
    layout_scores = cp.full(n_vertices, -cp.inf, dtype=cp.float32)
    x_values = cp.asarray(layout_frame["x"].values, dtype=cp.float32)
    y_values = cp.asarray(layout_frame["y"].values, dtype=cp.float32)
    layout_scores[layout_vertices] = cp.sqrt(x_values * x_values + y_values * y_values)
    force_nodes, force_topk_seconds = _timed_gpu(cp, topk, cp, layout_scores, k)

    del graph, edge_frame, core_frame, layout_frame
    return {
        "methods": {
            "kcore": cp.asnumpy(core_nodes).tolist(),
            "forceatlas2_radius": cp.asnumpy(force_nodes).tolist(),
        },
        "scores": {
            "kcore": core_scores,
            "forceatlas2_radius": layout_scores,
        },
        "timings": {
            "graph_construction_seconds": graph_seconds,
            "kcore_seconds": core_seconds + core_topk_seconds,
            "forceatlas2_seconds": force_seconds + force_topk_seconds,
        },
    }


def sampled_spearman(cp, first, second, sample_size, seed):
    size = min(sample_size, first.size)
    cp.random.seed(seed)
    indices = cp.random.choice(first.size, size=size, replace=False)
    first_host = cp.asnumpy(first[indices])
    second_host = cp.asnumpy(second[indices])
    return float(stats.spearmanr(first_host, second_host).statistic)


def environment_record(cp, cuvs):
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "gpu": name,
        "gpu_memory_bytes": int(properties["totalGlobalMem"]),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "cupy": cp.__version__,
        "cuvs": getattr(cuvs, "__version__", "unknown"),
    }


def run_case(args, n_vertices, requested_edges):
    import cupy as cp  # pylint: disable=import-outside-toplevel
    import cupyx.scipy.sparse as cpx_sparse  # pylint: disable=import-outside-toplevel
    import cuvs  # pylint: disable=import-outside-toplevel

    from graphem_rapids.backends.embedder_cuvs import GraphEmbedderCuVS
    from graphem_rapids.influence import (
        degree_discount_seed_selection,
        estimate_independent_cascade,
    )

    record = {
        "environment": environment_record(cp, cuvs),
        "requested_vertices": n_vertices,
        "requested_edges": requested_edges,
        "seed_count": args.k,
        "cascade_probability": args.cascade_probability,
        "cascade_trials": args.cascade_trials,
    }
    with GpuMemorySampler(cp) as memory:
        edges, elapsed = _timed_gpu(
            cp, generate_rmat, cp, n_vertices, requested_edges, args.seed
        )
        record["graph_generation_seconds"] = elapsed
        record["vertices"] = n_vertices
        record["edges"] = int(edges.shape[0])

        adjacency, elapsed = _timed_gpu(
            cp, make_undirected_csr, cp, cpx_sparse, edges, n_vertices
        )
        record["csr_construction_seconds"] = elapsed

        degree, elapsed = _timed_gpu(
            cp, cp.bincount, edges.reshape(-1), minlength=n_vertices
        )
        degree = degree.astype(cp.float32)
        degree_nodes, topk_time = _timed_gpu(cp, topk, cp, degree, args.k)
        record["degree_selection_seconds"] = elapsed + topk_time

        pagerank_scores, elapsed = _timed_gpu(
            cp, pagerank, cp, adjacency, args.pagerank_iterations
        )
        pagerank_nodes, topk_time = _timed_gpu(cp, topk, cp, pagerank_scores, args.k)
        record["pagerank_seconds"] = elapsed + topk_time

        discount_nodes, elapsed = _timed_gpu(
            cp,
            degree_discount_seed_selection,
            adjacency,
            args.k,
            args.cascade_probability,
        )
        record["degree_discount_seconds"] = elapsed

        cp.random.seed(args.seed)
        random_nodes, elapsed = _timed_gpu(
            cp, cp.random.choice, n_vertices, size=args.k, replace=False
        )
        record["random_selection_seconds"] = elapsed

        methods = {
            "random": cp.asnumpy(random_nodes).tolist(),
            "degree": cp.asnumpy(degree_nodes).tolist(),
            "pagerank": cp.asnumpy(pagerank_nodes).tolist(),
            "degree_discount": discount_nodes,
        }
        record["rank_correlation"] = {
            "degree_pagerank": sampled_spearman(
                cp, degree, pagerank_scores, args.correlation_sample, args.seed
            )
        }

        if args.with_cugraph:
            try:
                cugraph_result = optional_cugraph_baselines(
                    cp,
                    edges,
                    n_vertices,
                    degree,
                    args.k,
                    args.forceatlas_iterations,
                )
                methods.update(cugraph_result["methods"])
                record["cugraph_baselines"] = cugraph_result["timings"]
                for name, scores in cugraph_result["scores"].items():
                    record["rank_correlation"][f"{name}_degree"] = sampled_spearman(
                        cp, scores, degree, args.correlation_sample, args.seed
                    )
                    record["rank_correlation"][f"{name}_pagerank"] = sampled_spearman(
                        cp, scores, pagerank_scores, args.correlation_sample, args.seed
                    )
                del cugraph_result
            except (ImportError, RuntimeError, ValueError) as exc:
                record["cugraph_baselines"] = {"error": repr(exc)}

        for force_mode in args.force_modes.split(","):
            embedder, construction_time = _timed_gpu(
                cp,
                GraphEmbedderCuVS,
                edges=edges,
                n_vertices=n_vertices,
                assume_canonical_edges=True,
                n_components=args.dimensions,
                initialization="randomized",
                spectral_iterations=args.spectral_iterations,
                force_mode=force_mode,
                L_min=args.preferred_length,
                k_attr=args.attraction,
                k_inter=args.intersection_repulsion,
                n_neighbors=args.neighbors,
                sample_size=None,
                intersection_interval=args.intersection_interval,
                edge_chunk_size=args.edge_chunk_size,
                learning_rate=args.learning_rate,
                max_displacement=args.max_displacement,
                profile=True,
                seed=args.seed,
                verbose=False,
            )
            _, layout_time = _timed_gpu(cp, embedder.run_layout, args.layout_iterations)
            graph_nodes, selection_time = _timed_gpu(cp, embedder.topk_nodes, args.k)
            diverse_nodes, diverse_selection_time = _timed_gpu(
                cp, embedder.diverse_topk_nodes, args.k, args.diversity
            )
            graph_scores = embedder.get_scores(as_numpy=False)
            record["rank_correlation"][f"graphem_{force_mode}_degree"] = sampled_spearman(
                cp, graph_scores, degree, args.correlation_sample, args.seed
            )
            record["rank_correlation"][f"graphem_{force_mode}_pagerank"] = sampled_spearman(
                cp, graph_scores, pagerank_scores, args.correlation_sample, args.seed
            )
            method_name = f"graphem_{force_mode}"
            methods[method_name] = graph_nodes
            methods[f"{method_name}_diverse"] = diverse_nodes
            record[method_name] = {
                "construction_seconds": construction_time,
                "layout_seconds": layout_time,
                "selection_seconds": selection_time,
                "diverse_selection_seconds": diverse_selection_time,
                "diagnostics": embedder.get_diagnostics(),
            }
            del embedder, graph_scores
            cp.get_default_memory_pool().free_all_blocks()

        record["influence"] = {}
        influence_samples = {}
        for method_name, nodes in methods.items():
            estimate, elapsed = _timed_gpu(
                cp,
                estimate_independent_cascade,
                adjacency,
                nodes,
                p=args.cascade_probability,
                n_simulations=args.cascade_trials,
                random_seed=args.seed,
                backend="cupy",
                batch_size=args.cascade_batch_size,
            )
            record["influence"][method_name] = {
                "mean": estimate.mean,
                "std": estimate.std,
                "stderr": estimate.stderr,
                "minimum": estimate.minimum,
                "maximum": estimate.maximum,
                "seconds": elapsed,
            }
            influence_samples[method_name] = np.asarray(estimate.samples, dtype=np.float64)

        degree_samples = influence_samples["degree"]
        record["paired_spread_difference_vs_degree"] = {}
        for method_name, samples in influence_samples.items():
            difference = samples - degree_samples
            standard_error = (
                float(np.std(difference, ddof=1) / np.sqrt(difference.size))
                if difference.size > 1 else 0.0
            )
            mean_difference = float(np.mean(difference))
            record["paired_spread_difference_vs_degree"][method_name] = {
                "mean": mean_difference,
                "stderr": standard_error,
                "ci95": [
                    mean_difference - 1.96 * standard_error,
                    mean_difference + 1.96 * standard_error,
                ],
            }

    record["peak_device_memory_bytes"] = memory.maximum_used
    return record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=_parse_sizes("1_000_000:10_000_000,10_000_000:100_000_000"),
        help="comma-separated vertices:edges pairs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--spectral-iterations", type=int, default=16)
    parser.add_argument("--layout-iterations", type=int, default=20)
    parser.add_argument("--force-modes", default="legacy,attractive")
    parser.add_argument("--preferred-length", type=float, default=1.0)
    parser.add_argument("--attraction", type=float, default=0.2)
    parser.add_argument("--intersection-repulsion", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--intersection-interval", type=int, default=5)
    parser.add_argument("--edge-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-displacement", type=float, default=1.0)
    parser.add_argument("--diversity", type=float, default=0.2)
    parser.add_argument("--pagerank-iterations", type=int, default=30)
    parser.add_argument("--with-cugraph", action="store_true")
    parser.add_argument("--forceatlas-iterations", type=int, default=20)
    parser.add_argument("--correlation-sample", type=int, default=200_000)
    parser.add_argument("--cascade-probability", type=float, default=0.01)
    parser.add_argument("--cascade-trials", type=int, default=128)
    parser.add_argument("--cascade-batch-size", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    destination = args.output.open("a", encoding="utf-8") if args.output else None
    try:
        for n_vertices, n_edges in args.sizes:
            record = run_case(args, n_vertices, n_edges)
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            if destination:
                destination.write(line + "\n")
                destination.flush()
    finally:
        if destination:
            destination.close()


if __name__ == "__main__":
    main()
