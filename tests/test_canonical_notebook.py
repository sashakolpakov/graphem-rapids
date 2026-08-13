"""Structural contract for the output-free canonical CUDA notebook."""

# pylint: disable=missing-function-docstring

import ast
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "graphem_rapids_notebook.ipynb"
)

RETIRED_API_TOKENS = {
    "GraphEmbedderCuVS",
    "GraphEmbedderPyTorch",
    "create_graphem",
    "get_backend_info",
    "backend=",
    "force_mode",
    "initialization",
    "index_type",
    "learning_rate",
    "max_displacement",
    "intersection_interval",
    "midpoint_reference_size",
    "full_midpoint_index",
    "max_candidate_pairs",
    "ivf_n_probes",
    "ivf_flat",
    "ivf_pq",
    "topk_nodes",
    "diverse_topk_nodes",
    "candidate_pool_size",
    "graphem_seed_selection",
    "degree_discount_seed_selection",
    "estimate_independent_cascade",
}


def _load_notebook():
    raw_notebook = NOTEBOOK_PATH.read_text(encoding="utf-8")
    notebook = json.loads(raw_notebook)
    assert raw_notebook == json.dumps(
        notebook, ensure_ascii=False, indent=1
    ) + "\n"
    return notebook


def _source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _attribute_calls(tree, attribute):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def test_notebook_is_deterministic_output_free_and_python_valid():
    notebook = _load_notebook()
    assert set(notebook) == {"cells", "metadata", "nbformat", "nbformat_minor"}
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5

    cells = notebook["cells"]
    cell_ids = [cell["id"] for cell in cells]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(cell["metadata"] == {} for cell in cells)
    assert all(isinstance(cell["source"], list) for cell in cells)
    assert all(cell["cell_type"] in {"code", "markdown"} for cell in cells)
    assert all(
        set(cell) == {"cell_type", "id", "metadata", "source"}
        for cell in cells
        if cell["cell_type"] == "markdown"
    )

    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert set(cell) == {
            "cell_type",
            "execution_count",
            "id",
            "metadata",
            "outputs",
            "source",
        }
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile(_source(cell), f"{NOTEBOOK_PATH.name}:{cell['id']}", "exec")


def test_notebook_uses_only_the_canonical_embedder_interface():
    notebook = _load_notebook()
    code = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)

    for retired_token in RETIRED_API_TOKENS:
        assert retired_token not in code
    assert 'device="cpu"' not in code
    assert "torch.cuda.is_available()" in code

    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    assert any(
        alias.name == "graphem_rapids" and alias.asname == "gr"
        for node in imports
        for alias in node.names
    )

    constructors = _attribute_calls(tree, "GraphEmbedder")
    assert len(constructors) == 1
    constructor = constructors[0]
    assert isinstance(constructor.func.value, ast.Name)
    assert constructor.func.value.id == "gr"
    keywords = {keyword.arg: keyword.value for keyword in constructor.keywords}
    assert set(keywords) == {
        "adjacency",
        "n_components",
        "L_min",
        "k_attr",
        "k_inter",
        "n_neighbors",
        "sample_size",
        "midpoint_query_batch_size",
        "seed",
        "device",
        "verbose",
    }
    assert ast.literal_eval(keywords["device"]) == "cuda"
    assert ast.literal_eval(keywords["midpoint_query_batch_size"]) == 64

    layout_calls = _attribute_calls(tree, "run_layout")
    assert len(layout_calls) == 1
    assert not layout_calls[0].args
    assert [keyword.arg for keyword in layout_calls[0].keywords] == [
        "num_iterations"
    ]
    assert ast.literal_eval(layout_calls[0].keywords[0].value) == 30

    for method in (
        "get_positions",
        "get_scores",
        "get_top_k",
        "get_diagnostics",
    ):
        assert len(_attribute_calls(tree, method)) == 1


def test_notebook_states_hardware_and_evidence_boundaries():
    notebook = _load_notebook()
    prose = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "CUDA 12.9" in prose
    assert "NVIDIA H100 with 80 GB" in prose
    assert "There is no CPU fallback" in prose
    assert "not a benchmark or scientific record" in prose
    assert "Neither is a sealed qualification artifact" in prose
