"""Execution contract for the retained CPU-safe package notebook."""

import json
from pathlib import Path


def test_quickstart_notebook_executes_end_to_end(monkeypatch):
    """Run every code cell without requiring Jupyter or a CUDA device."""
    repository = Path(__file__).parents[1]
    notebook_path = repository / "examples" / "graphem_rapids_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("id") for cell in notebook["cells"])

    monkeypatch.chdir(repository)
    namespace = {"__name__": "__notebook_test__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(  # pylint: disable=exec-used
            compile(source, f"{notebook_path.name}:cell-{index}", "exec"),
            namespace,
        )

    assert namespace["positions"].shape == (80, 2)
    assert len(namespace["graph_seeds"]) == 5
    assert len(namespace["degree_discount_seeds"]) == 5
