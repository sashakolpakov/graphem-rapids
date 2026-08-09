"""Regression tests for the force convention documented by the paper."""

import torch

from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch


def _two_node_force(mode):
    embedder = object.__new__(GraphEmbedderPyTorch)
    embedder.k_attr = 1.0
    embedder.L_min = 1.0
    embedder.force_mode = mode
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    return embedder._compute_spring_forces(positions, edges)


def test_attractive_spring_shortens_a_stretched_edge():
    forces = _two_node_force("attractive")
    assert forces[0, 0] > 0
    assert forces[1, 0] < 0


def test_legacy_spring_lengthens_a_stretched_edge():
    forces = _two_node_force("legacy")
    assert forces[0, 0] < 0
    assert forces[1, 0] > 0


def test_diverse_selection_avoids_a_near_duplicate():
    embedder = object.__new__(GraphEmbedderPyTorch)
    embedder.n = 3
    embedder.device = torch.device("cpu")
    embedder._positions = torch.tensor([[10.0, 0.0], [9.0, 0.0], [-8.0, 0.0]])

    assert embedder.topk_nodes(2) == [0, 1]
    assert embedder.diverse_topk_nodes(2, diversity=0.8, candidate_pool_size=3) == [0, 2]
