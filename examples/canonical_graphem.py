"""Minimal canonical GraphEm run."""

import graphem_rapids as gr


def main():
    adjacency = gr.generate_er(n=1_000, p=0.1, seed=0)
    embedder = gr.GraphEmbedder(
        adjacency=adjacency,
        n_components=3,
        L_min=40.0,
        k_attr=1.0,
        k_inter=1.0,
        n_neighbors=15,
        sample_size=2_048,
        midpoint_query_batch_size=64,
        seed=0,
        device="cuda",
    )
    embedder.run_layout(num_iterations=30)
    print(embedder.get_top_k(50))


if __name__ == "__main__":
    main()
