#!/usr/bin/env python3
"""
Quick start example for GraphEm Rapids.

This example demonstrates the basic usage of GraphEm Rapids with
automatic backend selection and multiple graph types.
"""

import sys
import os
import time

# Add the parent directory to the path so we can import graphem_rapids
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import graphem_rapids as gr
    print("GraphEm Rapids imported successfully!")
except ImportError as e:
    print(f"Failed to import GraphEm Rapids: {e}")
    print("Please install the required dependencies.")
    sys.exit(1)


def main():
    """Run the quick start example."""
    print("=" * 60)
    print("GraphEm Rapids Quick Start Example")
    print("=" * 60)

    # Check backend availability
    print("\n1. Checking backend availability...")
    backend_info = gr.get_backend_info()
    for key, value in backend_info.items():
        print(f"   {key}: {value}")

    # Generate a small test graph
    print("\n2. Generating test graph...")
    n_vertices = 100
    edges = gr.erdos_renyi_graph(n=n_vertices, p=0.05)
    print(f"   Generated graph: {n_vertices} vertices, {len(edges)} edges")

    # Test automatic backend selection
    print("\n3. Testing automatic backend selection...")
    try:
        start_time = time.time()
        embedder = gr.create_graphem(
            edges=edges,
            n_vertices=n_vertices,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=min(1024, n_vertices),
            verbose=True
        )
        init_time = time.time() - start_time

        print(f"   Backend selected: {type(embedder).__name__}")
        print(f"   Initialization time: {init_time:.3f}s")

        # Run layout
        print("\n4. Running layout algorithm...")
        start_time = time.time()
        positions = embedder.run_layout(num_iterations=20)
        layout_time = time.time() - start_time

        print(f"   Layout completed in {layout_time:.3f}s")
        print(f"   Final positions shape: {positions.shape}")

        # Test influence maximization
        print("\n5. Testing influence maximization...")
        start_time = time.time()
        seeds = gr.graphem_seed_selection(embedder, k=5)
        influence_time = time.time() - start_time

        print(f"   Selected {len(seeds)} influential nodes in {influence_time:.3f}s")
        print(f"   Influential nodes: {seeds}")

        print("\n6. Success! GraphEm Rapids is working correctly.")

    except Exception as e:
        print(f"   Error during computation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test different graph sizes if backends are available
    if backend_info['torch_available']:
        print("\n7. Testing different graph sizes...")
        test_sizes = [50, 200, 500]

        for size in test_sizes:
            print(f"\n   Testing {size} vertices...")
            try:
                edges = gr.erdos_renyi_graph(n=size, p=0.02)
                start_time = time.time()

                embedder = gr.create_graphem(
                    edges=edges,
                    n_vertices=size,
                    dimension=3,
                    L_min=10.0,
                    k_attr=0.5,
                    k_inter=0.1,
                    knn_k=15,
                    sample_size=min(256, len(edges)),
                    batch_size=min(1024, size),
                    verbose=False
                )

                embedder.run_layout(num_iterations=10)
                total_time = time.time() - start_time

                print(f"      Completed in {total_time:.3f}s")
                print(f"      Backend: {type(embedder).__name__}")

            except Exception as e:
                print(f"      Failed: {e}")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("✅ GraphEm Rapids quick start completed successfully!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ GraphEm Rapids quick start failed!")
        print("=" * 60)
        sys.exit(1)