#!/usr/bin/env python3
"""
Comprehensive test runner for GraphEm Rapids.

This script runs the full test suite including slow integration tests,
performance tests, and GPU tests (if available). It's designed to be run
manually for thorough validation before releases or major changes.

Usage:
    python scripts/run_comprehensive_tests.py
    python scripts/run_comprehensive_tests.py --include-gpu  # Include GPU tests
    python scripts/run_comprehensive_tests.py --parallel     # Run tests in parallel
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        _ = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"SUCCESS ({duration:.1f}s): {description}")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"FAILED ({duration:.1f}s): {description}")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    """Run comprehensive test suite for GraphEm Rapids."""
    parser = argparse.ArgumentParser(description="Run comprehensive GraphEm Rapids tests")
    parser.add_argument("--include-gpu", action="store_true",
                       help="Include GPU tests (requires CUDA)")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel using pytest-xdist")
    parser.add_argument("--coverage", action="store_true", default=True,
                       help="Generate coverage report (default: True)")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first test failure")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose test output")

    args = parser.parse_args()

    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print("GraphEm Rapids Comprehensive Test Suite")
    print(f"Project root: {project_root}")
    print(f"Include GPU tests: {args.include_gpu}")
    print(f"Parallel execution: {args.parallel}")
    print(f"Coverage reporting: {args.coverage}")

    # Build base pytest command
    pytest_args = ["pytest"]

    if args.verbose:
        pytest_args.append("-v")

    if args.parallel:
        pytest_args.extend(["-n", "auto"])

    if args.fail_fast:
        pytest_args.append("--maxfail=1")
    else:
        pytest_args.append("--maxfail=5")

    if args.coverage:
        pytest_args.extend([
            "--cov=graphem_rapids",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])

    # Test phases
    test_results = []

    # Phase 1: Fast unit tests
    print("\n" + "="*80)
    print("PHASE 1: Fast Unit Tests")
    print("="*80)

    fast_cmd = " ".join(pytest_args + ['-m "fast"'])
    success = run_command(fast_cmd, "Fast unit tests")
    test_results.append(("Fast Tests", success))

    # Phase 2: Integration tests
    print("\n" + "="*80)
    print("PHASE 2: Integration Tests")
    print("="*80)

    integration_marker = 'integration and not gpu' if not args.include_gpu else 'integration'
    integration_cmd = " ".join(pytest_args + [f'-m "{integration_marker}"'])
    success = run_command(integration_cmd, "Integration tests")
    test_results.append(("Integration Tests", success))

    # Phase 3: Slow tests
    print("\n" + "="*80)
    print("PHASE 3: Slow Tests")
    print("="*80)

    slow_marker = 'slow and not gpu' if not args.include_gpu else 'slow'
    slow_cmd = " ".join(pytest_args + [f'-m "{slow_marker}"'])
    success = run_command(slow_cmd, "Slow tests")
    test_results.append(("Slow Tests", success))

    # Phase 4: GPU tests (if requested and available)
    if args.include_gpu:
        print("\n" + "="*80)
        print("PHASE 4: GPU Tests")
        print("="*80)

        gpu_cmd = " ".join(pytest_args + ['-m "gpu"'])
        success = run_command(gpu_cmd, "GPU tests")
        test_results.append(("GPU Tests", success))

    # Phase 5: Backend-specific comprehensive tests
    print("\n" + "="*80)
    print("PHASE 5: Backend-Specific Tests")
    print("="*80)

    backend_tests = [
        ("tests/test_backend_selection.py", "Backend Selection"),
        ("tests/test_memory_management.py", "Memory Management"),
    ]

    for test_file, description in backend_tests:
        if os.path.exists(test_file):
            cmd = " ".join(pytest_args + [test_file])
            success = run_command(cmd, f"{description} tests")
            test_results.append((description, success))

    # Phase 6: Documentation and example tests
    print("\n" + "="*80)
    print("PHASE 6: Examples and Documentation")
    print("="*80)

    if os.path.exists("examples"):
        examples_cmd = "cd examples && python quick_start_rapids.py"
        success = run_command(examples_cmd, "Example scripts")
        test_results.append(("Examples", success))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total_tests = len(test_results)
    passed_tests = sum(1 for _, success in test_results if success)
    failed_tests = total_tests - passed_tests

    for test_name, success in test_results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")

    if args.coverage and passed_tests > 0:
        print("\nCoverage report generated:")
        print("  - HTML: htmlcov/index.html")
        print("  - XML: coverage.xml")

    # Final status
    if failed_tests == 0:
        print("\nAll tests passed! The codebase is ready for deployment.")
        return 0
    print(f"\n{failed_tests} test suite(s) failed. Please review and fix issues.")
    return 1


if __name__ == "__main__":
    sys.exit(main())