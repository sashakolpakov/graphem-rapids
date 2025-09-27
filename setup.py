#!/usr/bin/env python
"""
Setup script for GraphEm Rapids package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Core requirements (PyTorch backend)
required = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "networkx>=2.6.0",
    "pandas>=1.3.0",
    "plotly>=5.5.0",
    "scipy>=1.7.0",
    "ndlib>=5.1.0",
    "loguru>=0.6.0",
    "requests>=2.25.0",
    "line_profiler>=4.0.0",
    "snakeviz>=2.2.0",
    "tensorboard>=2.10.0",
    "tqdm>=4.66.0",
    "pyinstrument>=5.0.0",
    "tabulate>=0.9.0"
]

# CUDA acceleration (optional)
cuda_required = [
    "cupy-cuda12x>=10.0.0",
    "pykeops>=2.1.0"
]

# RAPIDS acceleration (optional)
rapids_required = [
    "cudf-cu12",
    "cuml-cu12",
    "cuvs-cu12",
    "cupy-cuda12x>=10.0.0",
    "pykeops>=2.1.0"
]

# Documentation requirements
docs_required = [
    "sphinx>=4.0.0",
    "sphinx_rtd_theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.12.0",
    "myst-parser>=0.17.0"
]

# Testing requirements
test_required = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "pytest-xdist>=2.3.0"
]

# Development requirements
dev_required = test_required + docs_required + [
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950"
]

setup(
    name="graphem-rapids",
    version="0.1.0",
    description="A high-performance graph embedding library with PyTorch and RAPIDS acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Kolpakov (UATX), Igor Rivin (Temple University)",
    author_email="akolpakov@uaustin.org, rivin@temple.edu",
    url="https://github.com/sashakolpakov/graphem-rapids",
    project_urls={
        "Documentation": "https://sashakolpakov.github.io/graphem-rapids/",
        "Source": "https://github.com/sashakolpakov/graphem-rapids",
        "Bug Reports": "https://github.com/sashakolpakov/graphem-rapids/issues",
        "Paper": "https://arxiv.org/abs/2506.07435",
        "Original GraphEm": "https://github.com/sashakolpakov/graphem"
    },
    packages=find_packages(exclude=["tests", "docs", "examples", "benchmarks"]),
    install_requires=required,
    extras_require={
        "cuda": cuda_required,
        "rapids": rapids_required,
        "docs": docs_required,
        "test": test_required,
        "dev": dev_required,
        "all": rapids_required + docs_required + test_required,
    },
    python_requires=">=3.8",
    keywords=[
        "graph embedding", "node influence", "centrality measures", "network analysis",
        "force layout", "PyTorch", "CUDA", "RAPIDS", "cuVS", "GPU acceleration"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "graphem-rapids-info=graphem_rapids.utils.backend_selection:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)