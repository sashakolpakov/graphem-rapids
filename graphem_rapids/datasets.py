"""
Real-world dataset loader for Graphem.

This module provides functions to download and load standard large graph datasets
from various sources including SNAP (Stanford Network Analysis Project),
Network Repository, and other public graph repositories.
"""

from pathlib import Path
import gzip
import shutil
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
from tqdm import tqdm


def get_data_directory():
    """
    Get the data directory for storing downloaded datasets.
    Creates the directory if it doesn't exist.
    
    Returns:
        Path: Path to the data directory
    """
    # Get the base directory (same as this file)
    base_dir = Path(__file__).parent.parent
    
    # Create data directory
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    return data_dir


def download_file(url, filepath, description=None):
    """
    Download a file with progress bar.
    
    Parameters:
        url: str
            URL to download
        filepath: Path or str
            Path to save the downloaded file
        description: str, optional
            Description for the progress bar
    """
    # Create directory if it doesn't exist
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    if filepath.exists():
        logger.info(f"File already exists: {filepath}")
        return
    
    logger.info(f"Downloading from {url} to {filepath}")
    
    # Make HTTP request
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    # Get file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    # Download with progress bar
    with open(filepath, 'wb') as f, tqdm(
            desc=description or "Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))


def extract_file(filepath, extract_dir=None):
    """
    Extract a compressed file.
    
    Parameters:
        filepath: Path or str
            Path to the compressed file
        extract_dir: Path or str, optional
            Directory to extract to. If None, extracts to the same directory as the file.
    
    Returns:
        Path: Path to the extraction directory
    """
    filepath = Path(filepath)
    
    if extract_dir is None:
        extract_dir = filepath.parent
    else:
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Extracting {filepath} to {extract_dir}")
    
    # Handle different compression formats
    if filepath.suffix == '.gz':
        # For single gzipped files like .txt.gz
        with gzip.open(filepath, 'rb') as f_in:
            output_path = extract_dir / filepath.stem
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif filepath.suffix == '.zip':
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif filepath.suffix in ('.tar', '.tgz'):
        with tarfile.open(filepath, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        logger.warning(f"Unknown compression format: {filepath.suffix}")
    
    return extract_dir


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, name):
        """
        Initialize the dataset loader.
        
        Parameters:
            name: str
                Name of the dataset
        """
        self.name = name
        self.data_dir = get_data_directory() / name
        self.data_dir.mkdir(exist_ok=True, parents=True)
    
    def download(self):
        """Download the dataset. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement download()")
    
    def load(self):
        """Load the dataset as edges. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load()")
    
    def load_as_networkx(self):
        """
        Load the dataset as a NetworkX graph.
        
        Returns:
            networkx.Graph: The loaded graph
        """
        edges, vertices = self.load()
        
        # Create a NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        G = nx.convert_node_labels_to_integers(G,
                                               first_label=0,
                                               ordering='default',
                                               label_attribute=None)
        
        return G
    
    def info(self):
        """
        Print information about the dataset.
        """
        if not self.is_downloaded():
            print(f"Dataset '{self.name}' is not downloaded yet.")
            return
        
        vertices, edges = self.load()
        n_vertices = len(vertices)
        n_edges = len(edges)
        
        print(f"Dataset: {self.name}")
        print(f"Number of vertices: {n_vertices}")
        print(f"Number of edges: {n_edges}")
        print(f"Density: {2 * n_edges / (n_vertices * (n_vertices - 1)):.6f}")
        print(f"Average degree: {2 * n_edges / n_vertices:.2f}")
    
    def is_downloaded(self):
        """
        Check if the dataset is already downloaded.
        
        Returns:
            bool: True if downloaded, False otherwise
        """
        # Subclasses should implement more specific checks
        return self.data_dir.exists()


class SNAPDataset(DatasetLoader):
    """
    Loader for datasets from the Stanford Network Analysis Project (SNAP).
    
    SNAP datasets are commonly used in network analysis research.
    Source: https://snap.stanford.edu/data/
    """
    
    AVAILABLE_DATASETS = {
        "facebook_combined": {
            "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
            "description": "Facebook social network",
            "directed": False,
            "nodes": 4039,
            "edges": 88234
        },
        "ego-twitter": {
            "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
            "description": "Twitter ego network",
            "directed": True,
            "nodes": 81306,
            "edges": 1768149
        },
        "wiki-vote": {
            "url": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
            "description": "Wikipedia who-votes-on-whom network",
            "directed": True,
            "nodes": 7115,
            "edges": 103689
        },
        "ca-GrQc": {
            "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
            "description": "Collaboration network of Arxiv General Relativity",
            "directed": False,
            "nodes": 5242,
            "edges": 14496
        },
        "ca-HepTh": {
            "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
            "description": "Collaboration network of Arxiv High Energy Physics Theory",
            "directed": False,
            "nodes": 9877,
            "edges": 25998
        },
        "oregon1_010331": {
            "url": "https://snap.stanford.edu/data/oregon1_010331.txt.gz",
            "description": "AS peering network from Oregon route views",
            "directed": False,
            "nodes": 10670,
            "edges": 22002
        },
        "p2p-Gnutella04": {
            "url": "https://snap.stanford.edu/data/p2p-Gnutella04.txt.gz",
            "description": "Gnutella peer-to-peer network from August 4, 2002",
            "directed": True,
            "nodes": 10876,
            "edges": 39994
        },
        "email-Enron": {
            "url": "https://snap.stanford.edu/data/email-Enron.txt.gz",
            "description": "Email communication network from Enron",
            "directed": True,
            "nodes": 36692,
            "edges": 183831
        }
    }
    
    def __init__(self, dataset_name):
        """
        Initialize the SNAP dataset loader.
        
        Parameters:
            dataset_name: str
                Name of the SNAP dataset to load
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown SNAP dataset: {dataset_name}. Available datasets: {', '.join(self.AVAILABLE_DATASETS.keys())}")
        
        self.dataset_info = self.AVAILABLE_DATASETS[dataset_name]
        super().__init__(f"snap-{dataset_name}")
        self.dataset_name = dataset_name
        self.url = self.dataset_info["url"]
        self.is_directed = self.dataset_info["directed"]
    
    def download(self):
        """
        Download the SNAP dataset.
        """
        if self.is_downloaded():
            logger.info(f"Dataset {self.dataset_name} already downloaded.")
            return
        
        # Download the compressed file
        filename = self.url.split("/")[-1]
        download_path = self.data_dir / filename
        download_file(self.url, download_path, f"Downloading {self.dataset_name}")
        
        # Extract the file
        extract_file(download_path, self.data_dir)
    
    def is_downloaded(self):
        """
        Check if the dataset is already downloaded.
        """
        # Look for extracted txt file
        filename = self.url.split("/")[-1]
        extracted_file = self.data_dir / filename.replace(".gz", "")
        return extracted_file.exists()
    
    def load(self):
        """
        Load the SNAP dataset as edges.
        
        Returns:
            tuple: (vertices, edges)
                vertices: np.ndarray of shape (num_vertices,)
                edges: np.ndarray of shape (num_edges, 2)
        """
        if not self.is_downloaded():
            self.download()
        
        # Load the edges file
        filename = self.url.split("/")[-1].replace(".gz", "")
        file_path = self.data_dir / filename
        
        edges = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments
                if line.startswith('#'):
                    continue
                
                # Parse edge
                values = line.strip().split()
                if len(values) >= 2:
                    source = int(values[0])
                    target = int(values[1])
                    edges.append((source, target))
        
        # Convert to numpy array
        edges = np.array(edges)
        
        # Make undirected if needed (ensure i < j for all edges)
        if not self.is_directed:
            # Create a reversed copy of edges with source and target swapped
            reversed_edges = edges[:, [1, 0]]
            
            # Combine original and reversed edges
            all_edges = np.vstack((edges, reversed_edges))
            
            # Remove duplicates and sort
            unique_edges = np.unique(all_edges, axis=0)
            
            # Keep only edges where source < target
            mask = unique_edges[:, 0] < unique_edges[:, 1]
            edges = unique_edges[mask]
        
        # Extract vertices from edges
        vertices = np.unique(edges.flatten())
        
        return vertices, edges


class NetworkRepositoryDataset(DatasetLoader):
    """
    Loader for datasets from the Network Repository.
    
    Network Repository is a scientific network data repository with interactive analytics.
    Source: https://networkrepository.com/
    """
    
    AVAILABLE_DATASETS = {
        "soc-hamsterster": {
            "url": "https://nrvis.com/download/data/soc/soc-hamsterster.zip",
            "description": "Hamsterster social network",
            "directed": False,
            "file_pattern": "soc-hamsterster.mtx"
        },
        "socfb-MIT": {
            "url": "https://nrvis.com/download/data/socfb/socfb-MIT.zip",
            "description": "Facebook network from MIT",
            "directed": False,
            "file_pattern": "socfb-MIT.mtx"
        },
        "ca-cit-HepPh": {
            "url": "https://nrvis.com/download/data/ca/ca-cit-HepPh.zip",
            "description": "Citation network of Arxiv High Energy Physics",
            "directed": True,
            "file_pattern": "ca-cit-HepPh.mtx"
        },
        "web-google-dir": {
            "url": "https://nrvis.com/download/data/web/web-google-dir.zip",
            "description": "Google web graph",
            "directed": True,
            "file_pattern": "web-google-dir.edges"
        },
        "ia-reality": {
            "url": "https://nrvis.com/download/data/ia/ia-reality.zip",
            "description": "Reality Mining social network",
            "directed": False,
            "file_pattern": "ia-reality.mtx"
        }
    }
    
    def __init__(self, dataset_name):
        """
        Initialize the Network Repository dataset loader.
        
        Parameters:
            dataset_name: str
                Name of the Network Repository dataset to load
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown Network Repository dataset: {dataset_name}. Available datasets: {', '.join(self.AVAILABLE_DATASETS.keys())}")
        
        self.dataset_info = self.AVAILABLE_DATASETS[dataset_name]
        super().__init__(f"netrepo-{dataset_name}")
        self.dataset_name = dataset_name
        self.url = self.dataset_info["url"]
        self.is_directed = self.dataset_info["directed"]
        self.file_pattern = self.dataset_info["file_pattern"]
    
    def download(self):
        """
        Download the Network Repository dataset.
        """
        if self.is_downloaded():
            logger.info(f"Dataset {self.dataset_name} already downloaded.")
            return
        
        # Download the compressed file
        filename = self.url.split("/")[-1]
        download_path = self.data_dir / filename
        download_file(self.url, download_path, f"Downloading {self.dataset_name}")
        
        # Extract the file
        extract_file(download_path, self.data_dir)

    def is_downloaded(self):
        """
        Check if the expected dataset file exists without scanning the whole tree.

        Returns:
            bool: True if the expected file exists, False otherwise
        """
        expected_path = self.data_dir / self.file_pattern
        return expected_path.exists()

    def _find_data_file(self):
        """
        Find the data file after extraction.

        Returns:
            Path: Path to the data file
        """
        matches = list(self.data_dir.glob(f"**/{self.file_pattern}"))
        if not matches:
            raise FileNotFoundError(
                f"Could not find data file matching pattern {self.file_pattern} in {self.data_dir}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple files matched {self.file_pattern} in {self.data_dir}: {matches}"
            )
        return matches[0]

    def load(self):
        """
        Load the Network Repository dataset as edges.
        
        Returns:
            tuple: (vertices, edges)
                vertices: np.ndarray of shape (num_vertices,)
                edges: np.ndarray of shape (num_edges, 2)
        """
        if not self.is_downloaded():
            self.download()
        
        # Find and load the data file
        file_path = self._find_data_file()
        
        # Different file formats have different parsing methods
        if file_path.suffix == '.mtx':
            return self._load_mtx_file(file_path)
        if file_path.suffix == '.edges':
            return self._load_edges_file(file_path)
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_mtx_file(self, file_path):
        """
        Load a Matrix Market (.mtx) file.
        
        Parameters:
            file_path: Path
                Path to the .mtx file
        
        Returns:
            tuple: (vertices, edges)
        """
        edges = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip comments
            for line in f:
                if not line.startswith('%'):
                    # First non-comment line has matrix dimensions
                    # We need the edge data only, so we skip it
                    break
            
            # Read edge data
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    source = int(parts[0]) - 1  # MTX uses 1-based indexing
                    target = int(parts[1]) - 1
                    edges.append((source, target))
        
        # Convert to numpy array
        edges = np.array(edges)

        # Extract unique vertices from edges
        vertices = np.unique(edges.flatten())
        
        # Make undirected if needed
        if not self.is_directed:
            # Create reversed edges and keep unique ones where source < target
            reversed_edges = edges[:, [1, 0]]
            all_edges = np.vstack((edges, reversed_edges))
            unique_edges = np.unique(all_edges, axis=0)
            mask = unique_edges[:, 0] < unique_edges[:, 1]
            edges = unique_edges[mask]
        
        return vertices, edges
    
    def _load_edges_file(self, file_path):
        """
        Load an edges file.
        
        Parameters:
            file_path: Path
                Path to the edges file
        
        Returns:
            tuple: (vertices, edges)
        """
        edges = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments
                if line.startswith('#'):
                    continue
                
                # Parse edge
                parts = line.strip().split()
                if len(parts) >= 2:
                    source = int(parts[0])
                    target = int(parts[1])
                    edges.append((source, target))
        
        # Convert to numpy array
        edges = np.array(edges)

        # Extract unique vertices from edges
        vertices = np.unique(edges.flatten())
        
        # Make undirected if needed
        if not self.is_directed:
            # Create reversed edges and keep unique ones where source < target
            reversed_edges = edges[:, [1, 0]]
            all_edges = np.vstack((edges, reversed_edges))
            unique_edges = np.unique(all_edges, axis=0)
            mask = unique_edges[:, 0] < unique_edges[:, 1]
            edges = unique_edges[mask]
        
        return vertices, edges


class SemanticScholarDataset(DatasetLoader):
    """
    Loader for Semantic Scholar citation network datasets.
    
    Semantic Scholar is a free, AI-powered research tool for scientific literature.
    This loader downloads and processes the citation network from subsets of Semantic Scholar data.
    """
    
    AVAILABLE_DATASETS = {
        "s2-CS": {
            "url": "https://github.com/mattbierbaum/citation-networks/raw/master/s2-CS.tar.gz",
            "description": "Computer Science citation network from Semantic Scholar",
            "nodes_file": "s2-CS-nodes.csv",
            "edges_file": "s2-CS-citations.csv"
        }
    }
    
    def __init__(self, dataset_name="s2-CS"):
        """
        Initialize the Semantic Scholar dataset loader.
        
        Parameters:
            dataset_name: str
                Name of the Semantic Scholar dataset to load
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown Semantic Scholar dataset: {dataset_name}. Available datasets: {', '.join(self.AVAILABLE_DATASETS.keys())}")
        
        self.dataset_info = self.AVAILABLE_DATASETS[dataset_name]
        super().__init__(f"semanticscholar-{dataset_name}")
        self.dataset_name = dataset_name
        self.url = self.dataset_info["url"]
        self.nodes_file = self.dataset_info["nodes_file"]
        self.edges_file = self.dataset_info["edges_file"]
    
    def download(self):
        """
        Download the Semantic Scholar dataset.
        """
        if self.is_downloaded():
            logger.info(f"Dataset {self.dataset_name} already downloaded.")
            return
        
        # Download the compressed file
        filename = self.url.split("/")[-1]
        download_path = self.data_dir / filename
        download_file(self.url, download_path, f"Downloading {self.dataset_name}")
        
        # Extract the file
        extract_file(download_path, self.data_dir)
    
    def is_downloaded(self):
        """
        Check if the dataset is already downloaded.
        """
        # Check if both nodes and edges files exist
        nodes_path = self.data_dir / self.nodes_file
        edges_path = self.data_dir / self.edges_file
        return nodes_path.exists() and edges_path.exists()
    
    def load(self):
        """
        Load the Semantic Scholar dataset as edges.
        
        Returns:
            tuple: (vertices, edges)
                vertices: np.ndarray of shape (num_vertices,)
                edges: np.ndarray of shape (num_edges, 2)
        """
        if not self.is_downloaded():
            self.download()
        
        # Load the nodes file to create a mapping from paper IDs to indices
        nodes_path = self.data_dir / self.nodes_file
        nodes_df = pd.read_csv(nodes_path)
        
        # Create a mapping from paper IDs to indices
        paper_to_idx = {paper_id: idx for idx, paper_id in enumerate(nodes_df['id'])}
        
        # Load the edges file
        edges_path = self.data_dir / self.edges_file
        edges_df = pd.read_csv(edges_path)
        
        # Convert paper IDs to indices
        edges = []
        for _, row in edges_df.iterrows():
            source = paper_to_idx.get(row['source'])
            target = paper_to_idx.get(row['target'])
            
            # Skip edges with unknown papers
            if source is None or target is None:
                continue
            
            edges.append((source, target))
        
        # Convert to numpy array
        edges = np.array(edges)

        # Extract unique vertices from edges
        vertices = np.unique(edges.flatten())
        
        # Make undirected by keeping only edges where source < target
        reversed_edges = edges[:, [1, 0]]
        all_edges = np.vstack((edges, reversed_edges))
        unique_edges = np.unique(all_edges, axis=0)
        mask = unique_edges[:, 0] < unique_edges[:, 1]
        edges = unique_edges[mask]
        
        return vertices, edges


def list_available_datasets():
    """
    List all available datasets from all sources.
    
    Returns:
        dict: Dictionary with dataset information
    """
    all_datasets = {}
    
    # Add SNAP datasets
    for name, info in SNAPDataset.AVAILABLE_DATASETS.items():
        all_datasets[f"snap-{name}"] = {
            "source": "SNAP",
            "name": name,
            "description": info["description"],
            "nodes": info.get("nodes", "Unknown"),
            "edges": info.get("edges", "Unknown"),
            "directed": info["directed"]
        }
    
    # Add Network Repository datasets
    for name, info in NetworkRepositoryDataset.AVAILABLE_DATASETS.items():
        all_datasets[f"netrepo-{name}"] = {
            "source": "Network Repository",
            "name": name,
            "description": info["description"],
            "directed": info["directed"]
        }
    
    # Add Semantic Scholar datasets
    for name, info in SemanticScholarDataset.AVAILABLE_DATASETS.items():
        all_datasets[f"semanticscholar-{name}"] = {
            "source": "Semantic Scholar",
            "name": name,
            "description": info["description"]
        }
    
    return all_datasets


def load_dataset(dataset_name):
    """
    Load a dataset by name.
    
    Parameters:
        dataset_name: str
            Name of the dataset to load
    
    Returns:
        tuple: (vertices, edges)
            vertices: np.ndarray of shape (num_vertices,)
            edges: np.ndarray of shape (num_edges, 2)
    """
    loader = None
    if dataset_name.startswith("snap-"):
        name = dataset_name[5:]  # Remove "snap-" prefix
        loader = SNAPDataset(name)
    if dataset_name.startswith("netrepo-"):
        name = dataset_name[8:]  # Remove "netrepo-" prefix
        loader = NetworkRepositoryDataset(name)
    if dataset_name.startswith("semanticscholar-"):
        name = dataset_name[16:]  # Remove "semanticscholar-" prefix
        loader = SemanticScholarDataset(name)
    # Otherwise, try to guess the source
    if dataset_name in SNAPDataset.AVAILABLE_DATASETS:
        loader = SNAPDataset(dataset_name)
    if dataset_name in NetworkRepositoryDataset.AVAILABLE_DATASETS:
        loader = NetworkRepositoryDataset(dataset_name)
    if dataset_name in SemanticScholarDataset.AVAILABLE_DATASETS:
        loader = SemanticScholarDataset(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return loader.load()


def load_dataset_as_networkx(dataset_name):
    """
    Load a dataset as a NetworkX graph.
    
    Parameters:
        dataset_name: str
            Name of the dataset to load
    
    Returns:
        networkx.Graph: The loaded graph
    """
    vertices, edges = load_dataset(dataset_name)
    
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G,
                                           first_label=0,
                                           ordering='default',
                                           label_attribute=None)
    
    return G
