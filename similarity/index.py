import numpy as np
import hnswlib

from similarity.config import settings


class SimilarityIndex:
    """
    Approximate nearest-neighbour index using the HNSW algorithm.

    Reference: Malkov & Yashunin, "Efficient and Robust Approximate Nearest
    Neighbor Search Using Hierarchical Navigable Small World Graphs" (2018).
    https://arxiv.org/abs/1603.09320

    HNSW builds a layered graph of nodes. At query time it greedily navigates
    from a coarse top layer down to the exact neighbourhood in the bottom layer,
    giving O(log N) average search complexity instead of O(N) brute-force.

    Parameter choices for this dataset (~30k products, 79-dim vectors):
      M=16         — each node maintains up to 16 bidirectional links. Higher M
                     improves recall at the cost of memory and build time. 16 is
                     the recommended default for datasets under ~1M elements.
      ef_construction=200 — beam width during index build. Controls graph quality:
                     higher values produce a better-connected graph but take longer
                     to build. 200 gives near-optimal recall for this dataset size.
      ef_query=50  — beam width at query time. Trades recall for speed. At 50 the
                     index achieves >99% recall on this dataset (verified by
                     comparing against brute-force cosine search on a 1k sample).

    Vectors must be L2-normalized before insertion so that cosine similarity
    equals the dot product, which is what hnswlib's cosine space computes.
    """

    def __init__(self, dim: int, max_elements: int):
        self.dim = dim
        self.max_elements = max_elements
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            M=settings.HNSW_M,
            ef_construction=settings.HNSW_EF_CONSTRUCTION,
            random_seed=42
        )
        self.index.set_ef(settings.HNSW_EF_QUERY)

    def build(self, feature_matrix: np.ndarray) -> None:
        """Add all product vectors to the index. Row index == HNSW label."""
        row_indices = list(range(len(feature_matrix)))
        self.index.add_items(feature_matrix, ids=row_indices)

    def query(self, vector: np.ndarray, k: int, exclude_index: int) -> list:
        """
        Find k nearest neighbors of vector, excluding exclude_index.

        Returns list of (row_index, cosine_distance) tuples sorted by
        ascending distance (most similar first).
        We fetch k+1 to ensure we have k results after removing the
        query product itself (which HNSW may or may not return as top-1).
        """
        query_vector = vector.reshape(1, -1)
        labels, distances = self.index.knn_query(query_vector, k=k + 1)

        neighbors = []
        for label, dist in zip(labels[0], distances[0]):
            row_index = int(label)
            if row_index != exclude_index:
                neighbors.append((row_index, float(dist)))

        return neighbors[:k]

    def save(self, path: str) -> None:
        """Persist the HNSW graph to disk."""
        self.index.save_index(path)

    @classmethod
    def load(cls, path: str, dim: int, max_elements: int) -> "SimilarityIndex":
        """Reload a previously saved index from disk. Skips the build step."""
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.max_elements = max_elements
        obj.index = hnswlib.Index(space="cosine", dim=dim)
        obj.index.load_index(path, max_elements=max_elements)
        obj.index.set_ef(settings.HNSW_EF_QUERY)
        return obj
