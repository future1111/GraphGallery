import warnings
import numpy as np
import scipy.sparse as sp

from ..transforms import Transform
from ..get_transform import Transformers
from ..edge_level import filter_singletons
from ..adj_matrix import remove_edge
import graphgallery as gg

__all__ = ["JaccardDetection", "CosineDetection", "SVD",
           "jaccard_detection", "cosine_detection", "svd"]


def jaccard_similarity(A, B):
    intersection = np.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (np.count_nonzero(A, axis=1) + np.count_nonzero(B, axis=1) + intersection + gg.epsilon())
    return J


def cosine_similarity(A, B):
    inner_product = (A * B).sum(1)
    C = inner_product / (np.sqrt(np.square(A).sum(1)) * np.sqrt(np.square(B).sum(1)) + gg.epsilon())
    return C


def filter_edges_by_similarity(adj_matrix, node_attr,
                               similarity_fn, threshold=0.01,
                               allow_singleton=False):

    rows, cols = adj_matrix.nonzero()

    A = node_attr[rows]
    B = node_attr[cols]
    S = similarity_fn(A, B)
    idx = np.where(S <= threshold)[0]
    flips = np.vstack([rows[idx], cols[idx]])
    if not allow_singleton and flips.size > 0:
        flips = filter_singletons(flips, adj_matrix)
    return flips


def jaccard_detection(adj_matrix, node_attr, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj_matrix, node_attr,
                                      similarity_fn=jaccard_similarity,
                                      threshold=threshold,
                                      allow_singleton=allow_singleton)


def cosine_detection(adj_matrix, node_attr, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj_matrix, node_attr,
                                      similarity_fn=cosine_similarity,
                                      threshold=threshold,
                                      allow_singleton=allow_singleton)


@Transformers.register()
class JaccardDetection(Transform):

    def __init__(self, threshold=0., allow_singleton=False):
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton
        self.flips = None

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO, multiple graph
        assert not graph.multiple
        graph = graph.copy()
        adj_matrix = graph.adj_matrix
        node_attr = graph.node_attr
        structure_flips = jaccard_detection(adj_matrix, node_attr,
                                            threshold=self.threshold,
                                            allow_singleton=self.allow_singleton)
        self.flips = structure_flips
        graph.update(adj_matrix=remove_edge(adj_matrix, structure_flips, symmetric=False))
        return graph

    def extra_repr(self):
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


@Transformers.register()
class CosineDetection(Transform):

    def __init__(self, threshold=0., allow_singleton=False):
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton
        self.flips = None

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO: multiple graph
        assert not graph.multiple
        graph = graph.copy()
        adj_matrix = graph.adj_matrix
        node_attr = graph.node_attr
        structure_flips = cosine_detection(adj_matrix, node_attr,
                                           threshold=self.threshold,
                                           allow_singleton=self.allow_singleton)

        self.flips = structure_flips
        graph.update(adj_matrix=remove_edge(adj_matrix, structure_flips, symmetric=False))
        return graph

    def extra_repr(self):
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


@Transformers.register()
class SVD(Transform):

    def __init__(self, k=50, threshold=0.01, binaryzation=False):
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.binaryzation = binaryzation

    def __call__(self, graph):
        assert isinstance(graph, gg.data.HomoGraph), type(graph)
        # TODO: multiple graph
        assert not graph.multiple
        graph = graph.copy()
        adj_matrix = svd(graph.adj_matrix, k=self.k,
                         threshold=self.threshold,
                         binaryzation=self.binaryzation)
        graph.update(adj_matrix=adj_matrix)
        return graph

    def extra_repr(self):
        return f"k={self.k}, threshold={self.threshold}, binaryzation={self.binaryzation}"


def svd(adj_matrix, k=50, threshold=0.01, binaryzation=False):
    adj_matrix = adj_matrix.asfptype()

    U, S, V = sp.linalg.svds(adj_matrix, k=k)
    adj_matrix = (U * S) @ V

    if threshold is not None:
        # sparsification
        adj_matrix[adj_matrix <= threshold] = 0.

    adj_matrix = sp.csr_matrix(adj_matrix)

    if binaryzation:
        # TODO
        adj_matrix.data[adj_matrix.data > 0] = 1.0

    return adj_matrix
