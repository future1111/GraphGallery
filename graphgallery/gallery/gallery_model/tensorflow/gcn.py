import tensorflow as tf

from graphgallery.gallery import Trainer
from graphgallery.sequence import FullBatchSequence
from graphgallery import functional as gf
from graphgallery.gallery import TensorFlow
from graphgallery.gallery import Trainer
from graphgallery.nn.models import get_model


@TensorFlow.register()
class GCN(Trainer):
    """
        Implementation of Graph Convolutional Networks (GCN).
        `Semi-Supervised Classification with Graph Convolutional Networks
        <https://arxiv.org/abs/1609.02907>`
        Tensorflow 1.x implementation: <https://github.com/tkipf/gcn>
        Pytorch implementation: <https://github.com/tkipf/pygcn>

        Create a Graph Convolutional Networks (GCN) model.

        This can be instantiated in the following way:

            model = GCN(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.
    """

    def process(self, adj_transform="normalize_adj", **kwargs):
        super().process(adj_transform=adj_transform, **kwargs)
        transform = self.transform
        graph = transform.graph_transform(self.graph)
        adj_matrix = transform.adj_transform(graph.adj_matrix)
        node_attr = transform.attr_transform(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)

        return self

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hiddens=[16],
              activations=['relu'],
              dropout=0.5,
              weight_decay=5e-4,
              lr=0.01,
              use_bias=False):

        with tf.device(self.device):
            model = get_model("GCN", self.backend)
            model = model(self.graph.num_node_attrs,
                          self.graph.num_node_classes,
                          hiddens=hiddens,
                          activations=activations,
                          dropout=dropout,
                          weight_decay=weight_decay,
                          lr=lr,
                          use_bias=use_bias)
            model.use_tfn()
            self.model = model

        return self

    def train_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence(x=[self.cache.X, self.cache.A],
                                     y=labels,
                                     out_weight=index,
                                     device=self.device)
        return sequence
