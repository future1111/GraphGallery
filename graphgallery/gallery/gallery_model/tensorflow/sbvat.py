import numpy as np
import tensorflow as tf

from graphgallery.gallery import GalleryModel
from graphgallery.sequence import SBVATSampleSequence, FullBatchSequence
from graphgallery.utils.bvat_utils import get_normalized_vector, kl_divergence_with_logit, entropy_y_x

from graphgallery import functional as gf
from graphgallery.functional.tensor.tensorflow import mask_or_gather
from graphgallery.nn.models.tensorflow import GCN as tfGCN
from graphgallery.gallery import TensorFlow

from distutils.version import LooseVersion

if LooseVersion(tf.__version__) >= LooseVersion("2.2.0"):
    METRICS = "compiled_metrics"
    LOSS = "compiled_loss"
else:
    METRICS = "metrics"
    LOSS = "loss"


@TensorFlow.register()
class SBVAT(GalleryModel):
    """
        Implementation of sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT).
        `Batch Virtual Adversarial Training for Graph Convolutional Networks
        <https://arxiv.org/abs/1902.09192>`
        Tensorflow 1.x implementation: <https://github.com/thudzj/BVAT>


    """

    def __init__(self,
                 graph,
                 n_samples=50,
                 adj_transform="normalize_adj",
                 attr_transform=None,
                 graph_transform=None,
                 device="cpu",
                 seed=None,
                 name=None,
                 **kwargs):
        r"""Create a sample-based Batch Virtual Adversarial Training
        Graph Convolutional Networks (SBVAT) model.

        This can be instantiated in the following way:

            model = SBVAT(graph)
                with a `graphgallery.data.Graph` instance representing
                A sparse, attributed, labeled graph.

        Parameters:
        ----------
        graph: An instance of `graphgallery.data.Graph`.
            A sparse, attributed, labeled graph.
        n_samples (Positive integer, optional):
            The number of sampled subset nodes in the graph where the length of the
            shortest path between them is at least `4`. (default :obj: `50`)
        adj_transform: string, `transform`, or None. optional
            How to transform the adjacency matrix. See `graphgallery.functional`
            (default: :obj:`'normalize_adj'` with normalize rate `-0.5`.
            i.e., math:: \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}})
        attr_transform: string, `transform`, or None. optional
            How to transform the node attribute matrix. See `graphgallery.functional`
            (default :obj: `None`)
        device: string. optional
            The device where the model is running on.
            You can specified ``CPU``, ``GPU`` or ``cuda``
            for the model. (default: :str: `cpu`, i.e., running on the `CPU`)
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors across
            multiple calls. (default :obj: `None`, i.e., using random seed)
        name: string. optional
            Specified name for the model. (default: :str: `class.__name__`)
        kwargs: other custom keyword parameters.
        """
        super().__init__(graph, device=device, seed=seed, name=name,
                         adj_transform=adj_transform,
                         attr_transform=attr_transform,
                         graph_transform=graph_transform,
                         **kwargs)
        self.register_cache("n_samples", n_samples)

        self.process()

    def process_step(self):
        graph = self.transform.graph_transform(self.graph)
        adj_matrix = self.transform.adj_transform(graph.adj_matrix)
        node_attr = self.transform.attr_transform(graph.node_attr)

        X, A = gf.astensors(node_attr, adj_matrix, device=self.device)

        # ``A`` and ``X`` and ``neighbors`` are cached for later use
        self.register_cache("X", X)
        self.register_cache("A", A)
        self.register_cache("neighbors", gf.find_4o_nbrs(adj_matrix))

    # use decorator to make sure all list arguments have the same length
    @gf.equal()
    def build(self,
              hids=[16],
              acts=['relu'],
              dropout=0.5,
              lr=0.01,
              weight_decay=5e-4,
              use_bias=False,
              p1=1.,
              p2=1.,
              n_power_iterations=1,
              epsilon=0.03,
              xi=1e-6):

        with tf.device(self.device):
            self.model = tfGCN(self.graph.num_node_attrs,
                               self.graph.num_node_classes,
                               hids=hids,
                               acts=acts,
                               dropout=dropout,
                               weight_decay=weight_decay,
                               lr=lr,
                               use_bias=use_bias)

        self.register_cache("p1", p1)  # Alpha
        self.register_cache("p2", p2)  # Beta
        self.register_cache("xi", xi)  # Small constant for finite difference
        # Norm length for (virtual) adversarial training
        self.register_cache("epsilon", epsilon)
        self.register_cache("n_power_iterations", n_power_iterations)  # Number of power iterations

    @tf.function
    def train_step(self, sequence):
        model = self.model
        loss_fn = getattr(model, LOSS)
        metrics = getattr(model, METRICS)
        optimizer = model.optimizer

        with tf.device(self.device):

            for inputs, y, out_weight in sequence:
                x, adj, adv_mask = inputs
                with tf.GradientTape() as tape:
                    logit = model([x, adj], training=True)
                    out = mask_or_gather(logit, out_weight)
                    loss = loss_fn(y, out)
                    entropy_loss = entropy_y_x(logit)
                    vat_loss = self.virtual_adversarial_loss(x,
                                                             adj,
                                                             logit=logit,
                                                             adv_mask=adv_mask)
                    loss += self.cache.p1 * vat_loss + self.cache.p2 * entropy_loss

                    if isinstance(metrics, list):
                        for metric in metrics:
                            metric.update_state(y, out)
                    else:
                        metrics.update_state(y, out)

                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))

            results = [loss] + [metric.result() for metric in getattr(metrics, "metrics", metrics)]
            return dict(zip(model.metrics_names, results))

    def virtual_adversarial_loss(self, x, adj, logit, adv_mask):
        d = tf.random.normal(shape=tf.shape(x), dtype=self.floatx)
        model = self.model
        for _ in range(self.cache.n_power_iterations):
            d = get_normalized_vector(d) * self.cache.xi
            logit_p = logit
            with tf.GradientTape() as tape:
                tape.watch(d)
                logit_m = model([x + d, adj], training=True)
                dist = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
            grad = tape.gradient(dist, d)
            d = tf.stop_gradient(grad)

        r_vadv = get_normalized_vector(d) * self.cache.epsilon
        logit_p = tf.stop_gradient(logit)
        logit_m = model([x + r_vadv, adj], training=True)
        loss = kl_divergence_with_logit(logit_p, logit_m, adv_mask)
        return loss

    def train_sequence(self, index):

        labels = self.graph.node_label[index]

        sequence = SBVATSampleSequence([self.cache.X, self.cache.A],
                                       labels,
                                       out_weight=index,
                                       neighbors=self.cache.neighbors,
                                       n_samples=self.cache.n_samples,
                                       device=self.device)

        return sequence

    def test_sequence(self, index):

        labels = self.graph.node_label[index]
        sequence = FullBatchSequence([self.cache.X, self.cache.A],
                                     labels,
                                     out_weight=index,
                                     device=self.device)

        return sequence
