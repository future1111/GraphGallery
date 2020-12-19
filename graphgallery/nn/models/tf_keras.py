import warnings
import tensorflow as tf
from tensorflow.keras import Model
from graphgallery.utils import merge_as_list
from tensorflow.keras.activations import softmax
from graphgallery import functional as gf
from graphgallery.functional.tensor.tensorflow import mask_or_gather

from distutils.version import LooseVersion

if LooseVersion(tf.__version__) >= LooseVersion("2.2.0"):
    METRICS = "compiled_metrics"
    LOSS = "compiled_loss"
else:
    METRICS = "metrics"
    LOSS = "loss"


class TFKeras(Model):
    """High-level encapsulation of Tensorflow Keras Model."""
    _use_tfn = False

    def use_tfn(self):
        assert not self._use_tfn, "'tf.function' has been used."
        self.train_step_on_batch = tf.function(self.train_step_on_batch, experimental_relax_shapes=True)
        self.test_step_on_batch = tf.function(self.test_step_on_batch, experimental_relax_shapes=True)
        self.predict_step_on_batch = tf.function(self.predict_step_on_batch, experimental_relax_shapes=True)
        self._use_tfn = True

    # @tf.function(experimental_relax_shapes=True)
    def train_step_on_batch(self,
                            x,
                            y=None,
                            out_weight=None,
                            device="CPU"):
        # FIXME: self.metrics would return '[]' for tensorflow>=2.2.0
        # See <https://github.com/tensorflow/tensorflow/issues/37990>
        # the loss or metrics must be called to build the compiled_loss
        # or compiled_metrics
        loss_fn = getattr(self, LOSS)
        metrics = getattr(self, METRICS)
        optimizer = self.optimizer

        with tf.device(device):
            with tf.GradientTape() as tape:
                out = self(x, training=True)
                out = mask_or_gather(out, out_weight)
                loss = loss_fn(y, out) + tf.reduce_sum(self.losses)
                if isinstance(metrics, list):
                    for metric in metrics:
                        metric.update_state(y, out)
                else:
                    metrics.update_state(y, out)

            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))

            results = [loss] + [metric.result() for metric in getattr(metrics, "metrics", metrics)]
            return dict(zip(self.metrics_names, results))

    # @tf.function(experimental_relax_shapes=True)
    def test_step_on_batch(self,
                           x,
                           y=None,
                           out_weight=None,
                           device="CPU"):
        loss_fn = getattr(self, LOSS)
        metrics = getattr(self, METRICS)

        with tf.device(device):
            out = self(x, training=False)
            out = mask_or_gather(out, out_weight)
            loss = loss_fn(y, out) + tf.reduce_sum(self.losses)
            if isinstance(metrics, list):
                for metric in metrics:
                    metric.update_state(y, out)
            else:
                metrics.update_state(y, out)

            results = [loss] + [metric.result() for metric in getattr(metrics, "metrics", metrics)]
            return dict(zip(self.metrics_names, results))

    # @tf.function(experimental_relax_shapes=True)
    def predict_step_on_batch(self, x, out_weight=None,
                              return_logits=True,
                              device="CPU"):
        with tf.device(device):
            out = self(x, training=False)
            out = mask_or_gather(out, out_weight)
            if not return_logits:
                out = softmax(out)
        return out

    @property
    def custom_objects(self):
        return self._custom_objects

    @custom_objects.setter
    def custom_objects(self, objs):
        assert isinstance(objs, dict), objs
        self._custom_objects = objs
