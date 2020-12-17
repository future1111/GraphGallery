import tensorflow as tf
from tensorflow.keras import Model
from graphgallery.utils import merge_as_list
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

    @tf.function
    def train_step_on_batch(self,
                            x,
                            y=None,
                            sample_weight=None,
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
                out = mask_or_gather(out, sample_weight)
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

    @tf.function
    def test_step_on_batch(self,
                           x,
                           y=None,
                           sample_weight=None,
                           device="CPU"):
        loss_fn = getattr(self, LOSS)
        metrics = getattr(self, METRICS)

        with tf.device(device):
            out = self(x, training=False)
            out = mask_or_gather(out, sample_weight)
            loss = loss_fn(y, out) + tf.reduce_sum(self.losses)
            if isinstance(metrics, list):
                for metric in metrics:
                    metric.update_state(y, out)
            else:
                metrics.update_state(y, out)

            results = [loss] + [metric.result() for metric in getattr(metrics, "metrics", metrics)]
            return dict(zip(self.metrics_names, results))

    @tf.function
    def predict_step_on_batch(self, x, sample_weight=None, device="CPU"):
        with tf.device(device):
            out = self(x, training=False)
            out = mask_or_gather(out, sample_weight)
        return out
