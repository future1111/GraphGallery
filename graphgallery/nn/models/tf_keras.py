import tensorflow as tf
from tensorflow.keras import Model
from graphgallery.utils import merge_as_list


def mask_or_gather(out, imask):
    if imask is not None:
        if imask.dtype.is_bool:
            return tf.boolean_mask(out, imask)
        else:
            return tf.gather(out, imask)
    return out


class TFKeras(Model):
    """High-level encapsulation of Tensorflow Keras Model."""

    @tf.function
    def train_on_batch(self,
                       x,
                       imask=None,
                       y=None,
                       eager=True,
                       sample_weight=None,
                       class_weight=None,
                       reset_metrics=True,
                       return_dict=False):
        if not eager:
            # FIXME: for Tensorflow 2.4.0, if causes an error:
            # ValueError: Data cardinality is ambiguous.
            # https://github.com/tensorflow/tensorflow/issues/42175
            # this method does not work for Tensorflow>=2.4.0
            x = merge_as_list(x, imask)
            return super().train_on_batch(x=x,
                                          y=y,
                                          sample_weight=sample_weight,
                                          class_weight=class_weight,
                                          reset_metrics=reset_metrics)
        else:
            # FIXME: self.metrics would return '[]' for tensorflow>=2.2.0
            # See <https://github.com/tensorflow/tensorflow/issues/37990>
            # the loss or metrics must be called to build the compiled_loss
            # or compiled_metrics
            loss_fn = self.loss
            optimizer = self.optimizer
            metrics = self.metrics

            with tf.GradientTape() as tape:
                out = self(x, training=True)
                out = mask_or_gather(out, imask)
                loss = loss_fn(y, out)
                if isinstance(metrics, list):
                    for metric in metrics:
                        metric.update_state(y, out)
                else:
                    metrics.update_state(y, out)

            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))

            results = [loss] + [metric.result() for metric in metrics]
            return dict(zip(self.metrics_names, results))

    def test_on_batch(self,
                      x,
                      imask=None,
                      y=None,
                      eager=True,
                      sample_weight=None,
                      reset_metrics=True,
                      return_dict=False):
        if not eager:
            x = merge_as_list(x, imask)
            return super().test_on_batch(x=x,
                                         y=y,
                                         sample_weight=sample_weight,
                                         class_weight=class_weight,
                                         reset_metrics=reset_metrics)
        else:
            loss_fn = self.loss
            optimizer = self.optimizer
            metrics = self.metrics

            out = self(x, training=False)
            out = mask_or_gather(out, imask)
            loss = loss_fn(y, out)
            if isinstance(metrics, list):
                for metric in metrics:
                    metric.update_state(y, out)
            else:
                metrics.update_state(y, out)


            results = [loss] + [metric.result() for metric in metrics]
            return dict(zip(self.metrics_names, results))
