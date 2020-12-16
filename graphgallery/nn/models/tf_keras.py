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
    @property
    def metrics(self):
        m = super().metrics
        if not m:
            return self.default_metrics
        return m

    @property
    def metrics_names(self):
        m = super().metrics_names

        if not m:
            return self.default_metrics_names
        if m[0] != 'loss':
            m = ['loss'] + m
        return m

    @property
    def default_metrics(self):
        return self.compiled_metrics._metrics

    @property
    def default_metrics_names(self):
        return ['loss'] + [m.name for m in self.compiled_metrics._metrics]

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
            x = merge_as_list(x, imask)
            return super().train_on_batch(x=x,
                                          y=y,
                                          sample_weight=sample_weight,
                                          class_weight=class_weight,
                                          reset_metrics=reset_metrics)
        else:
            loss_fn = self.loss
            optimizer = self.optimizer

            if reset_metrics:
                self.reset_metrics()
                for metric in self.metrics:
                    metric.reset_states()

            with tf.GradientTape() as tape:
                out = self(x, training=True)
                out = mask_or_gather(out, imask)
                loss = loss_fn(y, out)
                for metric in self.metrics:
                    metric.update_state(y, out)

            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))

            results = [loss] + [metric.result() for metric in self.metrics]
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

            if reset_metrics:
                self.reset_metrics()
                for metric in self.metrics:
                    metric.reset_states()

            out = self(x, training=False)
            out = mask_or_gather(out, imask)
            loss = loss_fn(y, out)
            for metric in self.metrics:
                metric.update_state(y, out)

            results = [loss] + [metric.result() for metric in self.metrics]

            return dict(zip(self.metrics_names, results))