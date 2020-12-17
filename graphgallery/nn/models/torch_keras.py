import torch
import numpy as np
import torch.nn as nn

from torch import optim
from torch.autograd import Variable

from collections import OrderedDict
from graphgallery.utils import saver


class TorchKeras(nn.Module):
    """Keras like PyTorch Model."""

    def __init__(self, *args, **kwargs):
        self.__doc__ = super().__doc__

        super().__init__(*args, **kwargs)

        # To be compatible with TensorFlow
        self._in_multi_worker_mode = dummy_function
        self._is_graph_network = dummy_function
        self.distribute_strategy = None

        # initialize
        self.optimizer = None
        self.metrics = None
        self.loss = None

    def train_step_on_batch(self,
                            x,
                            y=None,
                            sample_weight=None,
                            device="cpu"):
        self.train()
        optimizer = self.optimizer
        loss_fn = self.loss
        metrics = self.metrics
        optimizer.zero_grad()

        out = self(*x if isinstance(x, (list, tuple)) else [x])
        if sample_weight is not None:
            out = out[sample_weight]
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def test_step_on_batch(self,
                           x,
                           y=None,
                           sample_weight=None,
                           device="cpu"):
        self.eval()
        loss_fn = self.loss
        metrics = self.metrics

        out = self(*x if isinstance(x, (list, tuple)) else [x])
        if sample_weight is not None:
            out = out[sample_weight]
        loss = loss_fn(out, y)
        for metric in metrics:
            metric.update_state(y.cpu(), out.detach().cpu())

        results = [loss.cpu().detach()] + [metric.result() for metric in metrics]
        return dict(zip(self.metrics_names, results))

    @torch.no_grad()
    def predict_step_on_batch(self, x, sample_weight=None, device="cpu"):
        self.eval()
        out = self(*x if isinstance(x, (list, tuple)) else [x])
        if sample_weight is not None:
            out = out[sample_weight]
        return out.cpu().detach()

    def build(self, inputs):
        # TODO
        pass

    def compile(self, loss=None, optimizer=None, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        self.metrics = metrics

    def reset_metrics(self):
        assert self.metrics is not None
        for metric in self.metrics:
            metric.reset_states()

    @property
    def metrics_names(self):
        assert self.metrics is not None
        return ['loss'] + [metric.name for metric in self.metrics]

    def summary(self):
        # TODO
        pass

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     **kwargs):
        saver.save_torch_weights(self,
                                 filepath,
                                 overwrite=overwrite,
                                 save_format=save_format,
                                 **kwargs)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        saver.save_torch_model(self,
                               filepath,
                               overwrite=overwrite,
                               save_format=save_format,
                               **kwargs)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


def dummy_function(*args, **kwargs):
    ...
