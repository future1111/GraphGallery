import os
import sys
import warnings
import uuid
import os.path as osp
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import History
from graphgallery.utils import Progbar

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.nn.functions import softmax
from graphgallery.data.io import makedirs_from_filepath
from graphgallery.utils.raise_error import raise_if_kwargs
from graphgallery.gallery import Model

# TensorFlow 2.1.x
# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
#     This is caused by `tf.gather` and it will be solved in future tensorflow version.
warnings.filterwarnings(
    'ignore',
    '.*Converting sparse IndexedSlices to a dense Tensor of unknown shape.*')

# TensorFlow 2.4.0
# Ignora warnings:
#     UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=...) to a dense Tensor of unknown shape.
#     This may consume a large amount of memory.
warnings.filterwarnings(
    'ignore')


def unravel_batch(batch):
    inputs = labels = out_weight = None
    if isinstance(batch, (list, tuple)):
        inputs = batch[0]
        labels = batch[1]
        if len(batch) > 2:
            out_weight = batch[-1]
    else:
        inputs = batch
    return inputs, labels, out_weight


def default_cfg(model):
    cfg = gg.CfgNode()
    cfg.name = model.name
    cfg.seed = model.seed
    cfg.device = str(model.device)
    cfg.task = "Node Classification"
    cfg.intx = model.intx
    cfg.floatx = model.floatx
    cfg.boolx = model.boolx

    cfg.process = gg.CfgNode()
    cfg.process.graph_transform = None
    cfg.process.adj_transform = None
    cfg.process.attr_transform = None
    cfg.process.label_transform = None

    cfg.model = gg.CfgNode()

    cfg.train = gg.CfgNode()
    cfg.train.epochs = 100
    cfg.train.verbose = 1
    cfg.train.save_best = True

    cfg.train.EarlyStopping = gg.CfgNode()
    cfg.train.EarlyStopping.enabled = False
    cfg.train.EarlyStopping.monitor = 'val_loss'
    cfg.train.EarlyStopping.verbose = 1
    cfg.train.EarlyStopping.mode = "auto"
    cfg.train.EarlyStopping.patience = None

    cfg.train.ModelCheckpoint = gg.CfgNode()
    cfg.train.ModelCheckpoint.enabled = True
    cfg.train.ModelCheckpoint.monitor = 'val_accuracy'
    # checkpoint path
    # use `uuid` to avoid duplication
    cfg.train.ModelCheckpoint.path = osp.join(os.getcwd(),
                                              f"{cfg.name}_checkpoint_{uuid.uuid1().hex[:6]}{gg.file_ext()}")
    cfg.train.ModelCheckpoint.monitor = 'val_accuracy'
    cfg.train.ModelCheckpoint.save_best_only = True
    cfg.train.ModelCheckpoint.save_weights_only = True
    cfg.train.ModelCheckpoint.vervose = 0

    cfg.train.Progbar = gg.CfgNode()
    cfg.train.Progbar.width = 20

    cfg.test = gg.CfgNode()
    cfg.test.verbose = 1

    cfg.test.Progbar = gg.CfgNode()
    cfg.test.Progbar.width = 20

    cfg.predict = gg.CfgNode()
    cfg.predict.return_logits = True
    return cfg


class Trainer(Model):
    def __init__(self, graph, device="cpu", seed=None, name=None, **kwargs):
        """
        Parameters:
        ----------
        graph: Graph or MultiGraph.
        device: string. optional
            The device where the model running on.
        seed: interger scalar. optional
            Used in combination with `tf.random.set_seed` & `np.random.seed`
            & `random.seed` to create a reproducible sequence of tensors
            across multiple calls.
        name: string. optional
            Specified name for the model. (default: :str: `class name`)
        kwargs: other custom keyword arguments. 
        """
        super().__init__(graph, device=device, seed=seed, name=name, **kwargs)
        self.setup()

    def custom_setup(self):
        pass

    def setup(self):
        self.cfg = default_cfg(self)
        self.custom_setup()

    def process(self, **kwargs):

        cfg = self.cfg.process
        _, kwargs = gf.wrapper(self.process_step)(**kwargs)
        cfg.merge_from_dict(kwargs)

        for k, v in cfg.items():
            if k.endswith("transform"):
                setattr(self.transform, k, gf.get(v))
        return self

    def process_step(self, *args, **kwargs):
        raise NotImplementedError

    def build(self, **kwargs):
        if self.backend == "tensorflow":
            with tf.device(self.device):
                self.model, kwargs = gf.wrapper(self.builder)(**kwargs)
        else:
            model, kwargs = gf.wrapper(self.builder)(**kwargs)
            self.model = model.to(self.device)

        self.cfg.model.merge_from_dict(kwargs)
        return self

    def builder(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, train_data, val_data=None, **kwargs):
        cache = self.cache
        cfg = self.cfg.train
        ckpt_cfg = cfg.ModelCheckpoint
        es_cfg = cfg.EarlyStopping
        pb_cfg = cfg.Progbar

        cfg.merge_from_dict(kwargs)

        model = self.model
        if model is None:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        if not isinstance(train_data, Sequence):
            train_data = self.train_sequence(train_data)

        cache.train_data = train_data

        validation = val_data is not None

        if validation:
            if not isinstance(val_data, Sequence):
                val_data = self.test_sequence(val_data)
            cache.val_data = val_data
        else:
            assert ckpt_cfg.enable and ckpt_cfg.monitor.startswith("val")

        callbacks = callbacks_module.CallbackList()

        history = History()
        callbacks.append(history)

        if es_cfg.enabled:
            assert es_cfg.monitor.startswith("val")
            es_callback = EarlyStopping(monitor=es_cfg.monitor,
                                        patience=es_cfg.monitor,
                                        mode=es_cfg.mode,
                                        verbose=es_cfg.verbose)
            callbacks.append(es_callback)

        if ckpt_cfg.enabled:
            if not ckpt_cfg.path.endswith(gg.file_ext()):
                ckpt_cfg.path += gg.file_ext()
            makedirs_from_filepath(ckpt_cfg.path)

            mc_callback = ModelCheckpoint(ckpt_cfg.path,
                                          monitor=ckpt_cfg.monitor,
                                          save_best_only=ckpt_cfg.save_best_only,
                                          save_weights_only=ckpt_cfg.save_weights_only,
                                          verbose=ckpt_cfg.vervose)
            callbacks.append(mc_callback)

        callbacks.set_model(model)
        model.stop_training = False

        verbose = cfg.verbose
        if verbose:
            if verbose <= 2:
                progbar = Progbar(target=cfg.epochs,
                                  width=pb_cfg.width,
                                  verbose=verbose)
            print("Training...")

        logs = gf.BunchDict()
        callbacks.on_train_begin()
        try:
            for epoch in range(cfg.epochs):
                if verbose > 2:
                    progbar = Progbar(target=len(train_data),
                                      width=pb_cfg.width,
                                      verbose=verbose - 2)

                callbacks.on_epoch_begin(epoch)
                callbacks.on_train_batch_begin(0)
                train_logs = self.train_step(train_data)
                train_data.on_epoch_end()
                logs.update(train_logs)

                if validation:
                    valid_logs = self.test_step(val_data)
                    logs.update({("val_" + k): v for k, v in valid_logs.items()})
                    val_data.on_epoch_end()

                callbacks.on_train_batch_end(len(train_data), logs)
                callbacks.on_epoch_end(epoch, logs)

                if verbose > 2:
                    print(f"Epoch {epoch+1}/{epochs}")
                    progbar.update(len(train_data), logs.items())
                elif verbose:
                    progbar.update(epoch + 1, logs.items())

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

            callbacks.on_train_end()
            if ckpt_cfg.enabled:
                if ckpt_cfg.save_weights_only:
                    model.load_weights(ckpt_cfg.path)
                else:
                    self.model = model.load(ckpt_cfg.path)

        finally:
            # to avoid unexpected termination of the model
            self.remove_weights()

        return history

    def test(self, data, **kwargs):

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        cache = self.cache
        cfg = self.cfg.test
        cfg.merge_from_dict(kwargs)

        if isinstance(data, Sequence):
            test_data = data
        else:
            test_data = self.test_sequence(data)

        cache.test_data = test_data

        if cfg.verbose:
            print("Testing...")

        progbar = Progbar(target=len(test_data),
                          width=cfg.Progbar.width,
                          verbose=cfg.verbose)
        logs = gf.BunchDict(**self.test_step(test_data))
        logs.update({k: v.numpy().item() for k, v in logs.items()})
        progbar.update(len(test_data), logs.items())
        return logs

    def train_step(self, sequence):
        model = self.model
        model.reset_metrics()

        for batch in sequence:
            inputs, labels, out_weight = unravel_batch(batch)
            results = model.train_step_on_batch(x=inputs,
                                                y=labels,
                                                out_weight=out_weight,
                                                device=sequence.device)
        return results

    def test_step(self, sequence):
        model = self.model
        model.reset_metrics()

        for batch in sequence:
            inputs, labels, out_weight = unravel_batch(batch)
            results = model.test_step_on_batch(x=inputs,
                                               y=labels,
                                               out_weight=out_weight,
                                               device=sequence.device)
        return results

    def predict(self, predict_data=None, **kwargs):

        if not self.model:
            raise RuntimeError(
                'You must compile your model before training/testing/predicting. Use `trainer.build()`.'
            )

        cache = self.cache
        cfg = self.cfg.predict
        cfg.merge_from_dict(kwargs)

        if predict_data is None:
            predict_data = np.arange(self.graph.num_nodes)

        if not isinstance(predict_data, Sequence):
            predict_data = self.predict_sequence(predict_data)

        cache.predict_data = predict_data

        logits = self.predict_step(predict_data)
        if not cfg.return_logits:
            logits = softmax(logits)

        return logits.squeeze()

    def predict_step(self, sequence):
        logits = []
        model = self.model
        for batch in sequence:
            inputs, labels, out_weight = unravel_batch(batch)
            logit = model.predict_step_on_batch(x=inputs,
                                                out_weight=out_weight,
                                                device=sequence.device)
            logits.append(logit)

        return np.vstack(logits)

    def train_sequence(self, inputs, **kwargs):
        raise NotImplementedError

    def test_sequence(self, inputs, **kwargs):
        return self.train_sequence(inputs, **kwargs)

    def predict_sequence(self, inputs, **kwargs):
        return self.test_sequence(inputs, **kwargs)

    def _test_predict(self, index):
        logit = self.predict(index)
        predict_class = logit.argmax(1)
        labels = self.graph.node_label[index]
        return (predict_class == labels).mean()

    def reset_weights(self):
        # TODO: add pytorch support
        """reset the model to the first time."""
        model = self.model
        if self.backup is None:
            raise RuntimeError(
                "You must store the `backup` before `reset_weights`."
                "`backup` will be automatically stored when the model is built."
            )
        for w, wb in zip(model.weights, self.backup):
            w.assign(wb)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        # Back up
        if isinstance(m, tf.keras.Model) and m.weights:
            self.backup = tf.identity_n(m.weights)
        # TODO assert m is None or isinstance(m, tf.keras.Model) or torch.nn.Module
        self._model = m

    def reset_optimizer(self):
        # TODO: add pytorch support
        model = self.model
        if hasattr(model, 'optimizer'):
            for var in model.optimizer.variables():
                var.assign(tf.zeros_like(var))

    def reset_lr(self, value):
        # TODO: add pytorch support
        model = self.model
        if not hasattr(model, 'optimizer'):
            raise RuntimeError("The model has not attribute `optimizer`!")
        model.optimizer.learning_rate.assign(value)

    def remove_weights(self):
        filepath = self.cfg.train.ModelCheckpoint.path
        if self.backend == "tensorflow":
            remove_extra_tf_files(filepath)

        if osp.exists(filepath):
            os.remove(filepath)

#     def __getattr__(self, attr):
#         ##### FIXME: This may cause ERROR ######
#         try:
#             return self.__dict__[attr]
#         except KeyError:
#             if hasattr(self, "_model") and hasattr(self._model, attr):
#                 return getattr(self._model, attr)
#             raise AttributeError(
#                 f"'{self.name}' and '{self.name}.model' objects have no attribute '{attr}'"
#             )


def remove_extra_tf_files(filepath):
    # for tensorflow weights that saved without h5 formate
    for ext in (".data-00000-of-00001", ".data-00000-of-00002",
                ".data-00001-of-00002", ".index"):
        path = filepath + ext
        if osp.exists(path):
            os.remove(path)

    file_dir = osp.split(osp.realpath(filepath))[0]

    path = osp.join(file_dir, "checkpoint")
    if osp.exists(path):
        os.remove(path)
