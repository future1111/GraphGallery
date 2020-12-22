from .model import Model
from .trainer import Trainer
from .registered_models import (TensorFlow, PyTorch, PyG,
                                DGL_PyTorch, DGL_TensorFlow,
                                Common,
                                MAPPING)

from .common import *

import sys
import importlib
from typing import Tuple
from graphgallery import backend

__all__ = ["Model", "enabled_models",
           "is_enabled", "Trainer",
           "get_registry"]

def _gen_missing_model(model, backend):
    def _missing_model(*args, **kwargs):
        raise ImportError(f"model {model} is not supported by '{backend}'."
                          " You can switch to other backends by setting"
                          " the 'graphgallery.backend' environment.")

    return _missing_model

def get_registry(backend_name=None):
    _backend = backend(backend_name)
    registry = MAPPING[_backend.abbr]
    return registry


def load_models(backend_name=None):
    _backend = backend(backend_name)
    thismod = sys.modules[__name__]
    importlib.import_module(f".gallery_model.{_backend.abbr}", __name__)
    
    for model, model_class in get_registry(_backend).items():
        setattr(thismod, model, model_class)
        
def is_enabled(model: str) -> bool:
    """Return true if the model is enabled by the current backend.

    Parameters
    ----------
    model : str
        The model name.

    Returns
    -------
    bool
        True if the model is enabled by the current backend.
    """
    return model in enabled_models()


def enabled_models(with_common=True):
    """Return the models in the gallery enabled by the current backend.

    Returns
    -------
    tuple
        A list of models enabled by the current backend.
    """
    return get_registry() + Common


load_models()
