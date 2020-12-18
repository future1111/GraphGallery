from .model import Model
from .graph_model import GraphModel
from .gallery_model.gallery_model import GalleryModel
from .model_zoos import (TensorFlow, PyTorch, PyG,
                         DGL_PyTorch, DGL_TensorFlow,
                         Common,
                         MAPPING)
from .sklearn_model.sklearn_model import SklearnModel
from .sklearn_model import *

import sys
import importlib
from typing import Tuple
from graphgallery import backend

__all__ = ["Model", "enabled_models",
           "is_enabled", "GraphModel",
           "GalleryModel", "SklearnModel"]


def get_registry(backend_name=None):
    _backend = backend(backend_name)
    gallery = MAPPING[_backend.abbr]
    return gallery

    mod = importlib.import_module(f".gallery_model.{_backend.abbr}", __name__)


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
    return model in get_registry()


def enabled_models() -> Tuple[str]:
    """Return the models in the gallery enabled by the current backend.

    Returns
    -------
    tuple
        A list of models enabled by the current backend.
    """
    return get_registry() + Common


load_models()
