from graphgallery import functional as gf

TensorFlow = gf.Registry("TensorFlow-Gallery")
PyTorch = gf.Registry("PyTorch-Gallery")
PyG = gf.Registry("PyG-Gallery")
DGL_PyTorch = gf.Registry("DGL-PyTorch-Gallery")
DGL_TensorFlow = gf.Registry("DGL-Tensor-FlowGallery")
Common = gf.Registry("Common")

MAPPING = {"tensorflow": TensorFlow,
           "pytorch": PyTorch,
           "pyg": PyG,
           "dgl_torch": DGL_PyTorch,
           "dgl_tf": DGL_TensorFlow}
