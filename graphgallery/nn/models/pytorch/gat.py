import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Dropout
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import GraphAttention, SparseGraphAttention
from graphgallery.nn.metrics.pytorch import Accuracy


class GAT(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids=[8],
                 n_heads=[8],
                 acts=['elu'],
                 dropout=0.6,
                 weight_decay=5e-4,
                 lr=0.01,
                 use_bias=True):

        super().__init__()

        layers = ModuleList()
        paras = []

        inc = in_channels
        pre_head = 1
        for hid, n_head, activation in zip(hids, n_heads, acts):
            layer = SparseGraphAttention(inc * pre_head,
                                         hid,
                                         activation=act,
                                         attn_heads=n_head,
                                         reduction='concat',
                                         use_bias=use_bias)
            layers.append(layer)
            paras.append(
                dict(params=layer.parameters(), weight_decay=weight_decay))
            inc = hid
            pre_head = n_head

        layer = SparseGraphAttention(inc * pre_head,
                                     out_channels,
                                     attn_heads=1,
                                     reduction='average',
                                     use_bias=use_bias)
        layers.append(layer)
        # do not use weight_decay in the final layer
        paras.append(dict(params=layer.parameters(), weight_decay=0.))

        self.layers = layers
        self.dropout = Dropout(dropout)
        self.compile(loss=torch.nn.CrossEntropyLoss(),
                     optimizer=optim.Adam(paras, lr=lr),
                     metrics=[Accuracy()])

    def forward(self, x, adj):

        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, adj)

        return x
