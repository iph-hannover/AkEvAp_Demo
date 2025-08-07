import matplotlib as mpl
mpl.use('Agg')  # use matplotlib agg backend for writing to file instead of rendering to display
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn


class SimpleClassificator(nn.Module):
    def __init__(self, cfg):
        super(SimpleClassificator, self).__init__()
        self.cfg = cfg

        emb_size = cfg.emb_size
        self.dimensionality = 3 if cfg.use3D else 2
        self.block_in = nn.Linear(cfg.num_joints*self.dimensionality, emb_size)
        self.relu = nn.ReLU()
        self.lin_out = nn.Linear(in_features=emb_size, out_features=cfg.num_classes)

    def forward(self, input):
        batch, num_joints = input.size()  # b x joints*2
        assert (num_joints//self.dimensionality == self.cfg.num_joints)

        x0 = self.block_in(input)       # b x emb_size
        x1 = self.relu(x0)
        out = self.lin_out(x1)          # b x num_classes

        if not self.training:
            pass
        return dict(logits=out)
