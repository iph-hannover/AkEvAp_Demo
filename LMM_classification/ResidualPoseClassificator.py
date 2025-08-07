import matplotlib as mpl
mpl.use('Agg')  # use matplotlib agg backend for writing to file instead of rendering to display
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn


class ResidualPoseClassificator(nn.Module):
    def __init__(self, cfg):
        super(ResidualPoseClassificator, self).__init__()
        self.cfg = cfg

        self.emb_size = cfg.emb_size
        self.dropout_rate = cfg.dropout_rate
        self.dimensionality = 3 if cfg.use3D else 2
        self.activation = self._get_activation_fn(cfg.activation_fn)
        self.num_residual_blocks = cfg.num_residual_blocks

        self.block_in = nn.Linear(cfg.num_joints*self.dimensionality, self.emb_size)

        self.residual_blocks = nn.ModuleList([ResidualBlock(self.emb_size, activation=self.activation) for _ in range(self.num_residual_blocks)])

        self.bn1 = nn.BatchNorm1d(self.emb_size)
        self.drop1 = nn.Dropout(self.dropout_rate)
        self.lin_out = nn.Linear(in_features=self.emb_size, out_features=cfg.num_classes)

        if cfg.weight_init:
            self._initialize_weights()

    def forward(self, input):
        batch, num_joints = input.size()  # b x joints*dim
        assert (num_joints//self.dimensionality == self.cfg.num_joints)

        x0 = self.block_in(input)       # b x emb_size
        x = self.activation(x0)

        for residual_block in self.residual_blocks:
            x = residual_block(x)

        out = self.lin_out(x)         # b x num_classes

        if not self.training:
            pass
        return dict(logits=out)

    def _get_activation_fn(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def _initialize_weights(self):
        """Initializes the weights with a more robust method (He or Xavier initialization)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    def __init__(self, emb_size, activation=nn.ReLU()):
        super().__init__()
        self.fc = nn.Linear(emb_size, emb_size)
        self.bn = nn.BatchNorm1d(emb_size)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.fc(x) + x))
