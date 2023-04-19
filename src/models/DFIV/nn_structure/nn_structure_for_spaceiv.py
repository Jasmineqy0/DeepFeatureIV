import torch
from torch import nn
import os


def build_net_for_spaceiv(div: int, dts: int):
    treatment_net = nn.Sequential(nn.Linear(dts, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.Tanh())

    instrumental_net = nn.Sequential(nn.Linear(div, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    covariate_net = None
    return treatment_net, instrumental_net, covariate_net

