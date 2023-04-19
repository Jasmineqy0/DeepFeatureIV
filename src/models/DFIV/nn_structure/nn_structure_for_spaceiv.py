import torch
from torch import nn
from dotenv import load_dotenv
load_dotenv()
import os


def build_net_for_spaceiv():
    treatment_net = nn.Sequential(nn.Linear(int(os.getenv('dts')), 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.Tanh())

    instrumental_net = nn.Sequential(nn.Linear(int(os.getenv('div')), 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    covariate_net = None
    return treatment_net, instrumental_net, covariate_net

