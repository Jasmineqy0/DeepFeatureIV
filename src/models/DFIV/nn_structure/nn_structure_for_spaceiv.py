from typing import Any, Dict
import torch
from torch import nn
import os
from src.models.DFIV.nn_structure.utils import create_neural_network


def build_net_for_spaceiv(dim_iv: int, dim_ts: int, model_configs: Dict[str, Any]):
    # instrumental_net = create_neural_network(dim_iv, model_configs['num_layer'], model_configs['hidden_dim'], model_configs['batch_normalization'])
    
    # treatment_net = create_neural_network(dim_ts, model_configs['num_layer'], model_configs['hidden_dim'], model_configs['batch_normalization'])
    treatment_net = nn.Sequential(nn.Linear(20, 16),
                                  nn.ReLU(),
                                  nn.Linear(16, 1))

    instrumental_net = nn.Sequential(nn.Linear(10, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    # covariate_net = nn.Sequential(nn.Linear(2, 128),
    #                               nn.ReLU(),
    #                               nn.Linear(128, 32),
    #                               nn.BatchNorm1d(32),
    #                               nn.ReLU(),
    #                               nn.Linear(32, 16),
    #                               nn.ReLU())

    covariate_net = None
    return treatment_net, instrumental_net, covariate_net

