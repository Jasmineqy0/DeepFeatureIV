from typing import Any, Dict
import torch
from torch import nn
import os
from src.models.DFIV.nn_structure.utils import create_neural_network


def build_net_for_spaceiv(dim_iv: int, dim_ts: int, model_configs: Dict[str, Any]):
    instrumental_net = create_neural_network(dim_iv, model_configs['num_layer'], model_configs['hidden_dim'], model_configs['batch_normalization'])
    
    treatment_net = create_neural_network(dim_ts, model_configs['num_layer'], model_configs['hidden_dim'], model_configs['batch_normalization'])

    covariate_net = None
    return treatment_net, instrumental_net, covariate_net

