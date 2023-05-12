from typing import Any, Dict
from torch import nn
import yaml

import torch
import torch.nn as nn

def create_neural_network(input_size, hidden_layer, hidden_dim, output_dim, batch_normalization):
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_dim))
    if batch_normalization:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())
    
    # Hidden layers
    for _ in range(hidden_layer - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if batch_normalization:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        
    # output layer
    layers.append(nn.Linear(hidden_dim, output_dim))
    if batch_normalization:
        layers.append(nn.BatchNorm1d(hidden_dim))
    
    # Create and return the sequential model
    model = nn.Sequential(*layers)
    return model

def build_net_for_fully_random_iv(simulation_info: Dict[str, Any], model_configs: Dict[str, Any]):
    # Load the variable dimension information
    var_info_path = simulation_info['var_info_path']
    with open(var_info_path, 'r') as f:
        var_info = yaml.safe_load(f)
    dim_ts = len(var_info['observed']['ts'])
    dim_iv = len(var_info['observed']['iv'])
    dim_cf = len(var_info['observed']['cf'])

    # Create the neural networks
    treatment_net = create_neural_network(dim_ts, model_configs['treatment_hidden_layer'], model_configs['treatment_hidden_dim'], 
                                            model_configs['treatment_output_dim'], model_configs['batch_normalization'])
    instrumental_net = create_neural_network(dim_iv, model_configs['instrumental_hidden_layer'], model_configs['instrumental_hidden_dim'], 
                                                model_configs['instrumental_output_dim'], model_configs['batch_normalization'])
    if dim_cf:
        covariate_net = create_neural_network(dim_cf, model_configs['covariate_hidden_layer'], model_configs['covariate_hidden_dim'],
                                                 model_configs['covariate_output_dim'], model_configs['batch_normalization'])
    else:
        covariate_net = None

    # treatment_net = nn.Sequential(nn.Linear(dim_ts, 128),
    #                               nn.ReLU(),
    #                               nn.Linear(128, 32),
    #                               nn.BatchNorm1d(32),
    #                               nn.ReLU(),
    #                               nn.Linear(32, 16),
    #                               nn.ReLU())

    # instrumental_net = nn.Sequential(nn.Linear(dim_iv, 128),
    #                               nn.ReLU(),
    #                               nn.Linear(128, 32),
    #                               nn.BatchNorm1d(32),
    #                               nn.ReLU(),
    #                               nn.Linear(32, 16),
    #                               nn.ReLU())

    # if dim_cf:
    #     covariate_net = nn.Sequential(nn.Linear(dim_cf, 128),
    #                               nn.ReLU(),
    #                               nn.Linear(128, 32),
    #                               nn.BatchNorm1d(32),
    #                               nn.ReLU(),
    #                               nn.Linear(32, 16),
    #                               nn.ReLU())
    # else:
    #     covariate_net = None
    
    return treatment_net, instrumental_net, covariate_net
    