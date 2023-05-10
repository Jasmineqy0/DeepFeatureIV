import torch
from torch import nn
import os
import yaml

def build_net_for_fully_random_iv(config_info):
    var_info_path = config_info['var_info_path']
    with open(var_info_path, 'r') as f:
        var_info = yaml.safe_load(f)
    
    dim_ts = len(var_info['observed']['ts'])
    dim_iv = len(var_info['observed']['iv'])
    dim_cf = len(var_info['observed']['cf'])

    treatment_net = nn.Sequential(nn.Linear(dim_ts, 16),
                                  nn.ReLU(),
                                  nn.Linear(16, 1))

    instrumental_net = nn.Sequential(nn.Linear(dim_iv, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    if dim_cf:
        covariate_net = nn.Sequential(nn.Linear(dim_cf, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 32),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    nn.Linear(32, 16),
                                    nn.ReLU())
    else:
        covariate_net = None
    
    return treatment_net, instrumental_net, covariate_net
    