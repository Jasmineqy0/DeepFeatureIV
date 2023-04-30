from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np
import yaml
from typing import Union
import logging
import os

logger = logging.getLogger()

def get_config_file(config_name: str):
    parcs_config_dir = os.sep.join(['src', 'data', 'parcs_simulation', 'configs'])
    parcs_config_path = os.path.join(parcs_config_dir, f'{config_name}.yml')

    return parcs_config_path

def revise_parcs_config(rho: float, sigma: Union[None, float], config_name: str):
    logger.info(f'Revising parcs config file with rho={rho} and sigma={sigma}.')

    config_yml = get_config_file(config_name)

    with open(config_yml, 'r') as f:
        parcs_config = yaml.safe_load(f)
        parcs_config['rho'] = f'constant({rho})'
        if sigma is not None:
            parcs_config['sigma'] = f'constant({sigma})'

    with open(config_yml, 'w') as f:
        yaml.dump(parcs_config, f)

def parcs_simulate(data_size: int, parcs_config:str, rand_seed: int = 42):

    # obtain the graph
    config_yml = get_config_file(parcs_config)
    assert os.path.exists(config_yml), f'error: parcs config file {config_yml} does not exist.'
    nodes, edges = graph_file_parser(config_yml)
    g = Graph(nodes=nodes, edges=edges)

    # obtain the original state of rng
    state = np.random.get_state()
    # change the seed of rng
    np.random.seed(rand_seed)
    # sample from the graph
    samples = g.sample(size=data_size)
    # restore the original state of rng
    np.random.set_state(state)

    return samples