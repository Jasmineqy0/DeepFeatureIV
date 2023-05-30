from itertools import product
import numpy as np
import logging
from typing import  Union
from pathlib import Path

from ..data.data_class import TrainDataSet, TestDataSet
from ..data.parcs_simulation.parcs_simulate import get_config_file
from src.data.demand_design import f, psi
from src.data.params.ops import get_parcs_config_name

from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser

np.random.seed(42)
logger = logging.getLogger()

TEST_SEED = 9999

def generate_test_demand_design_parcs(function: str, noise_price_bias: Union[None, float] = None) -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    # obtain the approx range of the expected price
    boostrap_size = 5000
    time = np.linspace(0.0, 10, boostrap_size)
    # E[C] = 0, E[V] = noise_price_bias -> E[P | t] = 25 + (0 + 3) * psi(t) + noise_price_bias
    noise_price_bias = 0.0 if noise_price_bias is None else noise_price_bias
    exp_true_price =  25 + (0 + 3) * psi(function, time) + noise_price_bias
    exp_min_price, exp_max_price = np.round(np.min(exp_true_price)), np.round(np.max(exp_true_price))
    
    # price = np.linspace(10, 25, 20)
    price = np.linspace(exp_min_price, exp_max_price, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    data = []
    target = []
    for p, t, s in product(price, time, emotion):
        data.append([p, t, s])
        target.append(f(function, p, t, s))
    features = np.array(data)
    targets: np.ndarray = np.array(target)[:, np.newaxis]

    test_data = TestDataSet(treatment=features[:, 0:1],
                            covariate=features[:, 1:],
                            structural=targets)
    return test_data


def generate_train_demand_design_parcs(data_size: int,
                                       function: str,
                                       hetero: bool = False,
                                       noise_price_bias: Union[None, float] = None,
                                       rand_seed: int = 42,
                                       **args) -> TrainDataSet:
    """

    Parameters
    ----------
    data_size : int
        size of data
    rho : float
        parameter for noise correlation
    rand_seed : int
        random seed


    Returns
    -------
    train_data : TrainDataSet
    """
    # obtain the parcs config name
    parcs_config = get_parcs_config_name(hetero, function, noise_price_bias)
    
    # obtain the graph
    config_yml = get_config_file(parcs_config)
    assert Path(config_yml).exists(), f'error: parcs config file {config_yml} does not exist.'
    
    # parse the config file
    nodes, edges = graph_file_parser(config_yml)
    g = Graph(nodes=nodes, edges=edges)
    
    # obtain the original state of rng
    state = np.random.get_state()
    
    # change the seed of rng
    np.random.seed(rand_seed)
    
    # sample from the graph
    samples = g.sample(size=data_size)
    assert np.isnan(samples.to_numpy()).sum() == 0, f'error: {np.isnan(samples).sum()} NaNs in the samples.'
    
    # restore the original state of rng
    np.random.set_state(state)

    emotion = samples['emotion'].to_numpy()
    time = samples['time'].to_numpy()
    cost = samples['cost_fuel'].to_numpy() 
    price = samples['price'].to_numpy()
    structural = samples['outcome'].to_numpy() - samples['noise_demand'].to_numpy()
    outcome = samples['outcome'].to_numpy()

    treatment: np.ndarray = price[:, np.newaxis]
    covariate: np.ndarray = np.c_[time, emotion]
    instrumental: np.ndarray = np.c_[cost, time, emotion]
    train_data = TrainDataSet(treatment=treatment,
                              instrumental=instrumental,
                              covariate=covariate,
                              outcome=outcome[:, np.newaxis],
                              structural=structural[:, np.newaxis])
    return train_data
