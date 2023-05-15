from itertools import product
import numpy as np
import logging
from typing import  Union
from pathlib import Path

from .data_class import TrainDataSet, TestDataSet
from .parcs_simulation.parcs_simulate import get_config_file

from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser

np.random.seed(42)
logger = logging.getLogger()

TEST_SEED = 9999

def psi(t: np.ndarray) -> np.ndarray:
    return 2 * (((t - 3) ** 3 )/ 500 + np.exp(-6 * ((t-5) ** 2) ) - np.sqrt(t) + np.log(25 * (t ** 2) + 5) + np.sin(t)  - 7)

def f(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    # original f: 100 + 10 * s * psi(t) - (s * psi(t) - 2) * p**3
    return 125 + 20 * np.log2(s) * psi(t) - 1.5 * s + (s * psi(t) - s ** 0.5) * p

def generate_test_demand_design_parcs_revise(parcs_config: Union[Path, str]) -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    
    # obtain the graph
    config_yml = get_config_file(parcs_config)
    assert Path(config_yml).exists(), f'error: parcs config file {config_yml} does not exist.'
    
    # parse the config file
    nodes, edges = graph_file_parser(config_yml)
    g = Graph(nodes=nodes, edges=edges)
    
    # original test data for demand design with size 2800
    price = np.linspace(10, 25, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    
    data, target = [], []
    for p, t, s in product(price, time, emotion):
        # considering that time and emotion have no parents in the graph, we can do joint intervention
        intervention = {'price': p, 'time': t, 'emotion': s}
        samples = g.do(size=1, interventions=intervention)
        data.append([samples['price'][0], samples['time'][0], samples['emotion'][0]])
        target.append(f(samples['price'][0], samples['time'][0], samples['emotion'][0]))
    
    features = np.array(data)
    targets: np.ndarray = np.array(target)[:, np.newaxis]
    
    test_data = TestDataSet(treatment=features[:, 0:1],
                            covariate=features[:, 1:],
                            structural=targets)
    return test_data


def generate_train_demand_design_parcs_revise(data_size: int,
                                              parcs_config: str,
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
