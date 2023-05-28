from itertools import product
import numpy as np
from numpy.random import default_rng
import logging

from ..data.data_class import TrainDataSet, TestDataSet

np.random.seed(42)
logger = logging.getLogger()


def f(function: str, p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    if function == 'original':
        res = f_original(p, t, s)
    elif function == 'revised':
        res = f_revised(p, t, s)
    else:
        raise ValueError(f'function {function} is not valid')
    return res

def psi(function: str, t: np.ndarray) -> np.ndarray:
    if function == 'original':
        res = psi_original(t)
    elif function == 'revised':
        res = psi_revised(t)
    else:
        raise ValueError(f'function {function} is not valid')
    return res
        
def psi_revised(t: np.ndarray) -> np.ndarray:
    return 2 * (((t - 3) ** 3 )/ 500 + np.exp(-6 * ((t-5) ** 2) ) - np.sqrt(t) + np.log(25 * (t ** 2) + 5) + np.sin(t)  - 7)

def f_revised(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 125 + 20 * np.log2(s) * psi_revised(t) - 1.5 * s + (s * psi_revised(t) - s ** 0.5) * p

def psi_original(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

def f_original(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100 + (10 + p) * s * psi_original(t) - 2 * p


def generate_test_demand_design_test(function: str = 'original', 
                                     noise_price_mean: float = 0.0,
                                     old_flg: bool = False) -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    # obtain the approx range of the expected price
    boostrap_size = 5000
    time = np.linspace(0.0, 10, boostrap_size)
    # E[C] = 0, E[V] = noise_price_mean -> E[P | t] = 25 + (0 + 3) * psi(t) + noise_price_mean
    exp_true_price =  25 + (0 + 3) * psi(function, time) + noise_price_mean
    exp_min_price, exp_max_price = np.round(np.min(exp_true_price)), np.round(np.max(exp_true_price))
    
    # intervene within the range of the expected price
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
    if old_flg:
        test_data = TestDataSet(treatment=features,
                                structural=targets,
                                covariate=None)
    else:
        test_data = TestDataSet(treatment=features[:, 0:1],
                                covariate=features[:, 1:],
                                structural=targets)
    return test_data


def generate_train_demand_design_test(data_size: int,
                                      function: str = 'original',
                                      noise_price_mean: float = 0.0,
                                      noise_price_std: float = 1.0,
                                      rand_seed: int = 42,
                                      old_flg: bool = False) -> TrainDataSet:
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

    rng = default_rng(seed=rand_seed)
    emotion = rng.choice(list(range(1, 8)), data_size)
    time = rng.uniform(0, 10, data_size)
    cost = rng.normal(0, 1.0, data_size)
    noise_price = rng.normal(noise_price_mean, noise_price_std, data_size)
    noise_demand = rng.normal(0, 1, data_size)
    price = 25 + (cost + 3) * psi(function, time) + noise_price
    structural: np.ndarray = f(function, price, time, emotion).astype(float)
    outcome: np.ndarray = (structural + noise_demand + noise_price).astype(float)
    if old_flg:
        treatment = np.c_[price, time, emotion]
        instrumental = np.c_[cost, time, emotion]
        train_data = TrainDataSet(treatment=treatment,
                                  instrumental=instrumental,
                                  covariate=None,
                                  outcome=outcome[:, np.newaxis],
                                  structural=structural[:, np.newaxis])
    else:
        treatment: np.ndarray = price[:, np.newaxis]
        covariate: np.ndarray = np.c_[time, emotion]
        instrumental: np.ndarray = np.c_[cost, time, emotion]
        train_data = TrainDataSet(treatment=treatment,
                                  instrumental=instrumental,
                                  covariate=covariate,
                                  outcome=outcome[:, np.newaxis],
                                  structural=structural[:, np.newaxis])
    return train_data