import numpy as np
from src.data.demand_design_parcs_revise import psi, f
from src.data.preprocess import rescale_t


def price(data):
    return 25 + (data['cost_fuel'] + 3) * psi(data['time']) + data['noise_price']

def outcome(data):
    t = data['time']
    s = data['emotion']
    p = data['price']
    return f(p, t, s) + data['noise_demand']

def emotion(data):
    return np.random.choice(7, len(data), replace=True) + 1


def noise_demand(data):
    rescaled_time = rescale_t(data['time'])
    assert  np.all(rescaled_time <= 1) and np.all(rescaled_time >= 0), 'rescaled time is not in [0, 1]'
    return np.random.normal(data['rho']*data['noise_price'], rescaled_time * data['sigma'], size=len(data))