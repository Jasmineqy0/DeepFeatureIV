import numpy as np
from src.data.demand_design import psi, f


def price(data):
    return 25 + (data['cost_fuel'] + 3) * psi(data['time']) + data['noise_price']

def outcome(data):
    t = data['time']
    s = data['emotion']
    p = data['price']
    return f(p, t, s) + data['noise_demand']

def emotion(data):
    return np.random.choice(7, len(data), replace=True) + 1

def rescale_p(price):
    psd = 3.7
    pmu = 17.779
    p_normalized = (price - pmu) / psd
    return 1/(1 + np.exp(-p_normalized))

def noise_demand(data):
    rescaled_price = rescale_p(data['price'])
    assert  np.all(rescaled_price <= 1) and np.all(rescaled_price >= 0), 'rescaled price is not in [0, 1]'
    return np.random.normal(data['rho']*data['noise_price'], rescaled_price * data['sigma'], size=len(data))