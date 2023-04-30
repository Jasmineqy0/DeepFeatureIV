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

def noise_demand(data):
    return np.random.normal(data['rho']*data['noise_price'], 1-data['rho']**2, size=len(data))