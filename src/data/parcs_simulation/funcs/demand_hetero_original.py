import numpy as np
from src.data.demand_design import psi, f

FUNCTION = 'original'

def price(data):
    return 25 + (data['cost_fuel'] + 3) * psi(FUNCTION, data['time']) + data['noise_price']

def outcome(data):
    t = data['time']
    s = data['emotion']
    p = data['price']
    return f(FUNCTION, p, t, s) + data['noise_demand']

def emotion(data):
    return np.random.choice(7, len(data), replace=True) + 1


def noise_demand(data):
    scaled_noise_price = 1 / (1 + np.exp(-data['noise_price']))
    return np.random.normal(data['rho']*data['noise_price'], scaled_noise_price * np.sqrt((1-data['rho']**2)), size=len(data))