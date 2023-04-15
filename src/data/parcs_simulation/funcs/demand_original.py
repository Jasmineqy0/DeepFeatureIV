import numpy as np

def phi_t(data):
    return 2 * ((data['time'] - 5) ** 4 / 600 + np.exp (-4 * (data['time']- 5) ** 2 ) + data['time']/10 - 2)

def price(data):
    return 25 + (data['cost_fuel'] + 3) * data['phi_t'] + data['noise_price']

def outcome(data):
    return 100 + (10 + data['price']) * data['emotion'] * data['phi_t'] - 2 * data['price'] + data['noise_demand']

def emotion(data):
    return np.random.choice(7, len(data), replace=True) + 1

def noise_demand(data):
    return np.random.normal(data['rho']*data['noise_price'], 1-data['rho']**2, size=len(data))