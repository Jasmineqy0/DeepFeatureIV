import numpy as np

def phi_t(data):
    return 2 * ((data['time'] - 5) ** 4 / 600 + np.exp (-4 * (data['time']- 5) ** 2 ) + data['time']/10 - 2)

def price(data):
    return 25 + (data['cost_fuel'] + 3) * data['phi_t'] + data['noise_price']

def outcome(data):
    return 100 + (10 + data['price']) * data['emotion'] * data['phi_t'] - 2 * data['price'] + data['noise_demand']

def emotion(data):
    return np.random.choice(7, len(data), replace=True) + 1

def rescale_p(price):
    min_p, max_p = 1.78, 28.5
    return (price - min_p) / (max_p - min_p)

def noise_demand(data):
    rescaled_price = rescale_p(data['price'])
    assert  np.all(rescaled_price <= 1) and np.all(rescaled_price >= 0), 'rescaled price is not in [0, 1]'
    return np.random.normal(data['rho']*data['noise_price'], rescaled_price * data['sigma'], size=len(data))