import numpy as np

def phi_t(data):
    return 2 * ((data['time'] - 5) ** 4 / 600 + np.exp (-4 * (data['time']- 5) ** 2 ) + data['time']/10 - 2)

def price(data):
    return 25 + (data['cost_fuel'] + 3) * data['phi_t'] + data['noise_price']

def outcome(data):
    return 100 + (10 + data['price']) * data['emotion'] * data['phi_t'] - 2 * data['price'] + data['noise_demand']

def emotion(data):
    return np.random.choice(7, len(data), replace=True) + 1

def rescale_t(time):
    tmu = 5
    tsd = np.sqrt(10 ** 2 / 12)
    t_normalized = (time - tmu) / tsd
    return 1/(1 + np.exp(-t_normalized))

def noise_price(data):
    rescaled_time = rescale_t(data['time'])
    assert  np.all(rescaled_time <= 1) and np.all(rescaled_time >= 0), 'rescaled time is not in [0, 1]'
    return np.random.normal(0, rescaled_time * data['sigma'], size=len(data))

def noise_demand(data):
    return np.random.normal(data['rho']*data['noise_price'], 1-data['rho']**2, size=len(data))