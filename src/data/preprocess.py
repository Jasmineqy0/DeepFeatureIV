import numpy as np

def rescale_treatment(treatment, data_name: str):
    if data_name in ["demand", "demand_image"]:
        psd = 3.7
        pmu = 17.779
        return (treatment - pmu) / psd
    else:
        return treatment


def rescale_outcome(outcome, data_name: str):
    if data_name in ["demand", "demand_image"]:
        ysd = 158
        ymu = -292.1
        return (outcome - ymu) / ysd
    else:
        return outcome


def inv_rescale_outcome(predict, data_name: str):
    if data_name in ["demand", "demand_image"]:
        ysd = 158
        ymu = -292.1
        return (predict * ysd) + ymu
    else:
        return predict
    
def rescale_t(time):
    tmu = 5
    tsd = np.sqrt(10 ** 2 / 12)
    t_normalized = (time - tmu) / tsd
    return 1/(1 + np.exp(-t_normalized))
