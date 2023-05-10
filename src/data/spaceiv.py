import numpy as np
from numpy.random import default_rng
from pathlib import Path
import os
from glob import glob
import re
import yaml
import pandas as pd
import os

from ..data.data_class import TrainDataSet, TestDataSet


def generate_test_spaceiv(spaceiv_datadir, case, test_size, rand_seed, **kwargs) -> TestDataSet:
    case_dir = os.path.join(spaceiv_datadir, case)

    sample_dir = get_sample_dir(case_dir, rand_seed)

    _, treatments, structurals, _ = get_vars(sample_dir, test_size)

    return TestDataSet(treatment=treatments,
                       covariate=None,
                       structural=structurals)

def get_sample_dir(case_dir, rand_seed):
    sample_dir = [dir for dir in os.listdir(case_dir) if re.search(f'.*-{rand_seed}', dir)]
    sample_dir = sample_dir[0]
    sample_dir = os.path.join(case_dir, sample_dir)

    return sample_dir

def parse_terms(eq_str):
    var_pattern = r"([a-zA-Z0-9_]+)"
    coef_pattern = r"([-+]?)(\d*\.?\d+)*"

    pairs = re.findall(f"{coef_pattern}{var_pattern}", eq_str)

    biases, vars = [], {}
    for sign, coef, var in pairs:
        if not coef:
            if var.isnumeric():
                biases.append(float(var))
            else:
                vars[var] = 1 if not sign or sign == '+' else -1
        else:
            vars[var] = float(coef) if not sign or sign == '+' else -float(coef)

    return biases, vars

def get_data(sample_dir, data_size):
    data_file = os.path.join(sample_dir, f'data_size{data_size}.csv')
    with open(data_file, 'r') as f:
        data = pd.read_csv(f)
    return data

def get_parents(sample_dir):
    parents_file = 'parents.yml'
    parents_key = 'causal_parents'
    parent_file = os.path.join(sample_dir, parents_file)
    with open(parent_file, 'r') as f:
        parents = yaml.safe_load(f)[parents_key]
    return parents

def get_vars(sample_dir, data_size):
    size_dir = f'data_size:{data_size}'
    
    treatments = np.loadtxt(os.path.join(sample_dir, size_dir, 'X.csv'), delimiter=',')
    outcome = np.loadtxt(os.path.join(sample_dir, size_dir, 'Y.csv'), delimiter=',').reshape(-1, 1)
    instruments = np.loadtxt(os.path.join(sample_dir, size_dir, 'I.csv'), delimiter=',')
    beta_star = np.loadtxt(os.path.join(sample_dir, size_dir, 'beta_star.csv'), delimiter=',')
    structurals = treatments @ beta_star
    
    return instruments, treatments, structurals, outcome


def generate_train_spaceiv(spaceiv_datadir:  str, case: str, rand_seed: int, data_size: int, val_size: int, validation: bool, **kwargs) -> TrainDataSet:

    """
    Parameters
    ----------
    sparseiv_datadir : str, Path to the directory containing the sparseiv data
    case : str, Name of the case
    rand_seed : int, Random seed
    data_size : int, Size of the data
    validation : bool, Whether to generate validation data

    Returns
    -------
    train_data : TrainDataSet
    """
    case_dir = os.path.join(spaceiv_datadir, case)
    sample_dir = get_sample_dir(case_dir, rand_seed)

    instruments, treatments, structurals, outcome = get_vars(sample_dir, data_size)
    
    if validation:
        instruments = instruments[-val_size:]
        treatments = treatments[-val_size:]
        structurals = structurals[-val_size:]
        outcome = outcome[-val_size:]
    else:
        instruments = instruments[:-val_size]
        treatments = treatments[:-val_size]
        structurals = structurals[:-val_size]
        outcome = outcome[:-val_size]
    
    return TrainDataSet(treatment=treatments,
                        instrumental=instruments,
                        covariate=None,
                        structural=structurals,
                        outcome=outcome)
