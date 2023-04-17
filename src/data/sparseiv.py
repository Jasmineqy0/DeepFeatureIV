import numpy as np
from numpy.random import default_rng
from pathlib import Path
import os
from glob import glob
import re
import yaml
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os

from ..data.data_class import TrainDataSet, TestDataSet


def generate_test_sparseiv(case_dir, rand_seed) -> TestDataSet:
    data_size = int(os.getenv('sparseiv_test_size'))

    sample_dir = get_sample_dir(case_dir, rand_seed)

    _, treatments, structurals, _ = get_vars(sample_dir, data_size)

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
    outcome_key = 'y_0'
    x_prefix = 'x'
    i_prefix = 'i'

    beta_star_file = f'beta_star_{data_size}.npy'

    data = get_data(sample_dir, data_size)

    data_columns = data.columns
    instruments = data[[col for col in data_columns if i_prefix in col.lower()]].to_numpy()
    treatments = data[[col for col in data_columns if x_prefix in col.lower()]].to_numpy()
    outcome = data[outcome_key].to_numpy()[:, np.newaxis]

    # config_file = os.path.join(sample_dir, 'config.yml')
    # with open(config_file, 'r') as f:
    #     config = yaml.safe_load(f)
    # y_desc = re.search(r'\(.*=(.*),.*\)', config[outcome_key]).group(1)
    # biases, vars = parse_terms(y_desc)
    # structurals = [coef * data[var].to_numpy() for var, coef in vars.items()]
    # structurals = np.stack(structurals, axis=-1)
    # structurals += sum(biases)
    
    beta_star = np.load(os.path.join(sample_dir, beta_star_file))
    structurals = treatments @ beta_star
    
    return instruments, treatments, structurals, outcome


def generate_train_sparseiv(case_dir:  str, rand_seed: int, data_size: int, val_size: int, validation: bool) -> TrainDataSet:

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
