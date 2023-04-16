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


def generate_test_spaceiv(spaceiv_datadir, case, rand_seed) -> TestDataSet:
    data_size = int(os.getenv('spaceiv_test_size'))

    case_dir = os.path.join(spaceiv_datadir, case)

    sample_dir = get_sample_dir(case_dir, rand_seed)

    _, treatments, covariates, structurals, _ = get_vars(sample_dir, data_size)

    return TestDataSet(treatment=treatments,
                       covariate=covariates,
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

    data = get_data(sample_dir, data_size)
    parents = get_parents(sample_dir)

    data_columns = data.columns
    instruments = data[[col for col in data_columns if 'i' in col.lower()]].to_numpy()
    treatments = data[parents].to_numpy()
    covariates = data[[col for col in data_columns if 'x' in col.lower() and col not in parents]].to_numpy()
    outcome = data[outcome_key].to_numpy()[:, np.newaxis]

    config_file = os.path.join(sample_dir, 'config.yml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    y_desc = re.search(r'\(.*=(.*),.*\)', config[outcome_key]).group(1)
    biases, vars = parse_terms(y_desc)
    structurals = [coef * data[var].to_numpy() for var, coef in vars.items()]
    structurals = np.stack(structurals, axis=-1)
    structurals += sum(biases)

    return instruments, treatments, covariates, structurals, outcome


def generate_train_spaceiv(spaceiv_datadir, case, rand_seed, data_size: int) -> TrainDataSet:

    """
    Parameters
    ----------
    data_size : int
        size of data

    Returns
    -------
    train_data : TrainDataSet
    """

    case_dir = os.path.join(spaceiv_datadir, case)

    sample_dir = get_sample_dir(case_dir, rand_seed)

    instruments, treatments, covariates, structurals, outcome = get_vars(sample_dir, data_size)
    
    return TrainDataSet(treatment=treatments,
                        instrumental=instruments,
                        covariate=covariates,
                        structural=structurals,
                        outcome=outcome)
