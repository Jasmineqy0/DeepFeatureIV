from pathlib import Path
import shutil
from typing import Any, Dict, Union

import yaml
from simulator.full_random_simulation import randomize_once
from src.data.parcs_simulation.parcs_simulate import get_config_file
import re
import logging

logger = logging.getLogger()

CONFIG_FILE = 'sweep.yml'

BOOTSTRAP_SIZE = 5000

def get_parcs_config_name(hetero: bool, function: str):
    hetero_str = '_hetero' if hetero else ''
    if function == 'original':
        config_name = f'demand{hetero_str}_original'
    elif function == 'revised':
        config_name = f'demand{hetero_str}_revised'
    else:
        raise ValueError(f'error: function {function} is not valid.')
    return config_name

def generate_data_config_info(guideline_path, fully_random_num, bootsrap_seed):
    config_file, var_info_file = 'config.yml', 'var_info.yml'
    guideline_path = Path(guideline_path)
    dfiv_config_dir = guideline_path.parent
        
    data_config_info = []
        
    for i in range(fully_random_num):
        data_config_dir = dfiv_config_dir / str(i)
        data_config_path = data_config_dir / config_file
        data_var_info_path = data_config_dir / var_info_file

        if not data_config_path.exists() or not data_var_info_path.exists():
            _, tmp_config_path, tmp_var_info_path = randomize_once(guideline_path, BOOTSTRAP_SIZE, bootsrap_seed)
            data_config_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(tmp_config_path, data_config_path)
            shutil.copy(tmp_var_info_path, data_var_info_path)

        data_config_info.append({'config_path': data_config_path, 
                                 'var_info_path': data_var_info_path})
        
    for file in dfiv_config_dir.glob('*.yml'):
        if file.name not in [guideline_path.name, CONFIG_FILE]:
            file.unlink()

    return data_config_info


def add_data_param(data_config: Dict[str, Any]):
        if data_config['data_name'] == 'fully_random_iv':
            data_config['simulation_info'] = generate_data_config_info(data_config['guideline_path'], 
                                                                       data_config['fully_random_num'],
                                                                       data_config['bootstrap_seed'])
        else:
            data_config['simulation_info'] = None

def revise_dump_name(data_param, dump_name):
    if data_param['data_name'] == 'fully_random_iv':
        seed_str = f"seed:{data_param['bootstrap_seed']}"
        guideline_str = f"guideline:{Path(data_param['guideline_path']).stem}"
        match = re.search(r'simulation_info:{.*}', dump_name)
        start_idx, end_idx = match.start(), match.end()
        dump_name = dump_name[:start_idx] + seed_str + '-'+ guideline_str + dump_name[end_idx:]

    return dump_name

def constant_rho_sigma(rho: float, sigma: Union[None, float], config_name: str):
    config_yml = get_config_file(config_name)

    with open(config_yml, 'r') as f:
        parcs_config = yaml.safe_load(f)
        parcs_config['rho'] = f'constant({rho})'
        if sigma is not None:
            parcs_config['sigma'] = f'constant({sigma})'

    with open(config_yml, 'w') as f:
        yaml.dump(parcs_config, f)
        
    logger.info(f'Revised parcs config file with rho={rho} and sigma={sigma}.')

def revise_parcs_config(data_param):
     # revise parcs' config for different rho and sigma in demand simulation
    if data_param['data_name'] == 'demand_parcs':
            config_name = get_parcs_config_name(data_param['hetero'], data_param['function'])
            constant_rho_sigma(data_param['rho'], data_param['sigma'], config_name)