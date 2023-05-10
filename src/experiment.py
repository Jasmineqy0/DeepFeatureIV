from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
from simulator.full_random_simulation import randomize_once, FINAL_CONFIG
import logging
import torch
import shutil
from glob import glob
import wandb
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from src.utils import grid_search_dict
from src.models.DFIV.trainer import DFIVTrainer
from src.models.DeepIV.trainer import DeepIVTrainer
from src.models.KernelIV.trainer import KernelIVTrainer
from src.models.RFFKernelIV.trainer import RFFKernelIVTrainer
from src.models.DeepGMM.trainer import DeepGMMTrainer
from src.data.parcs_simulation.parcs_simulate import revise_parcs_config


logger = logging.getLogger()

def get_trainer(alg_name: str):
    if alg_name == "DeepIV":
        return DeepIVTrainer
    elif alg_name == "DFIV":
        return DFIVTrainer
    elif alg_name == "KernelIV":
        return KernelIVTrainer
    elif alg_name == "RFFKernelIV":
        return RFFKernelIVTrainer
    elif alg_name == "deepGMM":
        return DeepGMMTrainer
    else:
        raise ValueError(f"invalid algorithm name {alg_name}")


def run_one(alg_name: str, data_param: Dict[str, Any], train_config: Dict[str, Any],
            use_gpu: bool, dump_dir_root: Optional[Path], experiment_id: int, verbose: int,
            dump_dir: str):
    Trainer_cls = get_trainer(alg_name)
    one_dump_dir = None
    if dump_dir_root is not None:
        one_dump_dir = dump_dir_root.joinpath(f"{experiment_id}/")
        os.makedirs(one_dump_dir, exist_ok=True)
    trainer = Trainer_cls(data_param, train_config, use_gpu, one_dump_dir, train_config['wandb'])

    # set up wandb logging
    if 'wandb' in train_config and train_config['wandb']:
        wandb.login(key = os.getenv('wandb_key'))
        config = {**data_param, **train_config}
        wandb.init(project=f'{alg_name}', group=dump_dir, config=config)

    return trainer.train(experiment_id, verbose)


def experiments(alg_name: str,
                configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int, num_gpu: Optional[int]):
    train_config = configs["train_params"]
    org_data_config = configs["data"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    use_gpu: bool = (num_gpu is not None)

    if org_data_config['data_name'] == 'fully_random_iv':
        config_info = []
        for i in range(org_data_config['fully_random_num']):
            _, final_config_path, var_info_path = randomize_once(org_data_config['guideline_path'], 1000, org_data_config['bootsrap_seed'])
            config_dir = final_config_path.parent / str(i)
            config_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(final_config_path, config_dir / final_config_path.name)
            shutil.copy(var_info_path, config_dir / var_info_path.name)
            config_info.append({'config_path': str(config_dir / final_config_path.name), 'var_info_path': str(config_dir / var_info_path.name)})
        org_data_config['config_info'] = config_info

        guideline_path = Path(org_data_config['guideline_path'])
        for file in (guideline_path.parent).glob('*.yml'):
            if file.name != guideline_path.name:
                file.unlink()
    
    for dump_name, data_param in grid_search_dict(org_data_config):
        if data_param['data_name'] == 'fully_random_iv':
            config_idx = data_param['bootsrap_seed']
            dict_idx = dump_name.index('{')
            dump_name = dump_name[:dict_idx] + str(config_idx)
            
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.makedirs(one_dump_dir, exist_ok=True)

        # revise parcs' config for different rho and sigma in demand simulation
        if data_param['data_name'] == 'demand':
            if 'parcs' in data_param and data_param['parcs']:
                assert all(key in data_param for key in ['rho', 'sigma', 'parcs_config']), f'error: parcs simulation requires rho, sigma and parcs_config'
                revise_parcs_config(data_param['rho'], data_param['sigma'], data_param['parcs_config'])

        tasks = [run_one(alg_name, data_param, train_config,
                                use_gpu, one_dump_dir, idx, verbose, dump_dir.name) for idx in range(n_repeat)]
        
        # save treatment, covariate, prediction & oos_loss
        res_new = defaultdict(list)
        for item in tasks:
            for key in item.keys():
                res_new[key].append(item[key])
        res_new = {key: np.array(res_new[key]) for key in res_new.keys()}
        np.savez(one_dump_dir.joinpath("result.npz"), **res_new)

        logger.critical(f"{dump_name} ended")