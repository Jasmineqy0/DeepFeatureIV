from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import logging
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
from src.data.params.ops import add_data_param, revise_dump_name, revise_parcs_config


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
            group_name: str):
    # if len(dict(wandb.config)) == 0:

    wandb.init(project="DeepFeatureIV", group=group_name,
                config={**data_param, **train_config, 'experiment_id': experiment_id})
    
    # else:
        # wandb.init(group='sweep')
        # wandb.config.update({**data_param,  'experiment_id': experiment_id})
    
    Trainer_cls = get_trainer(alg_name)
    one_dump_dir = None
    if dump_dir_root is not None:
        one_dump_dir = dump_dir_root.joinpath(f"{experiment_id}/")
        os.makedirs(one_dump_dir, exist_ok=True)
    trainer = Trainer_cls(data_param, wandb.config, use_gpu, one_dump_dir)

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
    logging.info(f"using gpu: {use_gpu}")

    # wandb login
    wandb.login(key = os.getenv('wandb_key')) 
    # add data params for fully randomized iv data
    add_data_param(org_data_config)

    for dump_name, data_param in grid_search_dict(org_data_config):
        # revise dump_name for fully randomized iv data
        dump_name = revise_dump_name(data_param, dump_name)
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.makedirs(one_dump_dir, exist_ok=True)

        # revise data_param for some parcs config
        revise_parcs_config(data_param)

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