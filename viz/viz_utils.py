from pathlib import Path
from typing import Union
import plotly.express as px
import numpy as np
import pandas as pd
import numpy as np
import wandb
import torch
from torch import nn
import sys
sys.path.append(str(Path.cwd().parent))
from src.models.DFIV.model import DFIVModel
from src.models.DFIV.nn_structure import build_extractor

def load_dfiv_runs(entity, project, filters=None):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project, filters=filters) 

    summary_list, config_list, name_list = [], [], []
    id_list, state_list = [], []
    for run in runs: 
        # .summary contains output keys/values for 
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        
        # get run id
        id_list.append(run.id)
        
        # get run state
        state_list.append(run.state)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        'id': id_list,
        'state': state_list
        })

    return runs_df

def load_dfiv_result(run_path):
    result_file = wandb.restore('result.npz', run_path=run_path)
    return np.load(result_file.name)

def load_dfiv_model(run_path, data_config_ops=None):
    # delete model.pth if exists
    if Path('model.pth').exists():
        Path('model.pth').unlink()
    # load and extract weights from saved model file
    best_model = wandb.restore('model.pth', run_path=run_path)
    checkpoint = torch.load(best_model.name, map_location=torch.device('cpu'))
    treatment_net_weight = checkpoint['treatment_net']
    covariate_net_weight = checkpoint['covariate_net'] if 'covariate_net' in checkpoint else None
    stage_2_weight = checkpoint['stage_2_weight']

    api = wandb.Api()
    run = api.run(run_path)

    # get run configs
    model_configs = run.config.get('model_configs')
    data_configs = run.config.get('data_configs')
    data_configs['model_configs'] = model_configs
    if data_config_ops:
        data_configs = {**data_configs, **data_config_ops}

    # build treatment net
    networks = build_extractor(**data_configs)
    treatment_net = networks[0]

    # load weights for treatment net
    treatment_net.load_state_dict(treatment_net_weight)
    if covariate_net_weight is not None:
        covariate_net = networks[2]
        covariate_net.load_state_dict(covariate_net_weight)
    else:
        covariate_net = None

    # build DFIV model and pass treatment net
    mdl = DFIVModel(treatment_net, instrumental_net=None, covariate_net=covariate_net,
                            add_stage1_intercept=True, add_stage2_intercept=True, wandb_log=False)

    # set stage 2 weight
    mdl.stage2_weight = stage_2_weight
    return mdl

def predict_dfiv_model(mdl, treatment: np.ndarray, covariate: Union[None, np.ndarray]):
    treatment = torch.from_numpy(treatment).to(torch.float32)
    if covariate is not None:
        covariate = torch.from_numpy(covariate).to(torch.float32)
    # predict
    return mdl.predict_t(treatment=treatment, covariate=covariate).detach().numpy()