import json
from pathlib import Path
import shutil
import logging
import wandb
import yaml

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')

SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
CONFIGS_DIR = Path(__file__).parent.parent / 'final_configs'
PARCS_SIMULATION_GUIDELINE_FILE = 'simulation.yml'
DFIV_CONFIG_FILE = 'configs.json'
WANDB_SWEEP_FILE = 'sweep.yml'
WANDB_ID = 'jasmineqy0'
SLURM_SHELL_TEMPLATE = 'dfiv_vanilla.sh'

config_type = 'dfiv_fully_random_interaction_10x'
config_root_dir = Path(f'{config_type}_0')
wandb_exp_count = 300
dup = 4


for i in range(1, dup+1):
    # destination dir
    config_dup_dir = Path(f'{config_type}_{i}')
    logging.info(f'Simulating config {config_dup_dir} from {str(config_root_dir)}')
    
    if not config_dup_dir.exists():
        shutil.copytree(config_root_dir, config_dup_dir)
    
        # delete simulated parcs config in original config dir
        for file in config_dup_dir.glob('*/*'):
            file.unlink()
        
        # read and later change configs.json
        with open(config_dup_dir / DFIV_CONFIG_FILE, 'r') as f:
            dfiv_configs = json.load(f)
        # change bootstrap seed
        dfiv_configs['data_configs']['bootstrap_seed'] = i
        # change guideline path (path w.r.t. the root folder of the project)
        dfiv_configs['data_configs']['guideline_path'] = f'{CONFIGS_DIR}/{str(config_dup_dir)}/{PARCS_SIMULATION_GUIDELINE_FILE}'
        # save configs.json
        json.dump(dfiv_configs, open(config_dup_dir / DFIV_CONFIG_FILE, 'w'), indent=4)
        
        # read and later change sweep.yml for wandb hyperparameter search
        with open(config_dup_dir / WANDB_SWEEP_FILE, 'r') as f:
            sweep_configs = yaml.safe_load(f)
        # change bootstrap seed
        sweep_configs['parameters']['data_configs']['parameters']['bootstrap_seed']['value'] = i
        # change guideline path (path w.r.t. the root folder of the project)
        sweep_configs['parameters']['data_configs']['parameters']['guideline_path']['value'] = f'{CONFIGS_DIR}/{str(config_dup_dir)}/{PARCS_SIMULATION_GUIDELINE_FILE}'
        # change dfiv config path for command
        sweep_configs['command'][2] = sweep_configs['command'][2].replace(str(config_root_dir), str(config_dup_dir))
        # save sweep.yml
        with open(config_dup_dir / WANDB_SWEEP_FILE, 'w') as f:
            yaml.safe_dump(sweep_configs, f, default_flow_style=False, sort_keys=False)

        # create sweep
        sweep_id = wandb.sweep(sweep=sweep_configs, project=config_type)
        logging.info(f'Created sweep {sweep_id} for config {config_dup_dir}')
        
        # copy slurm shell script
        slurm_template_script = Path(SCRIPTS_DIR) / SLURM_SHELL_TEMPLATE
        slurm_exp_script = Path(SCRIPTS_DIR) /  f'{str(config_dup_dir)}.sh'
        shutil.copyfile(slurm_template_script, slurm_exp_script)
        # append wandb sweep command and slurm command to slurm shell script
        sweep_command = f'wandb agent --count {wandb_exp_count} {WANDB_ID}/{config_type}/{sweep_id}'
        slurn_command = f'# sbatch --gres=gpu:1 {SCRIPTS_DIR}/{str(slurm_exp_script)}'
        append_text = f'\n\n{sweep_command}\n{slurn_command}'
        with open(slurm_exp_script, 'a') as f:
            f.write(append_text)
        logging.info(f'Appended wandb sweep command and slurm command to {slurm_exp_script}')
        
    