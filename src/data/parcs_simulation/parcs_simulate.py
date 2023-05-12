import logging
import os

logger = logging.getLogger()

def get_config_file(config_name: str):
    parcs_config_dir = os.sep.join(['src', 'data', 'parcs_simulation', 'configs'])
    parcs_config_path = os.path.join(parcs_config_dir, f'{config_name}.yml')

    return parcs_config_path