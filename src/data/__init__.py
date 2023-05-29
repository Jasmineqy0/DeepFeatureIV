from typing import Tuple, NamedTuple, Optional
import numpy as np

from .demand_design_image import generate_test_demand_design_image, generate_train_demand_design_image
from .demand_design import generate_test_demand_design, generate_train_demand_design
from .demand_design_test import generate_train_demand_design_test, generate_test_demand_design_test
from .demand_design_parcs import generate_train_demand_design_parcs, generate_test_demand_design_parcs
from .dsprine import generate_train_dsprite, generate_test_dsprite
from .spaceiv import generate_train_spaceiv, generate_test_spaceiv
from .data_class import TrainDataSet, TestDataSet
from .fully_random_iv import generate_train_fully_random_iv, generate_test_fully_random_iv


def generate_train_data(data_name: str, rand_seed: int, validation: bool, **args) -> TrainDataSet:
    if data_name == "demand":
        return generate_train_demand_design(args["data_size"], args["rho"], args['function'], 
                                            args['noise_price_mean'], args['noise_price_std'], args['hetero'], args['uniform_iv'],
                                            rand_seed, False)
    elif data_name == 'demand_test':
        return generate_train_demand_design_test(args["data_size"], args['function'], args['noise_price_mean'], args['noise_price_std'],
                                                 rand_seed, False)
    elif data_name == "demand_parcs":
        return generate_train_demand_design_parcs(args['data_size'], args['function'], args['hetero'], args['noise_price_bias'], rand_seed=rand_seed)
    elif data_name == "demand_old":
        # Demand design for no covariate (deprecated)
        return generate_train_demand_design(rand_seed=rand_seed, old_flg=True, **args)
    elif data_name == "demand_image":
        return generate_train_demand_design_image(args["data_size"], args["rho"], rand_seed)
    elif data_name == "dsprite":
        return generate_train_dsprite(args["data_size"], rand_seed)
    elif data_name == 'spaceiv':
        return generate_train_spaceiv(rand_seed=rand_seed, validation=validation, **args)
    elif data_name == 'fully_random_iv':
        return generate_train_fully_random_iv(args['simulation_info'], args["data_size"], rand_seed=rand_seed)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_name: str, rand_seed, **args) -> TestDataSet:
    if data_name == "demand":
        return generate_test_demand_design(args['function'], False)
    elif data_name == 'demand_test':
        return generate_test_demand_design_test(args['function'], args['noise_price_mean'], False)
    elif data_name == "demand_parcs":
        return generate_test_demand_design_parcs(args['function'], args['noise_price_bias'])
    elif data_name == "demand_old":
        # Demand design for no covariate (deprecated)
        return generate_test_demand_design(True)
    elif data_name == "demand_image":
        return generate_test_demand_design_image()
    elif data_name == "dsprite":
        return generate_test_dsprite()
    elif data_name == 'spaceiv':
        return generate_test_spaceiv(rand_seed=rand_seed, **args)
    elif data_name == 'fully_random_iv':
        return generate_test_fully_random_iv(args['simulation_info'])
    else:
        raise ValueError(f"data name {data_name} is not valid")

