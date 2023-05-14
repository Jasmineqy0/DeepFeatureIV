from typing import Tuple, Optional

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .nn_structure_for_demand import build_net_for_demand
from .nn_structure_for_demand_old import build_net_for_demand_old
from .nn_structure_for_demand_image import build_net_for_demand_image
from .nn_structure_for_dsprite import build_net_for_dsprite
from .nn_structure_for_spaceiv import build_net_for_spaceiv
from .nn_structure_for_fully_random_iv import build_net_for_fully_random_iv
from .nn_structure_for_demand_revise import build_net_for_demand_revise

import logging

logger = logging.getLogger()


def build_extractor(data_name: str, **args) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    if data_name == "demand" or data_name == "demand_parcs":
        logger.info("build without image")
        return build_net_for_demand()
    if data_name == "demand_parcs_revise":
        logger.info("build without image")
        return build_net_for_demand_revise()
    elif data_name == "demand_image":
        logger.info("build with image")
        return build_net_for_demand_image()
    elif data_name == "demand_old":
        logger.info("build old model without image")
        return build_net_for_demand_old()
    elif data_name == "dsprite":
        logger.info("build dsprite model")
        return build_net_for_dsprite()
    elif data_name == 'spaceiv':
        assert all(key in args for key in ['dim_iv', 'dim_ts']), "dim_iv and dim_ts must be specified"
        logger.info("build spaceiv model")
        return build_net_for_spaceiv(args['dim_iv'], args['dim_ts'], args['model_configs'])
    elif data_name == 'fully_random_iv':
        logger.info("build fully random iv model")
        return build_net_for_fully_random_iv(args['simulation_info'], args['model_configs'])
    else:
        raise ValueError(f"data name {data_name} is not valid")
