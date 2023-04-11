from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np
import yaml


def parcs_simulate(parcs_config_yml, data_size, rho, rand_seed, hetero=False, sigma=1):
    np.random.seed(rand_seed)

    with open(parcs_config_yml, 'r') as f:
        parcs_config = yaml.safe_load(f)
        parcs_config['rho'] = f'constant({rho})'
        if hetero:
            parcs_config['sigma'] = f'constant({sigma})'

    with open(parcs_config_yml, 'w') as f:
        yaml.dump(parcs_config, f)

    nodes, edges = graph_file_parser(parcs_config_yml)
    g = Graph(nodes=nodes, edges=edges)
    samples = g.sample(size=data_size)
    return samples

if __name__ == '__main__':
    print(parcs_simulate('demand.yml', 50, 0.5, 0))