from simulator.full_random_simulation import sample_iv_data
from ..data.data_class import TrainDataSet, TestDataSet
from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np
import yaml

TEST_SEED = 9999
    
def generate_train_fully_random_iv(simulation_info, sample_size, rand_seed):
    config_path = simulation_info['config_path']
    var_info_path = simulation_info['var_info_path']

    # get the initial random state
    start_random_state = np.random.get_state()
    np.random.seed(rand_seed)

    # sample from the randomized config file
    sample_dict = sample_iv_data(config_path, var_info_path, sample_size)

    # create the train dataset
    train_data = TrainDataSet(treatment=sample_dict['ts'],
                              instrumental=sample_dict['iv'],
                              covariate=sample_dict['cf'] if 'cf' in sample_dict else None,
                              outcome=sample_dict['ot'],
                              structural=sample_dict['st'])
    
    # restore the initial random state
    np.random.set_state(start_random_state)
    return train_data
    
def generate_test_fully_random_iv(simulation_info):
    config_path = simulation_info['config_path']
    var_info_path = simulation_info['var_info_path']
    
    # extract observed covariates
    with open(var_info_path, 'r') as f:
        var_info = yaml.safe_load(f)
    covariate_cols = sorted(var_info['observed']['cf'])

    # get the initial random state
    start_random_state = np.random.get_state()
    np.random.seed(TEST_SEED)

    # parse the graph
    nodes, edges = graph_file_parser(config_path)
    g = Graph(nodes=nodes, edges=edges)

    # bootstrap treatments and covariates
    bootstrap_size = 1000
    bootstrap_samples = g.sample(size=bootstrap_size)
    assert not np.any(np.isnan(bootstrap_samples.to_numpy())), 'NaN values are not allowed in bootstrap samples' 
    treatment_cols = sorted([col for col in bootstrap_samples.columns if col.startswith('ts')])
    structural_cols = sorted([col for col in bootstrap_samples.columns if col.startswith('ot')])
    
    # get the min and max of the treatments and covariates from bootstrap samples
    treatment_min = bootstrap_samples[treatment_cols].min(axis=0)
    treatment_max = bootstrap_samples[treatment_cols].max(axis=0)
    covariate_min = bootstrap_samples[covariate_cols].min(axis=0)
    covariate_max = bootstrap_samples[covariate_cols].max(axis=0)
    
    # generate interventions
    num_intervention, samples_per_intervention = 250, 10
    treatment_intervention = np.zeros((num_intervention, len(treatment_cols)))
    covariate_intervention = np.zeros((num_intervention, len(covariate_cols)))
    
    for i in range(len(treatment_cols)):
        treatment_intervention[:, i] = np.linspace(treatment_min[i], treatment_max[i], num_intervention)
        np.random.shuffle(treatment_intervention[:, i])
    
    for i in range(len(covariate_cols)):
        covariate_intervention[:, i] = np.linspace(covariate_min[i], covariate_max[i], num_intervention)
        np.random.shuffle(covariate_intervention[:, i])
    
    interventions = []
    for i in range(num_intervention):
        intervention_cols = treatment_cols + covariate_cols
        intervention_values = np.hstack((treatment_intervention[i, :], covariate_intervention[i, :])).squeeze().tolist()
        interventions.append(dict(zip(intervention_cols, intervention_values)))
    
    # generate test data
    test_treatments, test_covariates, test_structurals  = [], [], []
    for i in range(num_intervention):
        samples = g.do(size=samples_per_intervention, interventions=interventions[i])
        test_treatments.append(samples[treatment_cols].to_numpy())
        test_structurals.append(samples[structural_cols].to_numpy())
        if covariate_cols:
            test_covariates.append(samples[covariate_cols].to_numpy())
    
    # # randomize interventions from sampled treatments
    # num_intervention, samples_per_intervention = 50, 50
    # bootstrap_treatments = bootstrap_samples[treatment_cols].to_numpy()
    # bootstrap_treatments.sort(axis=0)
    # idxes = np.round(np.linspace(0, bootstrap_size - 1, num_intervention)).astype(int)
    # interventions = []
    # for i in range(len(treatment_cols)):
    #     intervention_i = bootstrap_treatments[idxes, i].squeeze()
    #     np.random.shuffle(intervention_i)
    #     interventions.append(intervention_i.tolist())
    # interventions = zip(*interventions)
    # interventions = [dict(zip(treatment_cols, intervention)) for intervention in interventions]

    # # generate test data
    # test_treatments, test_covariates, test_structurals  = [], [], []
    # for intervention in interventions:
    #     samples = g.do(size=samples_per_intervention, interventions=intervention)
    #     test_treatments.append(samples[treatment_cols].to_numpy().round(2))
    #     test_structurals.append(samples[structural_cols].to_numpy().round(2))
    #     if covariate_cols:
    #         test_covariates.append(samples[covariate_cols].to_numpy().round(2))
        
    # intervention 
    test_data = TestDataSet(treatment=np.vstack(test_treatments),
                            covariate=np.vstack(test_covariates) if covariate_cols else None,
                            structural=np.vstack(test_structurals))

    # restore the initial random state
    np.random.set_state(start_random_state)
    
    return test_data