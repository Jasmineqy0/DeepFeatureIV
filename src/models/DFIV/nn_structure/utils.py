from torch import nn


# def create_neural_network(input_size, hidden_layer, hidden_dim, output_dim, batch_normalization):
#     layers = []
    
#     # Input layer
#     layers.append(nn.Linear(input_size, hidden_dim))
#     if batch_normalization:
#         layers.append(nn.BatchNorm1d(hidden_dim))
#     layers.append(nn.ReLU())
    
#     # Hidden layers
#     for _ in range(hidden_layer - 1):
#         layers.append(nn.Linear(hidden_dim, hidden_dim))
#         if batch_normalization:
#             layers.append(nn.BatchNorm1d(hidden_dim))
#         layers.append(nn.ReLU())
        
#     # output layer
#     layers.append(nn.Linear(hidden_dim, output_dim))
    
#     # Create and return the sequential model
#     model = nn.Sequential(*layers)
#     return model

def create_neural_network(input_size, num_layer, hidden_dim, batch_normalization):
    assert num_layer >= 1, "num_layer must be greater than or equal to 1"
    layers = []

    # Input layer
    layers.append(nn.Linear(input_size, hidden_dim))
    if batch_normalization:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())

    # Hidden layers
    for i in range(num_layer - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if batch_normalization:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if i not in [num_layer - 2]:
            layers.append(nn.ReLU())

    # Create and return the sequential model
    model = nn.Sequential(*layers)
    return model