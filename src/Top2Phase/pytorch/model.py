import os
import itertools
import numpy as np
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.transforms import GCNNorm

class PhaseModel(nn.Module):
    def __init__(self, 
            Node_dim: int = 2,
            Edge_dim: int = 1, 
            Output_dim: int = 1, 
            kernel_network: list = [30,60,30], 
            ecc_layers: int = 3, 
            ecc_hidden_factor: int = 3,
            mlp_layers: int = 3,
            pool_type: str = 'sum',
            activation: str = 'relu',
            use_bias: bool = True
            ):
        # Store all networks architecture in config
        self.config = {'Node_dim': Node_dim,
                       'Edge_dim': Edge_dim,
                       'Output_dim': Output_dim,
                       'kernel_network': kernel_network,
                       'ecc_layers': ecc_layers,
                       'ecc_hidden_factor': ecc_hidden_factor,
                       'mlp_layers': mlp_layers,
                       'pool_type': pool_type,
                       'activation':activation,
                       'use_bias': use_bias}
        activation_dict = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }
        super(PhaseModel, self).__init__()
        ################################################################################
        # CREATE NETWORK
        ################################################################################
        self.ECCNet = []
        print('information about the model: ')
        for cnt_layer in range(ecc_layers):
            in_channels = int(Node_dim*ecc_hidden_factor**(cnt_layer))
            out_channels = int(Node_dim*ecc_hidden_factor**(cnt_layer+1))
            kernel_network = [Edge_dim] + kernel_network
            MLP = itertools.zip_longest(
                    [nn.Linear(kernel_network[i], kernel_network[i+1]) for counter, _ in enumerate(kernel_network)],
                    [activation_dict[activation] for _ in range(len(kernel_network)-2)],
                    fillvalue=None
                    )
            kernel = torch.nn.Sequential(MLP)
            layer = pyg_nn.conv.NNConv(
                    in_channels, 
                    out_channels, 
                    kernel,
                    bias=use_bias
                    )
            self.ECCNet.append(layer)
            print("hidden dimensions of layer : ", cnt_layer+1, " of GNN equals to : ", out_channels)

        # Add pooling layer
        #self.pool = global_pool.get(pool_type)()

        # Add MLP layers
        self.MLPNet = []
        for cnt_layer in range(mlp_layers-1):
            in_channels = ?
            out_channels = max(Output_dim, int(Node_dim*ecc_hidden_factor**(ecc_layers-cnt_layer-1)))
            layer = [nn.Linear(?, out_channels), activation_dict[activation]]
            self.MLPNet.append(layer)
            print("hidden dimensions of layer : ", cnt_layer+1, ' of MLP equals to : ', out_channels)

        # output layer with no activation
        self.MLPNet.append(nn.Linear(out_channels, Output_dim))


    def get_config(self):
        return self.config


    def forward(self, inputs):
        x, a, e = inputs

        for ecc_layer in range(self.config['ecc_layers']):
            x = self.ECCNet[ecc_layer]([x,a,e])

        x = global_mean_pool(x)

        for mlp_layer in range(self.config['mlp_layers']):
            x = self.MLPNet[mlp_layer](x)
        return x
