import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers import ECCConv


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
        super(PhaseModel, self).__init__()
        ################################################################################
        # CREATE NETWORK
        ################################################################################
        self.ECCNet = [] 
        # [ECCConv(int(self.Edge_dim*ecc_hidden_factor**(i+1), kernel_network=self.kernel_network, activation="relu")for i in range(ecc_layer)]
        print('information about the model: ')
        for cnt_layer in range(ecc_layers):
            self.ECCNet.append(ECCConv(int(Node_dim*ecc_hidden_factor**(cnt_layer+1)), use_bias=use_bias,\
                                       kernel_network=kernel_network, activation=activation))
            print("hidden dimensions of layer : ", cnt_layer+1, " of GNN equals to : ", int(Node_dim*ecc_hidden_factor**(cnt_layer+1)))
        
        # Add pooling layer
        self.pool = global_pool.get(pool_type)()
        
        # Add MLP layers 
