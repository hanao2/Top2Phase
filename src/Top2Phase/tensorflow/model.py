import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from spektral.layers.pooling import global_pool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers import ECCConv


class PhaseModel(Model):
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
        self.MLPNet = []
        for cnt_layer in range(mlp_layers-1):
            self.MLPNet.append(Dense(max(Output_dim, int(Node_dim*ecc_hidden_factor**(ecc_layers-cnt_layer-1))),use_bias=use_bias, activation=activation))
            print("hidden dimensions of layer : ", cnt_layer+1, ' of MLP equals to : ', int(Node_dim*ecc_hidden_factor**(ecc_layers-cnt_layer-1)))
         
        # output layer with no activation
        self.MLPNet.append(Dense(Output_dim))
        self.non_functional = False
        if self.non_functional:
            pass


    def get_config(self):
        return self.config
    

    def call(self, inputs, training: bool = True):
        x, a, e = inputs
        
        for ecc_layer in range(self.config['ecc_layers']):
            x = self.ECCNet[ecc_layer]([x,a,e])
        
        x = self.pool(x)
        
        for mlp_layer in range(self.config['mlp_layers']):
            x = self.MLPNet[mlp_layer](x)
        return x
