import torch as th 
import torch.nn as nn 

import functools, itertools 

class MLP_Model(nn.Module):
    """
        layers_conf: array of integer, define number of neurons per layer
        non_linears: binary array, each val defines if we should apply relu 
        normalizers: binary array, each val defines if we should apply batch normalization 
    """
    def __init__(self, layers_conf, non_linears, normalizers):
        super(MLP_Model, self).__init__()
        self.shapes = list(zip(layers_conf[:-1], layers_conf[1:]))
        self.non_linears = non_linears 
        self.normalizers = normalizers
        self.layers_array = nn.ModuleList([])

        zipped_iterables = zip(self.shapes, self.non_linears, self.normalizers)
        for (in_dim, out_dim), apply_fn, apply_bn in zipped_iterables:
            activation_funct = nn.ReLU() if apply_fn == 1 else nn.Identity()
            linear_transform = nn.Linear(in_features=in_dim, out_features=out_dim)
            batch_normalizer = nn.BatchNorm1d(out_dim) if apply_bn else nn.Identity()
            block = nn.Sequential(
                linear_transform, 
                batch_normalizer,
                activation_funct
            )
            self.layers_array.append(block)
        # end for loop ...! 

        self.nb_layers = len(self.layers_array)

    def forward(self, X0):
        XN = functools.reduce(
            lambda Xi,Li: Li(Xi), 
            self.layers_array, 
            X0
        )
        return XN 

if __name__ == '__main__':
    model = MLP_Model([32, 32, 32, 32], [1, 1, 0], [1, 1, 0])
    print(model)