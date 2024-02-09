import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

import geotorch



# Note - I would like to credit the pytorch tutorial, that the formatting of my functions is similar to:
# https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

class LowROrth(nn.Module):
    """
    Class for SCA model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowROrth, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b
        geotorch.orthogonal(self.fc2,"weight") #Make V orthogonal

    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output



class Sphere(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)
    def right_inverse(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)


class LowRNorm(nn.Module):
    """
    Class for SCA (with unit norm, but not orthogonal) model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowRNorm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias

        self.fc2 = P.register_parametrization(nn.Linear(self.hidden_size, self.output_size), "weight", Sphere(dim=0))
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b


    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output

# Used to clamp stddev > 1
class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=1, max=torch.inf)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class mLowRNorm(LowRNorm):
    def __init__(self, *args, n_regions=2, filter_length=41):
        super(mLowRNorm, self).__init__(*args)
        self.filter_length = filter_length
        self.n_regions = n_regions

        # Learned filter params
        self.mus = torch.nn.Parameter(
            torch.zeros(self.n_regions, self.hidden_size)
        )
        self.sigmas = torch.nn.Parameter(
            torch.ones(self.n_regions, self.hidden_size)
        )
        self.scaling = torch.nn.Parameter(
            torch.ones(self.n_regions, self.hidden_size)
        )

        # X-values used for evaluating the filters
        self._x_vals = torch.arange(
            -self.filter_length//2,
            self.filter_length//2
        )

    def encode_region(self, reg_num, x_r, d0, d1):
        z_r = (x_r @ self.fc1.weight[d0:d1])
        print('something')

    def encode(self, x):
        zs, d0 = [], 0

        ### TODO: problem is that we need a single-trial for first loss computation
        for reg_num, (r, x_r) in enumerate(x.items()):
            d1 = d0 + x_r.shape[1]
            zs.append(self.encode_region(reg_num, x_r, d0, d1))

    def forward(self, x):
        zs = self.encode(x)
        print('something')