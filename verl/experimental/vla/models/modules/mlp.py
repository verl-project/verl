# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) module.
    It supports dynamic layer construction, multiple activation functions,
    and various weight initialization strategies.

    Attributes:
        input_dim (int): The number of input features.
        hidden_dims (list of int): List containing the number of units in each hidden layer.
        output_dim (int): The number of output units.
        activation (str): The non-linear activation function to use.
            Options: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'none'.
        init_method (str): The weight initialization strategy.
            Options: 'kaiming', 'xavier', 'normal', 'orthogonal'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        init_method: str = "kaiming",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.init_method = init_method.lower()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            act_layer = self._get_activation(self.activation_name)
            if act_layer is not None:
                layers.append(act_layer)
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def _get_activation(self, name):
        """
        Factory method to return the activation layer based on string name.
        Available options: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu'.
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "none": None,
        }
        return activations.get(name, nn.ReLU())

    def init_weights(self, m):
        """
        Public method to initialize weights for Linear layers.
        Can be used with self.apply(model.init_weights).

        Supported methods:
            - 'kaiming': Best for ReLU/LeakyReLU. Uses kaiming_normal_.
            - 'xavier': Best for Tanh/Sigmoid. Uses xavier_normal_.
            - 'normal': Standard normal distribution (std=0.02).
            - 'orthogonal': Good for preventing gradient explosion in deep networks.
        """
        if isinstance(m, nn.Linear):
            if self.init_method == "kaiming":
                # Use 'relu' as default nonlinearity for Kaiming
                nonlinearity = self.activation_name if self.activation_name in ["relu", "leaky_relu"] else "relu"
                init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            elif self.init_method == "xavier":
                init.xavier_normal_(m.weight)
            elif self.init_method == "normal":
                init.normal_(m.weight, mean=0.0, std=0.02)
            elif self.init_method == "orthogonal":
                init.orthogonal_(m.weight)

            # Initialize bias to zero
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        """Defines the computation performed at every call."""
        return self.network(x)
