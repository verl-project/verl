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
        output_init_scale (float): Scale for uniform initialization of output layer weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        init_method: str = "kaiming",
        output_init_scale: float = 3e-3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.init_method = init_method.lower()
        self.output_init_scale = float(output_init_scale)

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

    def _get_activation(self, name: str):
        """
        Factory method to return a *fresh* activation layer instance based on string name.
        Available options: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'none'.
        """
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        if name == "elu":
            return nn.ELU()
        if name == "selu":
            return nn.SELU()
        if name == "none":
            return None
        return nn.ReLU()

    def init_weights(self, m: nn.Module):
        """
        Initialize weights for Linear layers.

        Hidden layers follow init_method.
        Output layer uses small uniform init (±output_init_scale) to keep initial outputs near 0.
        """
        if not isinstance(m, nn.Linear):
            return

        # Identify the output layer by matching out_features to the requested output_dim
        # (works because only the last Linear has out_features == self.output_dim in this MLP)
        is_output_layer = m.out_features == self.output_dim

        if is_output_layer:
            init.uniform_(m.weight, -self.output_init_scale, self.output_init_scale)
        else:
            if self.init_method == "kaiming":
                if self.activation_name == "leaky_relu":
                    init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                else:
                    init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif self.init_method == "xavier":
                init.xavier_normal_(m.weight)
            elif self.init_method == "normal":
                init.normal_(m.weight, mean=0.0, std=0.02)
            elif self.init_method == "orthogonal":
                init.orthogonal_(m.weight)

        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.network(x)
