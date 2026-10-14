"""MLPBlocks: a multilayer perceptron with optional residual connections."""

import torch as th
import torch.nn as nn

from fragnnet.utils.nn_utils import get_clones


class MLPBlocks(nn.Module):
    """
     Just Good Old Multilayer perceptron with residuals
    layer->dropout->activation(Relu)
    if residuals is True, add a skip connection between block
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        normalization: str,
        dropout: float,
        num_layers: int,
        output_size: int | None = None,
        use_residuals: bool = False,
    ) -> None:
        """MLP with optional residual connections.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden layer dimension.
            normalization (str): Type of normalization ('batch', 'layer', or 'none').
            dropout (float): Dropout probability.
            num_layers (int): Number of hidden layers (excluding input/output).
            output_size (int, optional): Output feature dimension. If None, no output layer is used. If not None, use a linear layer to project to output_size.
            use_residuals (bool, optional): If True, adds skip connections between hidden layers.
        """

        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

        self.output_layer = None
        self.output_size = output_size
        if self.output_size is not None:
            self.output_layer = nn.Linear(hidden_size, self.output_size)

        self.use_residuals = use_residuals

        # setup norm
        if normalization == "batch":
            norm_mod = nn.BatchNorm1d(hidden_size)
        elif normalization == "layer":
            norm_mod = nn.LayerNorm(hidden_size)
        else:
            if normalization != "none":
                raise ValueError(f"unknown normalization {normalization!r}")
            norm_mod = nn.Identity()
        self.norms = get_clones(norm_mod, num_layers - 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        output = x
        output = self.input_layer(x)
        output = self.dropout_layer(output)
        output = self.activation(output)
        old_op = output
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)
            output = self.norms[layer_index](output)
            if self.use_residuals:
                output = output + old_op
                old_op = output

        if self.output_layer is not None:
            output = self.output_layer(output)
        return output
