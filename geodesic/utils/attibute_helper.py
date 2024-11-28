from torch import Tensor
import torch

from captum.attr._utils.common import _format_input_baseline, _validate_input
from captum._utils.common import _is_tuple

def data_and_params_validator(inputs, baselines, n_steps, method, distance, additional_forward_args=None):
    # Keeps track whether original input is a tuple or not before
    # converting it into a tuple.
    is_inputs_tuple = _is_tuple(inputs)

    inputs, baselines = _format_input_baseline(inputs, baselines)

    # If baseline is float or int, create a tensor
    baselines = tuple(
        torch.ones_like(input) * baseline
        if isinstance(baseline, (int, float))
        else baseline
        for input, baseline in zip(inputs, baselines)
    )

    _validate_input(inputs, baselines, n_steps, method)

    # If additional_forward_args has a tensor, assert inputs
    # consists of one sample
    if additional_forward_args is not None:
        if any(isinstance(x, Tensor) for x in additional_forward_args):
            assert (
                len(inputs[0]) == 1
            ), "Only one sample must be passed when additional_forward_args has a tensor."

    # Check distance
    assert distance in [
        "geodesic",
        "euclidean",
    ], f"distance must be either 'geodesic' or 'euclidean', got {distance}"
    return inputs, baselines, is_inputs_tuple