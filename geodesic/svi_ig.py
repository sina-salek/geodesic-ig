import typing
from typing import Callable, List, Literal, Optional, Tuple, Union
import warnings

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
    _run_forward,
)
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)

from geodesic.svi_batching import _batch_attribution

from captum.attr._utils.approximation_methods import approximation_parameters


from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class SVI_IG(GradientAttribution):
    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        multiply_by_inputs: bool = True,
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs

    def _get_approx_paths(
        self,
        inputs: Tensor,
        baselines: Tensor,
        augmentation_data: Tensor,
        alphas: Tensor,
        n_steps: int,
        n_neighbors: int,
    ) -> Tensor:
        """
        Compute approximate shortest paths between inputs and baselines through augmentation points
        using A* algorithm and gradient-based edge weights.

        Args:
            inputs: Input points [batch_size, features]
            baselines: Baseline points [batch_size, features]
            augmentation_data: Additional points to consider in path [n_aug, features]
            n_steps: Number of points desired in final path

        Returns:
            Tensor: Interpolated paths [n_steps, batch_size, features]
        """

        # TODO: Use Joseph's implementation of the KNN/A* to get the shortest path with augmentation data.
        pass

    def potential_energy(
        self,
        path: Tuple[Tensor, ...],
        initial_paths: Tuple[Tensor, ...],
        beta: float,
        input_additional_args: Tuple[Tensor, ...],
    ) -> Tensor:
        """
        Args:
            path: Current path points [n_steps * batch_size, n_features]
            initial_paths: Initial path points [n_steps * batch_size, n_features]
            beta: Weight of curvature penalty
        Returns:
            Total potential energy (scalar)
        """
        # distance penalty
        distance_penalties = tuple(
            torch.norm(path[i] - initial_paths[i], p=2, dim=-1)
            for i in range(len(path))
        )

        # curvature penalty
        with torch.autograd.set_grad_enabled(True):
            outputs = _run_forward(
                self.forward_func, path, additional_forward_args=input_additional_args
            )
            path_grads = torch.autograd.grad(
                outputs,
                path,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True,
            )
        curvature_penalties = tuple(
            torch.norm(path_grads[i], p=2, dim=-1) for i in range(len(path_grads))
        )

        total_penalty = sum(
            (distance_penalties[i] - beta * curvature_penalties[i]).sum()
            for i in range(len(distance_penalties))
        )

        return total_penalty

    def model(
        self,
        initial_paths: Tuple[Tensor, ...],
        beta: float,
        input_additional_args: Tuple[Tensor, ...],
    ):
        """Model samples path deviations without reshaping."""
        # Sample path deviations directly
        delta_tuple = tuple(
            pyro.sample(
                f"path_delta_{i}",
                dist.Normal(torch.zeros_like(initial_paths[i]), 1.0).to_event(
                    initial_paths[i].dim()
                ),
            )
            for i in range(len(initial_paths))
        )

        # Create path with gradients
        paths = tuple(
            (initial_paths[i] + delta_tuple[i]).requires_grad_()
            for i in range(len(initial_paths))
        )

        energy = self.potential_energy(
            paths, initial_paths, beta, input_additional_args
        )
        pyro.factor("energy", -energy)

    def guide(
        self,
        initial_paths: Tuple[Tensor, ...],
        beta: float,
        input_additional_args: Tuple[Tensor, ...],
    ):
        """Guide learns optimal deviations without reshaping."""
        # Learn parameters directly in original shape
        delta_locs = tuple(
            pyro.param(f"delta_loc_{i}", lambda: torch.zeros_like(initial_paths[i]))
            for i in range(len(initial_paths))
        )

        delta_scales = tuple(
            pyro.param(
                f"delta_scale_{i}",
                lambda: 0.1 * torch.ones_like(initial_paths[i]),
                constraint=dist.constraints.positive,
            )
            for i in range(len(initial_paths))
        )

        # Sample and create path
        for i in range(len(initial_paths)):
            pyro.sample(
                f"path_delta_{i}",
                dist.Normal(delta_locs[i], delta_scales[i]).to_event(
                    initial_paths[i].dim()
                ),
            )

        optimized_paths = tuple(
            (initial_paths[i] + delta_locs[i]).requires_grad_()
            for i in range(len(initial_paths))
        )
        return optimized_paths

    def _optimize_paths(
        self,
        initial_paths: Tuple[Tensor, ...],
        input_additional_args: Tuple[Tensor, ...],
        beta_decay_rate: float,
        current_beta: float = 0.3,
        num_iterations: int = 1000,
        lr: float = 1e-2,
    ) -> Tensor:
        """Optimize paths between inputs and baselines using SVI."""

        # Setup optimizer and inference
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(retain_graph=True))

        # Run optimization with decaying beta
        beta = current_beta
        for step in range(num_iterations):
            # Optimize
            loss = svi.step(initial_paths, beta, input_additional_args)
            beta *= beta_decay_rate

            if step % 100 == 0:
                print(f"Step {step}: loss = {loss:.3f}, beta = {beta:.3f}")

        # Sample optimized paths
        with torch.no_grad():
            # Use guide to get mean of learned path distribution
            optimized_paths = self.guide(initial_paths, beta, input_additional_args)

        return optimized_paths

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        augmentation_data: Tensor = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        return_paths: bool = False,
        beta: float = 0.3,
        n_neighbors: int = 20,
        num_iterations: int = 1000,
        learning_rate: float = 0.01,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        """
        This is similar to IntegratedGradients, but instead of integrating over a straight line, we use the SVI method
        to integrate over a geodesic path. Geodesic paths are shortest paths between two points on a manifold. They
        avoid regions of high curvature, which are regions of high log-likelihood gradient.
        """
        if augmentation_data is not None and n_neighbors is None:
            raise ValueError(
                "Augmentation data is provided, but no n_neighbors is given. Please provide a n_neighbors."
            )
        if augmentation_data is None and n_neighbors is not None:
            warnings.warn(
                "n_neighbors is provided, but no augmentation data is given. Ignoring n_neighbors."
            )
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        formatted_inputs, formatted_baselines = _format_input_baseline(
            inputs, baselines
        )

        _validate_input(formatted_inputs, formatted_baselines, n_steps, method)
        paths = None
        if internal_batch_size is not None:
            num_examples = formatted_inputs[0].shape[0]
            attributions = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=formatted_inputs,
                baselines=formatted_baselines,
                augmentation_data=augmentation_data,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
                beta=beta,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
            )
        else:
            attributions, paths = self._attribute(
                inputs=formatted_inputs,
                baselines=formatted_baselines,
                augmentation_data=augmentation_data,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                n_neighbors=n_neighbors,
                method=method,
                beta=beta,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
            )
        formatted_outputs = _format_output(is_inputs_tuple, attributions)
        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )

        returned_variables = []
        returned_variables.append(formatted_outputs)
        if return_paths:
            returned_variables.append(paths)
        if return_convergence_delta:
            returned_variables.append(delta)
        return (
            tuple(returned_variables)
            if len(returned_variables) > 1
            else formatted_outputs
        )

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        augmentation_data: Tensor = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        n_neighbors: int = 20,
        method: Union[None, str] = None,
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
        num_iterations: int = 1000,
        beta: float = 0.3,
        learning_rate: float = 0.01,
    ) -> Tuple[Tensor, ...]:

        if step_sizes_and_alphas is None:
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        straight_line_tuple = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )  # straight line between input and baseline. Dim of each tensor in tuple: [n_steps * batch_size, n_features]

        if augmentation_data is not None:
            initial_paths = self._get_approx_paths(
                inputs,
                baselines,
                augmentation_data,
                alphas,
                self.n_steps,
                n_neighbors,
            )

            beta = 1 / beta if beta > 1 else beta
            current_beta = beta * 10
            beta_decay_rate = (current_beta * beta) ** (1 / num_iterations)
        else:
            initial_paths = straight_line_tuple
            current_beta = beta
            beta_decay_rate = 1

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # TODO: Whilst generally works as expected, there are not many points on the high curvature regions.
        # Need to be interpolated, to have uniform number of points along the path.
        optimized_paths = self._optimize_paths(
            initial_paths,
            input_additional_args,
            beta_decay_rate,
            current_beta,
            num_iterations,
            lr=learning_rate,
        )

        n_inputs = tuple(
            input.shape[0] for input in inputs
        )
        n_features = tuple(
            input.shape[1] for input in inputs
        )

        step_sizes_tuple = tuple(
            calculate_step_sizes(path, n_inputs[i], n_steps, n_features[i])
            for i, path in enumerate(optimized_paths)
        )

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=optimized_paths,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous()
            * torch.tensor(step_sizes).to(grad.device)
            for step_sizes, grad in zip(step_sizes_tuple, grads)
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        return attributions, optimized_paths
    
def calculate_step_sizes(path, n_inputs, n_steps, n_features):
    paths_reshaped = path.view(n_steps, n_inputs, n_features)
    
    # Calculate initial step sizes
    step_sizes = torch.norm(
        paths_reshaped[1:] - paths_reshaped[:-1],
        dim=-1
    )
    
    # Add final step to match dimensions
    last_step = step_sizes[-1:]
    step_sizes = torch.cat([step_sizes, last_step], dim=0)
    
    # Reshape to match original path dimensions
    step_sizes = step_sizes.view(n_steps * n_inputs, 1)
    
    return step_sizes
