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
)
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)

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
        self.last_batch_size = None

    def _get_straight_line(
        self, 
        inputs: Tuple[Tensor, ...], 
        baselines: Tuple[Union[Tensor, int, float], ...],
        n_steps: int
    ) -> Tuple[Tensor, ...]:
        """
        Helper function to compute straight line paths.
        """
        paths = []
        for input, baseline in zip(inputs, baselines):
            # Convert baseline to tensor if needed
            if not isinstance(baseline, Tensor):
                baseline = torch.tensor(baseline, device=input.device, dtype=input.dtype)
            
            # Expand baseline to match input shape if needed
            if baseline.shape != input.shape:
                baseline = baseline.expand_as(input)
                
            # Get input shape and batch size
            batch_size = input.shape[0]
            feature_dims = input.shape[1:]
            
            # Create alphas [n_steps]
            alphas = torch.linspace(0, 1, steps=n_steps, device=input.device)
            
            # Reshape input and baseline for broadcasting
            input_expanded = input.unsqueeze(0)  # [1, batch_size, *feature_dims]
            baseline_expanded = baseline.unsqueeze(0)  # [1, batch_size, *feature_dims]
            
            # Reshape alphas for broadcasting with input dimensions
            alphas = alphas.view(n_steps, 1, *([1] * len(feature_dims)))  # [n_steps, 1, 1, ...]
            
            # Compute interpolated path
            path = baseline_expanded + alphas * (input_expanded - baseline_expanded)  # [n_steps, batch_size, *feature_dims]
            paths.append(path)

        return tuple(paths)
    def _get_approx_paths(
        self,
        inputs: Tensor,
        baselines: Tensor,
        augmentation_data: Tensor,
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

        # TODO: I think Joseph's implementation of the KNN/A* works better. Either we should reformat his output to be compatible
        # with this, or we should refactor this code to use the same methodology as Joseph's.
        pass

    def potential_energy(
        self, 
        paths: Tuple[Tensor, ...], 
        initial_paths: Tuple[Tensor, ...], 
        beta: float
    ) -> Tensor:
        """
        Compute potential energy across all input tensors.
        
        Args:
            paths: Tuple of current path tensors
            initial_paths: Tuple of initial path tensors
            beta: Weight for curvature penalty
            
        Returns:
            Combined potential energy scalar
        """
        # Compute penalties for each tensor
        energies = []
        for path, init_path in zip(paths, initial_paths):
            # Distance penalty
            distance_penalty = torch.norm(path - init_path, p=2, dim=-1)
            
            # Curvature penalty
            outputs = self.forward_func(path)
            path_grads = torch.autograd.grad(
                outputs,
                path,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True,
            )[0]
            curvature_penalty = torch.norm(path_grads, p=2, dim=-1)
            
            # Combine penalties for this tensor
            energy = distance_penalty + beta * curvature_penalty
            energies.append(energy)
        
        # Sum energies across all tensors
        total_energy = sum(energies)
        
        return total_energy

    def model(
        self, 
        inputs: Tuple[Tensor, ...], 
        baselines: Tuple[Union[Tensor, int, float], ...],
        initial_paths: Tuple[Tensor, ...],
        beta: float
    ):
        """Probabilistic model for path optimization that handles tuple inputs."""
        input_tensor = inputs[0]  # Use first tensor for device/dtype
        dtype = input_tensor.dtype
        batch_size = input_tensor.size(0)

        if self.last_batch_size is not None and self.last_batch_size != batch_size:
            print(
                f"Batch size changed from {self.last_batch_size} to {batch_size}, clearing param store"
            )
            pyro.clear_param_store()
        self.last_batch_size = batch_size

        alphas = torch.linspace(
            0, 1, steps=self.n_steps, device=input_tensor.device, dtype=dtype
        ).view(-1, 1, 1)

        # Sample path deviations for each tensor
        deltas = tuple(
            pyro.sample(
                f"path_delta_{i}",
                dist.Normal(torch.zeros_like(path), 1.0).to_event(path.dim()),
            )
            for i, path in enumerate(initial_paths)
        )

        # Zero out the deviations at start and end points for each path
        deltas = tuple(
            delta * (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
            for delta in deltas
        )

        # The optimized paths are the initial paths plus deviations
        paths = tuple(
            init_path + delta
            for init_path, delta in zip(initial_paths, deltas)
        )
        
        # Enable gradients for all paths
        for path in paths:
            path.requires_grad_()

        # Compute the potential energy of all paths
        energy = self.potential_energy(paths, initial_paths, beta)

        # Include the potential energy in the model's log probability
        pyro.factor("energy", -energy.sum())

    def guide(
        self, 
        inputs: Tuple[Tensor, ...], 
        baselines: Tuple[Union[Tensor, int, float], ...],
        initial_paths: Tuple[Tensor, ...],
        beta: float
    ):
        """
        Variational guide for SVI that handles tuple inputs.
        """
        input_tensor = inputs[0]  # Use first tensor for device/dtype
        dtype = input_tensor.dtype
        batch_size = input_tensor.shape[0]

        if hasattr(self, "last_batch_size") and self.last_batch_size != batch_size:
            pyro.clear_param_store()
        self.last_batch_size = batch_size

        alphas = torch.linspace(
            0, 1, steps=self.n_steps, device=input_tensor.device, dtype=dtype
        ).view(-1, 1, 1)

        # Initialize parameters for each tensor in the tuple
        delta_locs = tuple(
            pyro.param(f"delta_loc_{i}", lambda: torch.zeros_like(path))
            for i, path in enumerate(initial_paths)
        )

        # Zero out the location parameters at start and end points for each tensor
        delta_locs = tuple(
            loc * (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
            for loc in delta_locs
        )

        # Initialize scale parameters for each tensor
        delta_scales = tuple(
            pyro.param(
                f"delta_scale_{i}",
                lambda: 0.1 * torch.ones_like(path),
                constraint=dist.constraints.positive,
            )
            for i, path in enumerate(initial_paths)
        )

        # Handle endpoints for scales
        endpoint_mask = (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        delta_scales = tuple(
            scale * endpoint_mask + 1e-6 * (1 - endpoint_mask)
            for scale in delta_scales
        )

        # Sample path deltas for each tensor
        for i, (loc, scale, path) in enumerate(zip(delta_locs, delta_scales, initial_paths)):
            pyro.sample(
                f"path_delta_{i}",
                dist.Normal(loc, scale).to_event(path.dim()),
            )

    def _optimize_paths(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        augmentation_data: Optional[Tensor] = None,
        n_neighbors: Optional[int] = None,
        beta: float = 0.3,
        num_iterations: int = 1000,
        lr: float = 1e-2,
    ) -> Tuple[Tensor, ...]:
        """
        Optimize paths between inputs and baselines using SVI.
        
        Args:
            inputs: Tuple of input tensors
            baselines: Tuple of baseline tensors or values
            augmentation_data: Optional tensor for path augmentation
            n_neighbors: Number of neighbors for augmented paths
            beta: Weight for curvature penalty
            num_iterations: Number of optimization iterations
            lr: Learning rate
            
        Returns:
            Tuple of optimized path tensors
        """
        if augmentation_data is not None:
            initial_paths = tuple(
                self._get_approx_paths(
                    input_tensor,
                    baseline_tensor,
                    augmentation_data,
                    self.n_steps,
                    n_neighbors,
                )
                for input_tensor, baseline_tensor in zip(inputs, baselines)
            )
            beta = 1 / beta if beta > 1 else beta
            current_beta = beta * 10
            decay_rate = (current_beta * beta) ** (1 / num_iterations)
        else:
            initial_paths = self._get_straight_line(inputs, baselines, self.n_steps)
            current_beta = beta
            decay_rate = 1

        optimizer = Adam({"lr": lr})
        svi = SVI(
            lambda inputs, baselines, initial_paths, beta: self.model(
                inputs, baselines, initial_paths, beta
            ),
            lambda inputs, baselines, initial_paths, beta: self.guide(
                inputs, baselines, initial_paths, beta
            ),
            optimizer,
            loss=Trace_ELBO(),
        )

        for i in range(num_iterations):
            current_beta *= decay_rate
            loss = svi.step(inputs, baselines, initial_paths, current_beta)

            if i % 100 == 0:
                print(f"Iteration {i} - Loss: {loss} - Beta: {current_beta}")

        # Get deltas for each tensor in the tuple
        try:
            delta_opts = tuple(
                pyro.param(f"delta_loc_{i}").detach()
                for i in range(len(inputs))
            )
        except KeyError as e:
            raise KeyError(
                f"Could not find parameter {e}. Available parameters: {list(pyro.get_param_store().keys())}"
            )

        optimized_paths = tuple(
            init_path + delta
            for init_path, delta in zip(initial_paths, delta_opts)
        )

        # Verification code remains the same
        rtol = 1e-4
        atol = 1e-4
        
        for path, input_tensor, baseline_tensor in zip(optimized_paths, inputs, baselines):
            if not torch.allclose(path[-1], input_tensor, rtol=rtol, atol=atol):
                max_diff = (path[-1] - input_tensor).abs().max()
                mean_diff = (path[-1] - input_tensor).abs().mean()
                print(f"Maximum difference: {max_diff}")
                print(f"Mean difference: {mean_diff}")
                print(f"Shape of input_tensor: {input_tensor.shape}")
                print(f"Shape of optimized_path: {path.shape}")
                
            assert torch.allclose(
                path[0], baseline_tensor, rtol=rtol, atol=atol
            ), "Path doesn't start at baseline"
            assert torch.allclose(
                path[-1], input_tensor, rtol=rtol, atol=atol
            ), "Path doesn't end at input"

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
        return_paths: bool = True,
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

        self.n_steps = n_steps
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        formatted_inputs, formatted_baselines = _format_input_baseline(
            inputs, baselines
        )

        _validate_input(formatted_inputs, formatted_baselines, n_steps, method)

        # TODO: add the batched version as well
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

        return _format_output(is_inputs_tuple, attributions), paths

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        augmentation_data: Tensor = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        n_neighbors: int = 20,
        method: str = "gausslegendre",
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

        optimized_paths = self._optimize_paths(
            inputs,
            baselines,
            augmentation_data,
            n_neighbors,
            beta,
            num_iterations,
            lr=learning_rate,
        )

        # Initialize accumulated gradients
        accumulated_grads = tuple(torch.zeros_like(input) for input in inputs)
        
        # Compute and accumulate gradients along the path
        for i, step_size in enumerate(step_sizes):
            path_points = tuple(path[i] for path in optimized_paths)
            for point in path_points:
                point.requires_grad_()

            grads = self.gradient_func(
                forward_fn=self.forward_func,
                inputs=path_points,
                target_ind=target,
                additional_forward_args=additional_forward_args,
            )

            # Accumulate scaled gradients
            accumulated_grads = tuple(
                acc_grad + grad * step_size 
                for acc_grad, grad in zip(accumulated_grads, grads)
            )

            for point in path_points:
                point.requires_grad_(False)

        # Compute final attributions
        if self._multiply_by_inputs:
            attributions = tuple(
                acc_grad * (input - baseline)
                for acc_grad, input, baseline in zip(accumulated_grads, inputs, baselines)
            )
        else:
            attributions = accumulated_grads

        return attributions, optimized_paths