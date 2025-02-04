import typing
from typing import Callable, List, Literal, Optional, Tuple, Union
import warnings
import numpy as np

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

from geodesic.utils.svi_batching import _batch_attribution

from captum.attr._utils.approximation_methods import approximation_parameters


from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class GeodesicIGSVI(GradientAttribution):
    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        multiply_by_inputs: bool = True,
        seed: int = 42,
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        pyro.set_rng_seed(seed)
        
        # Enable deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        self.forward_func = self.forward_func.to(self.device)
        for param in self.forward_func.parameters():
            param.data = param.data.to(self.device)

        print(f"Initialised SVI_IG on device: {self.device}")

    def _ensure_device(self, tensor_or_tuple):
        """Move tensor or tuple of tensors to correct device"""
        if isinstance(tensor_or_tuple, tuple):
            return tuple(t.to(self.device) if t is not None else None 
                        for t in tensor_or_tuple)
        return tensor_or_tuple.to(self.device) if tensor_or_tuple is not None else None

    def potential_energy(
        self,
        path: Tuple[Tensor, ...],
        initial_paths: Tuple[Tensor, ...],
        beta: float,
        input_additional_args: Tuple[Tensor, ...],
        use_endpoints_matching: bool = True,
    ) -> Tensor:
        """
        Calculates the potential energy of the path using distance and curvature penalties.

        Computes a total penalty based on:
        1. Distance penalty: L2 norm between current path and initial path
        2. Curvature penalty: L2 norm of path gradients
        3. Optional endpoint matching penalty for path endpoints (if use_endpoints_matching=True)

        Args:
            path: Tuple of tensors representing the current path coordinates
            initial_paths: Tuple of tensors representing the initial path coordinates
            beta: Float weighting factor for curvature penalty term
            input_additional_args: Additional arguments passed to the forward function
            use_endpoints_matching: Whether to apply additional penalties to constrain path endpoints

        Returns:
            Tensor: Total potential energy computed as sum of weighted penalties

        Notes:
            - When use_endpoints_matching is True, applies stronger constraints on first and last 10% 
            of path points to ensure path endpoints match initial coordinates
            - Endpoint matching uses a fixed weight of 100 for the constraint term
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
        if use_endpoints_matching:
            # Add endpoint matching penalties
            endpoint_weight = 100
            endpoint_penalties = 0
            n_batch = int(path[0].shape[0] // self.n_steps)
            n_features = path[0].shape[1:]
            view_shape = (self.n_steps, n_batch) + n_features

            # Calculate 10% of steps
            n_edge_steps = max(1, int(0.1 * self.n_steps))

            for i in range(len(path)):
                path_reshaped = path[i].view(view_shape)
                initial_reshaped = initial_paths[i].view(view_shape)
                
                # Penalise first 10% deviation
                endpoint_penalties += endpoint_weight * torch.norm(
                    path_reshaped[:n_edge_steps] - initial_reshaped[:n_edge_steps], 
                    p=2, dim=-1
                ).sum()
                
                # Penalise last 10% deviation
                endpoint_penalties += endpoint_weight * torch.norm(
                    path_reshaped[-n_edge_steps:] - initial_reshaped[-n_edge_steps:], 
                    p=2, dim=-1
                ).sum()
            return total_penalty  + endpoint_penalties
        else:
            return total_penalty

    def model(self, *args, **kwargs):
        """Defines the probabilistic model for path sampling using Pyro.

        Implements a stochastic model that:
        1. Takes initial paths as input
        2. Samples path deviations from a Normal distribution
        3. Combines initial paths with sampled deviations
        4. Computes potential energy of the resulting paths
        5. Adds energy as a factor in the probabilistic model

        Args:
            *args: Variable length argument list, expected contents:
                - args[0]: Tuple[Tensor, ...], initial path coordinates
                - args[1]: float, beta parameter for potential energy calculation
                - args[2]: Tuple[Tensor, ...], additional arguments for forward function
            **kwargs: Arbitrary keyword arguments (unused)

        Notes:
            - Path deviations are sampled from N(0,1) with same shape as initial paths
            - Resulting paths require gradients for energy calculation
            - Energy is added as a negative factor to define the probability density
        """
        initial_paths = self._ensure_device(args[0])
        input_additional_args = self._ensure_device(args[2])
        
        delta_tuple = tuple(
            pyro.sample(
                f"path_delta_{i}",
                dist.Normal(
                    torch.zeros_like(initial_paths[i]).to(self.device), 
                    torch.ones_like(initial_paths[i]).to(self.device)
                ).to_event(initial_paths[i].dim()),
            )
            for i in range(len(initial_paths))
        )

        paths = tuple(
            (initial_paths[i] + delta_tuple[i]).requires_grad_()
            for i in range(len(initial_paths))
        )

        energy = self.potential_energy(paths, initial_paths, args[1], 
                                     input_additional_args)
        pyro.factor("energy", -energy)

    def guide(self, *args, **kwargs):
        """Variational guide function for learning optimal path deviations.
    
        Implements a variational distribution that:
        1. Creates learnable location and scale parameters for each path component
        2. Samples path deviations from Normal distributions with learnable parameters
        3. Combines initial paths with optimised deviations
        
        Args:
            *args: Variable length argument list, expected contents:
                - args[0]: Tuple[Tensor, ...], initial path coordinates 
            **kwargs: Arbitrary keyword arguments (unused)
            
        Returns:
            Tuple[Tensor, ...]: Optimised paths created by combining initial paths 
            with learned deviations
            
        Notes:
            - Uses separate location and scale parameters for each path component
            - Scale parameters are constrained to be positive
            - Resulting paths require gradients for optimisation
            - Maintains original path dimensionality without reshaping
        """
        initial_paths = self._ensure_device(args[0])
        
        delta_locs = tuple(
            pyro.param(
                f"delta_loc_{i}", 
                lambda: torch.zeros_like(initial_paths[i])
            ).to(self.device)
            for i in range(len(initial_paths))
        )

        delta_scales = tuple(
            pyro.param(
                f"delta_scale_{i}",
                lambda: 0.1 * torch.ones_like(initial_paths[i]),
                constraint=dist.constraints.positive,
            ).to(self.device)
            for i in range(len(initial_paths))
        )

        for i in range(len(initial_paths)):
            pyro.sample(
                f"path_delta_{i}",
                dist.Normal(
                    delta_locs[i], 
                    delta_scales[i]
                ).to_event(initial_paths[i].dim()),
            )

        optimised_paths = tuple(
            (initial_paths[i] + delta_locs[i]).requires_grad_()
            for i in range(len(initial_paths))
        )
        return optimised_paths

    def _optimise_paths(
            self,
            initial_paths: Tuple[Tensor, ...],
            input_additional_args: Tuple[Tensor, ...],
            beta_decay_rate: float,
            current_beta: float = 0.3,
            num_iterations: int = 100000,
            initial_lr: float = 1e-3,
            min_lr: float = 1e-5,  
            lr_decay_factor: float = 0.5,  
            lr_patience: int = 25, 
            use_endpoints_matching: bool = True,
            do_linear_interp: bool = True,
            patience: int = 100,
            rel_improvement_threshold: float = 1e-4,
        ) -> Tensor:
            """Optimises paths using stochastic variational inference.

            Performs iterative path optimisation by:
            1. Setting up SVI with model and guide functions
            2. Iteratively optimising path parameters
            3. Implementing learning rate decay and early stopping
            4. Optionally applying linear interpolation to final paths

            Args:
                initial_paths: Starting path coordinates
                input_additional_args: Additional arguments for forward function
                beta_decay_rate: Rate at which beta parameter decays during optimisation
                current_beta: Initial beta value for potential energy calculation
                num_iterations: Maximum number of optimisation iterations
                initial_lr: Initial learning rate for optimiser
                min_lr: Minimum learning rate threshold
                lr_decay_factor: Factor by which to reduce learning rate
                lr_patience: Iterations to wait before decaying learning rate
                use_endpoints_matching: Whether to constrain path endpoints
                do_linear_interp: Whether to interpolate final paths
                patience: Iterations to wait before early stopping
                rel_improvement_threshold: Minimum relative improvement for convergence

            Returns:
                Tensor: Optimised path coordinates

            Notes:
                - Uses Adam optimiser with adaptive learning rate
                - Implements early stopping based on relative improvement
                - Monitors and prints progress every 100 iterations
                - Optionally interpolates final paths for uniform spacing
            """
                        
            initial_paths = self._ensure_device(initial_paths)
            input_additional_args = self._ensure_device(input_additional_args)
            
            current_lr = initial_lr
            optimiser = Adam({"lr": current_lr})
            svi = SVI(
                model=lambda *args, **kwargs: self.model(*args, **kwargs),
                guide=lambda *args, **kwargs: self.guide(*args, **kwargs),
                optim=optimiser,
                loss=Trace_ELBO(retain_graph=True)
            )

            best_loss = float('inf')
            patience_counter = 0
            lr_patience_counter = 0
            loss_history = []
            beta = current_beta
            
            for step in range(num_iterations):
                loss = svi.step(initial_paths, beta, input_additional_args, 
                            use_endpoints_matching=use_endpoints_matching)
                loss_history.append(loss)
                beta *= beta_decay_rate

                if len(loss_history) > 1:
                    rel_improvement = (loss_history[-2] - loss) / loss_history[-2]
                    
                    if loss < best_loss:
                        best_loss = loss
                        patience_counter = 0
                        lr_patience_counter = 0
                    else:
                        patience_counter += 1
                        lr_patience_counter += 1
                        
                        # Decay learning rate if no improvement
                        if lr_patience_counter >= lr_patience and current_lr > min_lr:
                            current_lr = max(current_lr * lr_decay_factor, min_lr)
                            # Create new optimiser with updated learning rate
                            optimiser = Adam({"lr": current_lr})
                            svi.optim = optimiser  # Update optimiser in SVI
                            lr_patience_counter = 0
                            print(f"Decreasing learning rate to {current_lr:.6f}")

                    if rel_improvement < rel_improvement_threshold and patience_counter >= patience:
                        print(f"Early stopping at step {step}: Loss converged with relative improvement {rel_improvement:.6f}")
                        break
                if step % 100 == 0:
                    print(f"Step {step}: loss = {loss:.3f}, beta = {beta:.3f}, lr = {current_lr:.6f}")

            with torch.no_grad():
                optimised_paths = self.guide(initial_paths, beta, input_additional_args, 
                                        use_endpoints_matching=use_endpoints_matching)
                optimised_paths = self._ensure_device(optimised_paths)

            if do_linear_interp:
                print("Interpolating paths...")
                optimised_paths = tuple(
                    self.make_uniform_spacing(opt_paths, n_steps=self.n_steps)
                    for opt_paths in optimised_paths
                )

            return optimised_paths
        
    def make_uniform_spacing(self, paths: Tensor, n_steps: int) -> Tuple[Tensor, int]:
        """Creates uniformly spaced paths by interpolating between points.

        Standardises the spacing between path points by:
        1. Calculating step sizes between existing points
        2. Standardising step sizes relative to total path length
        3. Interpolating additional points based on standardised spacing
        4. Resampling to achieve uniform point distribution

        Args:
            paths: Input tensor containing path coordinates
            n_steps: Number of steps in the path

        Returns:
            Tuple containing:
            - Tensor: Paths with uniformly spaced points
            - int: Number of steps in output paths

        Notes:
            - Maintains original path endpoints
            - Interpolates linearly between existing points
            - Resamples to ensure consistent number of points
            - Preserves batch dimension and feature dimensionality
        """
        device = paths.device
        batch_size = paths.shape[0] // n_steps
        feature_dims = paths.shape[1:]
        
        step_sizes = calculate_step_sizes(paths, n_inputs=batch_size, n_features=feature_dims, n_steps=n_steps)
        standardised_step_sizes = step_sizes / step_sizes.sum(dim=0).unsqueeze(0)
        
        paths = paths.view(n_steps, batch_size, *feature_dims)
        standardised_step_sizes = standardised_step_sizes.view(n_steps, batch_size, 1)

        dense_paths = [[] for _ in range(batch_size)]
        
        for j in range(batch_size):
            starts = paths[:-1, j]
            ends = paths[1:, j]
            
            max_step = standardised_step_sizes.max().item()
            scale_factor = n_steps / max_step
            num_points = (standardised_step_sizes[:, j] * scale_factor).long()
            
            all_points = []
            all_points.append(paths[0, j].unsqueeze(0))
            
            for i in range(n_steps-1):
                n = num_points[i].item()
                alphas = torch.linspace(0, 1, n+1, device=device).view(-1, *([1] * len(feature_dims)))
                segment_points = starts[i:i+1] + alphas * (ends[i:i+1] - starts[i:i+1])
                all_points.append(segment_points)
            
            dense_path = torch.cat(all_points, dim=0)
            indices = torch.linspace(0, len(dense_path)-1, n_steps, device=device).long()
            dense_paths[j] = dense_path[indices]
        
        return torch.stack(dense_paths).transpose(0, 1).reshape(n_steps * batch_size, *feature_dims)

    
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
        learning_rate: float = 0.001,
        use_endpoints_matching: bool = True,
        do_linear_interp: bool = True,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        """
        Similar to IntegratedGradients, but integrates over a geodesic path rather than a straight line.
        Uses the SVI method to integrate over geodesic paths, which are shortest paths between two points
        on a manifold. These paths avoid regions of high curvature, which correspond to regions of high
        log-likelihood gradient.

        Args:
            inputs: Input tensor or tuple of tensors
            baselines: Reference baseline values for comparison
            augmentation_data: Optional tensor for data augmentation
            target: Target for attribution
            additional_forward_args: Additional arguments for forward pass
            n_steps: Number of steps for integration
            method: Integration method ('gausslegendre' supported)
            internal_batch_size: Batch size for internal processing
            return_convergence_delta: Whether to return convergence measure
            return_paths: Whether to return computed paths
            beta: Weight parameter for optimisation
            n_neighbours: Number of nearest neighbours for augmentation
            num_iterations: Maximum optimisation iterations
            learning_rate: Initial learning rate for optimisation
            use_endpoints_matching: Whether to constrain path endpoints
            do_linear_interp: Whether to use linear interpolation

        Returns:
            Union of tensor/tuple of tensors and optional convergence delta/paths

        Notes:
            - Handles both single tensor and tuple inputs
            - Supports batch processing via internal_batch_size
            - Validates inputs before processing
            - Implements completeness axiom checking
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
                use_endpoints_matching=use_endpoints_matching,
                do_linear_interp=do_linear_interp,
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
        use_endpoints_matching: bool = True,
        do_linear_interp: bool = True,
    ) -> Tuple[Tensor, ...]:
        """Computes attributions using optimised geodesic paths.

        Implements the core attribution logic by:
        1. Generating initial paths (straight line or augmentation-based)
        2. Optimising paths using SVI
        3. Computing gradients along optimised paths
        4. Aggregating gradients to produce final attributions

        Args:
            inputs: Input tensors to attribute
            baselines: Reference values for comparison
            augmentation_data: Optional data for path augmentation
            target: Attribution target indices
            additional_forward_args: Extra arguments for forward pass
            n_steps: Number of integration steps
            n_neighbours: Number of neighbours for augmentation
            method: Integration method name
            step_sizes_and_alphas: Optional pre-computed integration parameters
            num_iterations: Maximum optimisation iterations
            beta: Weight parameter for optimisation
            learning_rate: Initial optimisation learning rate
            use_endpoints_matching: Whether to constrain path endpoints
            do_linear_interp: Whether to use linear interpolation

        Returns:
            Tuple containing:
            - Tuple[Tensor, ...]: Computed attributions
            - Tuple[Tensor, ...]: Optimised paths

        Notes:
            - Handles both standard and augmented path initialisation
            - Supports multiple input tensors
            - Implements gradient scaling and aggregation
            - Optional input multiplication for final attributions
        """

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
        # expanded_target = _expand_target(target, self.n_steps)

        optimised_paths = self._optimise_paths(
            initial_paths,
            input_additional_args,
            beta_decay_rate,
            current_beta,
            num_iterations,
            initial_lr=learning_rate,
            use_endpoints_matching=use_endpoints_matching,
            do_linear_interp=do_linear_interp,
        )

        n_inputs = tuple(
            input.shape[0] for input in inputs
        )
        n_features = tuple(
            input.shape[1:] for input in inputs
        )

        step_sizes_tuple = tuple(
            calculate_step_sizes(path, n_inputs[i], self.n_steps, n_features[i])
            for i, path in enumerate(optimised_paths)
        )

        expanded_target = _expand_target(target, self.n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=optimised_paths,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous()
            * step_sizes.view(step_sizes.shape[0], *([1] * (grad.dim() - 1))).to(grad.device)
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
        return attributions, optimised_paths
    
def calculate_step_sizes(path, n_inputs, n_steps, n_features):
    """Calculates step sizes between consecutive points along a path.

    Computes L2 norm distances between adjacent path points by:
    1. Reshaping path tensor to separate steps and batch dimensions
    2. Computing pairwise differences between consecutive points
    3. Calculating L2 norms across feature dimensions
    4. Padding final step to maintain dimensions

    Args:
        path: Tensor containing path coordinates
        n_inputs: Number of input examples in batch
        n_steps: Number of steps along path
        n_features: Tuple describing feature dimensions

    Returns:
        Tensor: Step sizes between consecutive path points, 
               shaped as [n_steps * n_inputs, 1]

    Notes:
        - Handles arbitrary feature dimensionality
        - Maintains batch structure
        - Pads final step by repeating last difference
        - Returns reshaped tensor matching original path dimensions
    """
    view_shape = (n_steps, n_inputs) + n_features
    paths_reshaped = path.view(view_shape)
    
    # Calculate initial step sizes
    step_sizes = torch.norm(
        paths_reshaped[1:] - paths_reshaped[:-1],
        p=2,
        dim=tuple(range(2, 2 + len(n_features))),
    )
    
    # Add final step to match dimensions
    last_step = step_sizes[-1:]
    step_sizes = torch.cat([step_sizes, last_step], dim=0)
    
    # Reshape to match original path dimensions
    step_sizes = step_sizes.view(n_steps * n_inputs, 1)
    
    return step_sizes