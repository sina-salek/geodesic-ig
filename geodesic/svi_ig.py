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
        self.last_batch_sizes = None

    def _check_batch_size(self, current_batch_sizes: Tuple[int, ...]) -> bool:
        """Check if batch sizes changed and need param store cleared"""
        if self.last_batch_sizes is None:
            self.last_batch_sizes = current_batch_sizes
            return False
            
        if self.last_batch_sizes != current_batch_sizes:
            sizes_str = ", ".join(
                f"tensor_{i}: {old} -> {new}" 
                for i, (old, new) in enumerate(zip(self.last_batch_sizes, current_batch_sizes))
            )
            print(f"Batch sizes changed ({sizes_str}), clearing param store")
            pyro.clear_param_store()
            self.last_batch_sizes = current_batch_sizes
            return True
            
        return False


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

        # TODO: I think Joseph's implementation of the KNN/A* works better. Either we should reformat his output to be compatible
        # with this, or we should refactor this code to use the same methodology as Joseph's.
        pass

    def potential_energy(self, path: Tensor, initial_paths: Tensor, beta: float) -> Tensor:
        """
        Args:
            path: Current path points [n_steps * batch_size, n_features]
            initial_paths: Initial path points [n_steps * batch_size, n_features]
            beta: Weight of curvature penalty
        Returns:
            Total potential energy (scalar)
        """
        # Distance penalty
        distance_penalty = torch.norm(path - initial_paths, p=2, dim=-1)
        
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
        
        # Sum for total energy
        return (distance_penalty + beta * curvature_penalty).sum()

    def model(self, initial_paths: Tensor, beta: float):
        """Model samples path deviations without reshaping."""
        # Sample path deviations directly
        delta = pyro.sample(
            "path_delta",
            dist.Normal(torch.zeros_like(initial_paths), 1.0).to_event(initial_paths.dim())
        )
        
        # Create path with gradients
        path = (initial_paths + delta).requires_grad_()
        
        energy = self.potential_energy(path, initial_paths, beta)
        pyro.factor("energy", -energy.sum())

    def guide(self, initial_paths: Tensor, beta: float):
        """Guide learns optimal deviations without reshaping."""
        # Learn parameters directly in original shape
        delta_loc = pyro.param(
            "delta_loc",
            lambda: torch.zeros_like(initial_paths)
        )
        delta_scale = pyro.param(
            "delta_scale",
            lambda: 0.1 * torch.ones_like(initial_paths),
            constraint=dist.constraints.positive
        )
        
        
        # Sample and create path
        pyro.sample(
            "path_delta",
            dist.Normal(delta_loc, delta_scale).to_event(initial_paths.dim())
        )
        
        # optimized_path = (initial_paths + delta_loc * mask).requires_grad_()
        optimized_path = (initial_paths + delta_loc ).requires_grad_()
        return optimized_path


    def _optimize_paths(
        self,
        initial_paths: Tensor,
        beta_decay_rate: float, 
        current_beta: float = 0.3,
        num_iterations: int = 1000,
        lr: float = 1e-2,
    ) -> Tensor:
        """Optimize paths between inputs and baselines using SVI."""
        
        # Setup optimizer and inference
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Run optimization with decaying beta
        beta = current_beta
        for step in range(num_iterations):
            # Optimize
            loss = svi.step(initial_paths, beta)
            beta *= beta_decay_rate
            
            if step % 100 == 0:
                print(f"Step {step}: loss = {loss:.3f}, beta = {beta:.3f}")
                
        # Sample optimized paths
        with torch.no_grad():
            # Use guide to get mean of learned path distribution
            optimized_paths = self.guide(initial_paths, beta)
            
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
        ) # straight line between input and baseline. Dim of each tensor in tuple: [n_steps * batch_size, n_features]

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

        optimized_paths = tuple(
            self._optimize_paths(
                init_path,
                beta_decay_rate,
                current_beta,
                num_iterations,
                lr=learning_rate
            )
            for init_path in initial_paths
        )


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
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
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

    
    def _batch_attribution(
        self,
        attr_method,
        num_examples,
        internal_batch_size,
        n_steps,
        include_endpoint=False,
        **kwargs,
    ):
        """
        This method applies internal batching to given attribution method, dividing
        the total steps into batches and running each independently and sequentially,
        adding each result to compute the total attribution.

        Step sizes and alphas are spliced for each batch and passed explicitly for each
        call to _attribute.

        kwargs include all argument necessary to pass to each attribute call, except
        for n_steps, which is computed based on the number of steps for the batch.

        include_endpoint ensures that one step overlaps between each batch, which
        is necessary for some methods, particularly LayerConductance.
        """
        if internal_batch_size < num_examples:
            warnings.warn(
                "Internal batch size cannot be less than the number of input examples. "
                "Defaulting to internal batch size of %d equal to the number of examples."
                % num_examples
            )
        # Number of steps for each batch
        step_count = max(1, internal_batch_size // num_examples)
        if include_endpoint:
            if step_count < 2:
                step_count = 2
                warnings.warn(
                    "This method computes finite differences between evaluations at "
                    "consecutive steps, so internal batch size must be at least twice "
                    "the number of examples. Defaulting to internal batch size of %d"
                    " equal to twice the number of examples." % (2 * num_examples)
                )

        total_attr = None
        cumulative_steps = 0
        step_sizes_func, alphas_func = approximation_parameters(kwargs["method"])
        full_step_sizes = step_sizes_func(n_steps)
        full_alphas = alphas_func(n_steps)

        while cumulative_steps < n_steps:
            start_step = cumulative_steps
            end_step = min(start_step + step_count, n_steps)
            batch_steps = end_step - start_step

            if include_endpoint:
                batch_steps -= 1

            step_sizes = full_step_sizes[start_step:end_step]
            alphas = full_alphas[start_step:end_step]
            current_attr, current_paths= attr_method._attribute(
                **kwargs, n_steps=batch_steps, step_sizes_and_alphas=(step_sizes, alphas)
            )

            if total_attr is None:
                total_attr = current_attr
            else:
                if isinstance(total_attr, Tensor):
                    total_attr = total_attr + current_attr.detach()
                else:
                    total_attr = tuple(
                        current.detach() + prev_total
                        for current, prev_total in zip(current_attr, total_attr)
                    )
            if include_endpoint and end_step < n_steps:
                cumulative_steps = end_step - 1
            else:
                cumulative_steps = end_step
        return total_attr