import typing
from typing import Callable, List, Literal, Optional, Tuple, Union

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
    _validate_input
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

    def _get_straight_line(self, inputs: Tensor, baselines: Tensor, n_steps: int) -> Tensor:
        """Helper function to compute the straight line path once."""
        alphas = torch.linspace(0, 1, steps=n_steps, device=inputs.device).view(-1, 1, 1)
        input_expanded = inputs.unsqueeze(0)
        baseline_expanded = baselines.unsqueeze(0)
        return baseline_expanded + alphas * (input_expanded - baseline_expanded)


    def potential_energy(self, path: Tensor, straight_line: Tensor, beta: float) -> Tensor:
        """
        The potential energy encourages the path between input and baseline pairs to be as short as possible, whilst
        avoiding regions of high curvature. 
        """
        # Distance penalty: How far the path deviates from the straight line
        distance_penalty = torch.norm(path - straight_line , p=2, dim=-1)  # Distance from straight line 
        
        # Curvature penalty: magnitude of model gradients along the path
        outputs = self.forward_func(path)
        path_grads = torch.autograd.grad(
            outputs,
            path,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        curvature_penalty = torch.norm(path_grads, p=2, dim=-1)
        
        return distance_penalty + beta * curvature_penalty
        
    def model(self, inputs: Tensor, baselines: Tensor, beta: float):
    
        input_tensor = inputs[0]
        baseline_tensor = baselines[0]
        batch_size = input_tensor.size(0)

        if self.last_batch_size is not None and self.last_batch_size != batch_size:
            print(f"Batch size changed from {self.last_batch_size} to {batch_size}, clearing param store")
            pyro.clear_param_store()
        self.last_batch_size = batch_size
        
        # Create a straight-line path between baselines and inputs
        straight_line = self._get_straight_line(input_tensor, baseline_tensor, self.n_steps)
        alphas = torch.linspace(0, 1, steps=self.n_steps, device=input_tensor.device).view(-1, 1, 1)
        
        # Sample path deviations
        delta = pyro.sample(
            "path_delta",
            dist.Normal(torch.zeros_like(straight_line), 1.0).to_event(straight_line.dim())
        )
        
        # Zero out the deviations at start and end points
        delta = delta * (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        
        # The optimized path is the straight line plus deviations
        path = straight_line + delta
        path.requires_grad_()
        
        # Compute the potential energy of the path
        energy = self.potential_energy(path, straight_line, beta)
        
        # Include the potential energy in the model's log probability
        pyro.factor("energy", -energy.sum())
        
    def guide(self, inputs: Tensor, baselines: Tensor, beta: float):
        input_tensor = inputs[0]
        baseline_tensor = baselines[0]
        batch_size = input_tensor.shape[0] 

        if hasattr(self, 'last_batch_size') and self.last_batch_size != batch_size:
            pyro.clear_param_store()
        self.last_batch_size = batch_size
        
        # Get straight line path once using the helper method
        straight_line = self._get_straight_line(input_tensor, baseline_tensor, self.n_steps)
        alphas = torch.linspace(0, 1, steps=self.n_steps, device=input_tensor.device).view(-1, 1, 1)
        
        # Initialize parameters with appropriate shapes
        delta_loc = pyro.param(
            "delta_loc",
            lambda: torch.zeros_like(straight_line)
        )
        
        # Zero out the location parameters at start and end points
        delta_loc = delta_loc * (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        
        # For scale, use a small positive value at endpoints
        delta_scale = pyro.param(
            "delta_scale",
            lambda: 0.1 * torch.ones_like(straight_line),
            constraint=dist.constraints.positive
        )
        
        # Instead of zeroing scale, make it very small at endpoints
        endpoint_mask = (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        delta_scale = delta_scale * endpoint_mask + 1e-6 * (1 - endpoint_mask)
        
        # Sample the path deviations
        pyro.sample(
            "path_delta",
            dist.Normal(delta_loc, delta_scale).to_event(straight_line.dim())
        )

    def _optimize_paths(
        self,
        inputs: Tensor,
        baselines: Tensor,
        beta: float,
        num_iterations: int = 1000,
        lr: float = 1e-2,
    ):
        """
        Optimizes the paths between inputs and baselines using SVI.
        Returns paths that follow low-energy regions between the points.
        """
        # Initialize the optimizer and SVI object
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Run the optimization
        for i in range(num_iterations):
            loss = svi.step(inputs, baselines, beta)
            if i % 100 == 0:
                print(f"Iteration {i} - Loss: {loss}")

        input_tensor = inputs[0]
        baseline_tensor = baselines[0]
        
        # After optimization, retrieve the optimized path
        alphas = torch.linspace(0, 1, steps=self.n_steps).unsqueeze(1).to(input_tensor.device)
        alphas = alphas.view(-1, 1, 1)

        input_expanded = input_tensor.unsqueeze(0)     # Shape: [1, batch, features]
        baseline_expanded = baseline_tensor.unsqueeze(0)  # Shape: [1, batch, features]

        straight_line = baseline_expanded + alphas * (input_expanded - baseline_expanded)
        
        delta_opt = pyro.param("delta_loc").detach()
        optimized_path = straight_line + delta_opt

        # Verify that the path starts at baseline and ends at input
        assert torch.allclose(optimized_path[0], baseline_expanded[0]), "Path doesn't start at baseline"
        assert torch.allclose(optimized_path[-1], input_expanded[0]), "Path doesn't end at input"


        return optimized_path

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        return_paths: bool = True,
        beta: float = 0.3,
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
        self.n_steps = n_steps
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        # pyre-fixme[9]: inputs has type `TensorOrTupleOfTensorsGeneric`; used as
        #  `Tuple[Tensor, ...]`.
        formatted_inputs, formatted_baselines = _format_input_baseline(
            inputs, baselines
        )

        # pyre-fixme[6]: For 1st argument expected `Tuple[Tensor, ...]` but got
        #  `TensorOrTupleOfTensorsGeneric`.
        _validate_input(formatted_inputs, formatted_baselines, n_steps, method)

        
        # TODO: add the batched version as well
        attributions, paths = self._attribute(
                inputs=formatted_inputs,
                baselines=formatted_baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
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
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
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

        # First, optimize the paths between inputs and baselines
        optimized_paths = self._optimize_paths(inputs, baselines, beta, num_iterations, lr=learning_rate)

        # We need to compute gradients for each step along the path
        all_grads = []
        for i in range(n_steps):
            path_point = optimized_paths[i]  # Shape: [batch, features]
            path_point.requires_grad_()
            
            # Compute gradients at this point
            grad = self.gradient_func(
                forward_fn=self.forward_func,
                inputs=(path_point,),  # Wrap in tuple as Captum expects
                target_ind=target,
                additional_forward_args=additional_forward_args,
            )[0]  # Extract from tuple
            all_grads.append(grad)
        
        # Stack all gradients
        grads = torch.stack(all_grads) 

        # Scale and aggregate gradients
        scaled_grads = [
            grads.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).float().view(n_steps, 1).to(grads.device)
        ]

        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, [grads])
        )

        # Compute final attributions
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        
        return attributions, optimized_paths
            



        

