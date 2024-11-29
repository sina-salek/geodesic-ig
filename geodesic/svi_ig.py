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

    def potential_energy(self, inputs: Tensor, baselines: Tensor, grads: Tensor, beta: float) -> Tensor:
        """
        The potential energy encourages the path between input and baseline pairs to be as short as possible, whilst
        avoiding regions of high curvature. 
        
        Args:
            inputs: The model input
            baselines: The baseline input (usually zero)
            grads: The gradients at the current points
            beta: The weight of the curvature penalty relative to the distance penalty
            
        Returns:
            Tensor: The potential energy at the current points
        """
        # Distance penalty: Euclidean distance from the straight line
        straight_line = baselines + (inputs - baselines)  # straight line between baseline and input
        distance_penalty = torch.norm(grads - straight_line, p=2, dim=-1)
        
        # Curvature penalty: magnitude of gradients
        curvature_penalty = torch.norm(grads, p=2, dim=-1)
        
        # Total potential energy
        energy = distance_penalty + beta * curvature_penalty
        
        return energy
    
    def model(self, inputs: Tensor, baselines: Tensor, beta: float):
        """
        Pyro model for optimizing the paths between inputs and baselines.
        We define a probabilistic model over the paths, where the paths are
        encouraged to have low potential energy as defined by the potential_energy function.
        """
        n_steps = self.n_steps  # Number of steps along the path
        input_tensor = inputs[0]
        baseline_tensor = baselines[0]
        # batch_size = inputs.size(0)
        
        # Create a straight-line path between baselines and inputs
        alphas = torch.linspace(0, 1, steps=n_steps).unsqueeze(1).to(input_tensor.device)
        alphas = alphas.view(-1, 1, 1)

        input_expanded = input_tensor.unsqueeze(0)     # Shape: [1, batch, features]
        baseline_expanded = baseline_tensor.unsqueeze(0)  # Shape: [1, batch, features]

        straight_line = baseline_expanded + alphas * (input_expanded - baseline_expanded)
        
        # Sample deviations from the straight-line path
        delta = pyro.sample(
            "path_delta",
            dist.Normal(torch.zeros_like(straight_line), 1.0).to_event(straight_line.dim())
        )
        
        # The optimized path is the straight line plus deviations
        path = straight_line + delta
        path.requires_grad_()
        
        # Compute gradients along the path
        outputs = self.forward_func(path)
        grads = torch.autograd.grad(
            outputs,
            path,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute the potential energy of the path
        energy = self.potential_energy(input_tensor, baseline_tensor, grads, beta)
        
        # Include the potential energy in the model's log probability
        pyro.factor("energy", -energy.sum())
        
    def guide(self, inputs: Tensor, baselines: Tensor, beta: float):
        """
        Pyro guide (variational distribution) for the paths.
        We learn a Gaussian distribution over the path deviations from the straight line,
        with learnable mean and scale parameters for each point along the path.
        
        Args:
            inputs: The target input points (used for shape/device information)
            baselines: The starting points
            beta: Not directly used in guide, but kept for interface consistency
        """
        input_tensor = inputs[0]
        baseline_tensor = baselines[0]
        # Create learnable parameters for the path deviations
        alphas = torch.linspace(0, 1, steps=self.n_steps).unsqueeze(1).to(input_tensor.device)
        alphas = alphas.view(-1, 1, 1)  # Shape: [n_steps, 1, 1]
    
        # Expand tensors for broadcasting
        input_expanded = input_tensor.unsqueeze(0)     # Shape: [1, batch, features]
        baseline_expanded = baseline_tensor.unsqueeze(0)  # Shape: [1, batch, features]

        straight_line = baseline_expanded + alphas * (input_expanded - baseline_expanded)
        
        # Initialize parameters with appropriate shapes
        delta_loc = pyro.param(
            "delta_loc",
            lambda: torch.zeros_like(straight_line)
        )
        
        delta_scale = pyro.param(
            "delta_scale",
            lambda: 0.1 * torch.ones_like(straight_line),
            constraint=dist.constraints.positive
        )
        
        # Sample the path deviations using these learned parameters
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
        attributions, optimized_paths = self._attribute(
                inputs=formatted_inputs,
                baselines=formatted_baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )
        
        return _format_output(is_inputs_tuple, attributions), optimized_paths
    

    
    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # First, optimize the paths between inputs and baselines
        optimized_paths = self._optimize_paths(inputs, baselines, 0.01)

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
            



        

