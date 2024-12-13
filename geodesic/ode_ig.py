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

import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm



class ModelManifoldGeodesicFlow(nn.Module):
    def __init__(self, forward_func):
        super().__init__()
        self.forward_func = forward_func
        
    def get_metric_tensor(self, x):
        """Compute metric tensor with better numerical stability"""
        with torch.enable_grad():
            x.requires_grad_(True)
            f = self.forward_func(x)
            Js = []
            for i in range(f.numel()):
                f_i = f.view(-1)[i]
                J_i = torch.autograd.grad(f_i, x, 
                                        create_graph=True,
                                        retain_graph=True)[0]
                Js.append(J_i)
            J = torch.stack(Js)
            # Add larger regularization
            G = J.T @ J + 1e-4 * torch.eye(J.shape[-1], device=x.device)
        return G

    def forward(self, t, state):
        deviation, dev_velocity = torch.chunk(state, 2, dim=-1)
        
        # Window function to ensure zero deviation at endpoints
        window = 4 * t * (1 - t)  # Peaks at t=0.5, zero at t=0,1
        
        x_straight = self.x0 + t * (self.x1 - self.x0)
        x = x_straight + window * deviation  # Apply window to deviation
        
        G = self.get_metric_tensor(x)
        
        # Compute dG and Christoffel symbols as before
        dG = torch.zeros(x.shape[0], G.shape[0], G.shape[1])
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                g_ij = G[i,j]
                dg_ij = torch.autograd.grad(g_ij, x,
                                          create_graph=True,
                                          retain_graph=True)[0]
                dG[:,i,j] = dg_ij
        
        dG = dG / (torch.norm(dG) + 1e-6)
        Gamma = 0.5 * (dG + dG.permute(0,2,1) - dG.permute(2,0,1))
        
        # Compute acceleration including window effect
        a = -torch.einsum('ijk,j,k->i', Gamma, dev_velocity, dev_velocity)
        a = a / (torch.norm(dev_velocity) + 1e-6)
        
        # Add restoration force towards straight line
        restoration = -0.1 * deviation  # Weak spring force
        
        return torch.cat([dev_velocity, a + restoration], dim=-1)
    
    def integrate(self, x0, x1, n_steps=50, rand_scale=0.2, seed=None):
        """
        Args:
            x0: Starting point
            x1: Ending point 
            n_steps: Number of integration steps
            rand_scale: Scale of randomization (0.0-1.0), default 0.2
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        self.x0 = x0
        self.x1 = x1
        
        t = torch.linspace(0, 1, n_steps)
        
        # Get metric at midpoint for initial conditions
        x_mid = 0.5 * (x0 + x1)
        G_mid = self.get_metric_tensor(x_mid)
        
        # Compute initial deviation direction
        straight_dir = (x1 - x0) / torch.norm(x1 - x0)
        eigenvals, eigenvecs = torch.linalg.eigh(G_mid)
        max_curve_dir = eigenvecs[:, -1]
        
        # Add random perturbation to direction
        rand_dir = torch.randn_like(max_curve_dir)
        rand_dir = rand_dir - (rand_dir @ straight_dir) * straight_dir
        rand_dir = rand_dir / (torch.norm(rand_dir) + 1e-8)
        
        # Mix original and random directions
        mixed_dir = max_curve_dir + rand_scale * rand_dir
        mixed_dir = mixed_dir - (mixed_dir @ straight_dir) * straight_dir
        mixed_dir = mixed_dir / (torch.norm(mixed_dir) + 1e-8)
        
        # Random velocity scaling
        base_scale = torch.sqrt(eigenvals[-1]) * 0.1
        rand_factor = 1.0 + rand_scale * (2 * torch.rand(1) - 1)
        velocity0 = base_scale * rand_factor * mixed_dir
        deviation0 = torch.zeros_like(x0)
        
        state0 = torch.cat([deviation0, velocity0])
        
        deviations = odeint(
            self,
            state0, 
            t,
            method='dopri5',
            rtol=1e-5,
            atol=1e-7,
            options={'max_num_steps': 1000}
        )
        
        # Construct final path with windowed deviations
        window = 4 * t[:, None] * (1 - t[:, None])
        straight_line = x0[None] + t[:, None] * (x1 - x0)[None]
        path = straight_line + window * deviations[:, :x0.shape[0]]
        
        return path
    
class OdeIG(GradientAttribution):
    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        multiply_by_inputs: bool = True,
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs


    def _optimize_paths(self, initial_paths, input_additional_args):
        # Get dimensions from first tensor in tuple
        n_total = initial_paths[0].shape[0]  # n_steps * n_inputs
        n_steps = self.n_steps
        n_inputs = n_total // n_steps
        
        flow = ModelManifoldGeodesicFlow(self.forward_func)
        
        # Process each tensor in tuple
        optimized_paths = []
        for tensor in initial_paths:
            # Reshape to [n_steps, n_inputs, features]
            paths = tensor.view(n_steps, n_inputs, -1)
            
            # Get start and end points
            baselines = paths[0]   # [n_inputs, features]
            inputs = paths[-1]     # [n_inputs, features]
            
            # Generate geodesic paths for each input-baseline pair
            batch_paths = []
            pbar = tqdm(range(n_inputs), desc='Optimizing paths')
            for idx in pbar:
                path = flow.integrate(baselines[idx], inputs[idx], n_steps)  # [n_steps, features]
                batch_paths.append(path)
                
            # Stack and reshape back to original format
            batch_paths = torch.stack(batch_paths, dim=1)  # [n_steps, n_inputs, features]
            batch_paths = batch_paths.reshape(n_steps * n_inputs, -1)  # [n_steps * n_inputs, features]
            optimized_paths.append(batch_paths)
        
        return tuple(optimized_paths)
    

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

        optimized_paths = self._optimize_paths(
                initial_paths,
                input_additional_args,
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
        return tuple(returned_variables) if len(returned_variables) > 1 else formatted_outputs
