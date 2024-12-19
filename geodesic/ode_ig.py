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


from captum.attr._utils.approximation_methods import approximation_parameters


from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor

import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm


class BatchedModelManifoldGeodesicFlow(nn.Module):
    def __init__(self, forward_func):
        super().__init__()
        self.forward_func = forward_func
            
    def get_metric_tensor(self, x_batch):
        """Vectorized metric tensor computation for arbitrary inputs"""
        batch_size = x_batch.shape[0]
        original_shape = x_batch.shape
        x_flat = x_batch.flatten(start_dim=1)
        n_features = x_flat.shape[1]
        
        with torch.enable_grad():
            # Reshape if image data (4D), keep flat if tabular (2D)
            if len(original_shape) == 4:
                x_model = x_flat.view(original_shape)
            else:
                x_model = x_flat
                
            x_model.requires_grad_(True)
            
            # Compute output once
            output = self.forward_func(x_model)
            out_dim = output.shape[1]
            
            # Initialize Jacobian tensor
            J = torch.zeros((batch_size, out_dim, n_features), device=x_model.device)
            
            # Compute gradients for each output dimension
            for i in range(out_dim):
                grad = torch.autograd.grad(
                    output[:, i].sum(),
                    x_model,
                    create_graph=True,
                    retain_graph=True
                )[0]
                J[:, i] = grad.flatten(start_dim=1)
            
            # Compute metric tensor with correct dimensions
            G = torch.bmm(J, J.transpose(1, 2))  # [batch, out_dim, out_dim]
            G = G + 1e-4 * torch.eye(out_dim, device=x_batch.device)[None].expand(batch_size, -1, -1)
        
        return G

    def forward(self, t, state_batch):
        batch_size = state_batch.shape[0] // 2
        deviation, dev_velocity = state_batch.chunk(2, dim=0)
        
        window = 4 * t * (1 - t)
        x_straight = self.x0_batch + t.view(-1, 1) * (self.x1_batch - self.x0_batch)
        x = x_straight + window * deviation
        
        # Reshape x to original dimensions for metric tensor computation
        if hasattr(self, 'original_shape'):
            x_reshaped = x.view(-1, *self.original_shape[1:])
        else:
            x_reshaped = x
        
        G = self.get_metric_tensor(x_reshaped)
        
        dG = torch.zeros(batch_size, G.shape[1], G.shape[2], x.shape[1], device=x.device)
        x_flat = x.flatten(start_dim=1)
        for i in range(G.shape[1]):
            for j in range(G.shape[2]):
                g_ij = G[:, i, j]
                dg_ij = torch.autograd.grad(g_ij.sum(), x_flat, create_graph=True, retain_graph=True)[0]
                dG[:, i, j] = dg_ij
        
        dG = dG / (torch.sqrt((dG ** 2).sum(dim=(1,2,3), keepdim=True)) + 1e-6)
        Gamma = 0.5 * (dG + dG.permute(0, 2, 1, 3) - dG.permute(0, 3, 1, 2))
        
        a = -torch.einsum('bijk,bj,bk->bi', Gamma, dev_velocity, dev_velocity)
        a = a / (torch.norm(dev_velocity, dim=1, keepdim=True) + 1e-6)
        
        restoration = -0.1 * deviation
        
        return torch.cat([dev_velocity, a + restoration], dim=0)

    def integrate_batch(self, x0_batch, x1_batch, n_steps=50, rand_scale=0.2):
        """Batched integration with flattened tensors"""
        self.original_shape = x0_batch.shape
        
        # Flatten inputs
        self.x0_batch = x0_batch.flatten(start_dim=1)
        self.x1_batch = x1_batch.flatten(start_dim=1)
        batch_size = x0_batch.shape[0]
        n_features = self.x0_batch.shape[1]
        
        t = torch.linspace(0, 1, n_steps, device=x0_batch.device)
        
        # Compute initial conditions in feature space
        x_mid = 0.5 * (self.x0_batch + self.x1_batch)
        G_mid = self.get_metric_tensor(x_mid.view(-1, *self.original_shape[1:]))
        
        # Get direction in feature space
        straight_dir = (self.x1_batch - self.x0_batch)
        straight_dir = straight_dir / torch.norm(straight_dir, dim=1, keepdim=True)
        
        eigenvals, eigenvecs = torch.linalg.eigh(G_mid)
        
        # Project max_curve_dir back to feature space using Jacobian
        with torch.enable_grad():
            x_mid_model = x_mid.view(-1, *self.original_shape[1:])
            x_mid_model.requires_grad_(True)
            output = self.forward_func(x_mid_model)
            max_curve_dir = torch.autograd.grad(
                torch.sum(output * eigenvecs[:, :, -1]), 
                x_mid_model,
                create_graph=True
            )[0].flatten(start_dim=1)
        
        # Generate random direction in feature space
        rand_dir = torch.randn_like(straight_dir)
        rand_dir = rand_dir - torch.sum(rand_dir * straight_dir, dim=1, keepdim=True) * straight_dir
        rand_dir = rand_dir / (torch.norm(rand_dir, dim=1, keepdim=True) + 1e-8)
        
        # Mix directions in feature space
        mixed_dir = max_curve_dir + rand_scale * rand_dir
        mixed_dir = mixed_dir - torch.sum(mixed_dir * straight_dir, dim=1, keepdim=True) * straight_dir
        mixed_dir = mixed_dir / (torch.norm(mixed_dir, dim=1, keepdim=True) + 1e-8)
        
        # Initialize velocity in feature space
        base_scale = torch.sqrt(eigenvals[:, -1]) * 0.1
        rand_factor = 1.0 + rand_scale * (2 * torch.rand(batch_size, 1, device=x0_batch.device) - 1)
        velocity0 = base_scale.unsqueeze(1) * rand_factor * mixed_dir
        deviation0 = torch.zeros_like(self.x0_batch)
        
        state0 = torch.cat([deviation0, velocity0], dim=0)
        
        deviations = odeint(
            self,
            state0,
            t,
            method='dopri5',
            rtol=1e-5,
            atol=1e-7,
            options={'max_num_steps': 1000}
        )
        
        window = 4 * t[:, None, None] * (1 - t[:, None, None])
        straight_line = self.x0_batch[None] + t[:, None, None] * (self.x1_batch - self.x0_batch)[None]
        path = straight_line + window * deviations[:, :x0_batch.shape[0]]
        
        if len(self.original_shape) > 2:
            path = path.view(n_steps, batch_size, *self.original_shape[1:])
        
        return path

class OdeIG(GradientAttribution):
    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        multiply_by_inputs: bool = True,
    ) -> None:
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs

    def _optimize_paths(self, inputs, baselines, input_additional_args):
        flow = BatchedModelManifoldGeodesicFlow(self.forward_func)
        
        optimized_paths = []
        for input_tensor, baseline_tensor in zip(inputs, baselines):
            paths = flow.integrate_batch(baseline_tensor, input_tensor, self.n_steps)
            paths = paths.reshape(self.n_steps * input_tensor.shape[0], -1)
            optimized_paths.append(paths)
        
        return tuple(optimized_paths)

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
    ) -> Tuple[Tensor, ...]:



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
            inputs,
            baselines,
            input_additional_args,
        ) #tuple of tensors of [n_steps * n_inputs, features] dimensions

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

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        return_paths: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        """
        This is similar to IntegratedGradients, but instead of integrating over a straight line, we use the SVI method
        to integrate over a geodesic path. Geodesic paths are shortest paths between two points on a manifold. They
        avoid regions of high curvature, which are regions of high log-likelihood gradient.
        """

        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        formatted_inputs, formatted_baselines = _format_input_baseline(
            inputs, baselines
        )

        _validate_input(formatted_inputs, formatted_baselines, n_steps)
        paths = None
        if internal_batch_size is not None:
            raise NotImplementedError(
                "Internal batch size is not supported yet for OdeIG"
            )
        else:
            attributions, paths = self._attribute(
                inputs=formatted_inputs,
                baselines=formatted_baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
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