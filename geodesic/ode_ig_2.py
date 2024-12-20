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

forward_call_counts = 0
class BatchedModelManifoldGeodesicFlow(nn.Module):
    def __init__(self, forward_func):
        super().__init__()
        self.forward_func = forward_func
        self.cached_metric_tensors = {}
        
    def _compute_metric_tensor(self, x_batch):
        """Single computation of metric tensor"""
        with torch.enable_grad():
            x_batch.requires_grad_(True)
            f_batch = self.forward_func(x_batch)
            
            Js = []
            for i in range(f_batch.shape[1]):
                f_i = f_batch[:, i]
                J_i = torch.autograd.grad(f_i.sum(), x_batch, create_graph=True)[0]
                Js.append(J_i)
            
            J = torch.stack(Js, dim=1)
            G = torch.bmm(J.transpose(1, 2), J)
            G = G + 1e-4 * torch.eye(G.shape[-1], device=x_batch.device)[None]
        return G

    def interpolate_metric_tensor(self, t):
        """Interpolate between precomputed metric tensors"""
        if t <= 0.5:
            w = 2 * t
            G0 = self.cached_metric_tensors[0.0]
            G1 = self.cached_metric_tensors[0.5]
        else:
            w = 2 * (t - 0.5)
            G0 = self.cached_metric_tensors[0.5]
            G1 = self.cached_metric_tensors[1.0]
        return (1 - w) * G0 + w * G1

    def forward(self, t, state_batch):
        global forward_call_counts
        forward_call_counts += 1
        batch_size = state_batch.shape[0] // 2
        deviation, dev_velocity = state_batch.chunk(2, dim=0)
        
        window = 4 * t * (1 - t)
        x_straight = self.x0_batch + t.view(-1, 1) * (self.x1_batch - self.x0_batch)
        x = x_straight + window * deviation

        # Make x require grad
        x.requires_grad_(True)
        G = self.interpolate_metric_tensor(t.item())
        
        # Vectorized Christoffel computation
        dG = torch.zeros(batch_size, G.shape[1], G.shape[2], x.shape[1], device=x.device)
        for i in range(G.shape[1]):
            for j in range(G.shape[2]):
                g_ij = G[:, i, j]
                try:
                    dg_ij = torch.autograd.grad(g_ij.sum(), x, create_graph=True, retain_graph=True, 
                                            allow_unused=True)[0]
                    if dg_ij is not None:  # If gradient exists
                        dG[:, i, j] = dg_ij
                except RuntimeError:
                    continue  # Skip if gradient computation fails
        
        # Rest remains the same
        dG = dG / (torch.sqrt((dG ** 2).sum(dim=(1,2,3), keepdim=True)) + 1e-6)
        Gamma = 0.5 * (dG + dG.permute(0, 2, 1, 3) - dG.permute(0, 3, 1, 2))
        
        a = -torch.einsum('bijk,bj,bk->bi', Gamma, dev_velocity, dev_velocity)
        a = a / (torch.norm(dev_velocity, dim=1, keepdim=True) + 1e-6)
        
        restoration = -0.1 * deviation
        
        return torch.cat([dev_velocity, a + restoration], dim=0)

    def integrate_batch(self, x0_batch, x1_batch, n_steps=50, rand_scale=0.2):
        """Batched integration"""
        self.x0_batch = x0_batch
        self.x1_batch = x1_batch
        batch_size = x0_batch.shape[0]
        
        # Precompute metric tensors at key points
        key_points = {
            0.0: x0_batch,
            0.5: 0.5 * (x0_batch + x1_batch),
            1.0: x1_batch
        }
        for t, x in key_points.items():
            self.cached_metric_tensors[t] = self._compute_metric_tensor(x)
        
        t = torch.linspace(0, 1, n_steps, device=x0_batch.device)
        
        # Use precomputed middle tensor for initial conditions
        G_mid = self.cached_metric_tensors[0.5]
        
        straight_dir = (x1_batch - x0_batch) / torch.norm(x1_batch - x0_batch, dim=1, keepdim=True)
        eigenvals, eigenvecs = torch.linalg.eigh(G_mid)
        max_curve_dir = eigenvecs[:, :, -1]
        
        rand_dir = torch.randn_like(max_curve_dir)
        rand_dir = rand_dir - torch.sum(rand_dir * straight_dir, dim=1, keepdim=True) * straight_dir
        rand_dir = rand_dir / (torch.norm(rand_dir, dim=1, keepdim=True) + 1e-8)
        
        mixed_dir = max_curve_dir + rand_scale * rand_dir
        mixed_dir = mixed_dir - torch.sum(mixed_dir * straight_dir, dim=1, keepdim=True) * straight_dir
        mixed_dir = mixed_dir / (torch.norm(mixed_dir, dim=1, keepdim=True) + 1e-8)
        
        base_scale = torch.sqrt(eigenvals[:, -1]) * 0.1
        rand_factor = 1.0 + rand_scale * (2 * torch.rand(batch_size, 1, device=x0_batch.device) - 1)
        velocity0 = base_scale.unsqueeze(1) * rand_factor * mixed_dir
        deviation0 = torch.zeros_like(x0_batch)
        
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
        straight_line = x0_batch[None] + t[:, None, None] * (x1_batch - x0_batch)[None]
        path = straight_line + window * deviations[:, :x0_batch.shape[0]]
        
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

        print(f"forward_call_counts: {forward_call_counts}")
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