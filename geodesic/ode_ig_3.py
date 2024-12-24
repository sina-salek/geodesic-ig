import typing
from typing import Callable, List, Literal, Optional, Tuple, Union
import warnings
import gc

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
from torch.linalg import svd

import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm


forward_call_counts = 0
class BatchedModelManifoldGeodesicFlow(nn.Module): 
    def __init__(self, forward_func):
        super().__init__()
        """ model: a PyTorch neural network (nn.Module). We'll assume model(input) produces a tensor of shape [n_batch, ...]. """ 
        self.model = forward_func

    def _compute_jacobian(self, x):
        """Ensure gradient preservation through Jacobian"""
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        y = self.model(x)
        y_flat = y.view(y.size(0), -1)
        
        n_batch, n_output = y_flat.shape
        n_dims = x.view(n_batch, -1).shape[1]
        
        # Create list to store gradients
        grads = []
        
        for i in range(n_output):
            grad = torch.autograd.grad(
                y_flat[:, i].sum(),  # Sum over batch to get scalar
                x,
                create_graph=True,   # Important for higher derivatives
                retain_graph=True    # Keep graph for multiple passes
            )[0]
            
            if grad is not None:
                grads.append(grad.view(n_batch, -1))
            else:
                grads.append(torch.zeros(n_batch, n_dims, device=x.device, dtype=x.dtype))
        
        # Stack gradients to form Jacobian
        jac = torch.stack(grads, dim=1)  # [n_batch, n_output, n_dims]
        
        return jac

    def _compute_metric(self, x):
        """Modified metric tensor computation with gradient tracking"""
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass with explicit gradient retention
        y = self.model(x)
        y_flat = y.view(y.size(0), -1)
        
        n_batch, n_output = y_flat.shape
        n_dims = x.view(n_batch, -1).shape[1]
        
        # print("\nGradient debug:")
        # print(f"x requires grad: {x.requires_grad}")
        # print(f"y_flat requires grad: {y_flat.requires_grad}")
        
        # Compute Jacobian directly with gradient tracking
        J = torch.zeros(n_batch, n_output, n_dims, device=x.device, dtype=x.dtype)
        for i in range(n_output):
            for b in range(n_batch):
                # Take gradient of single output w.r.t. x
                grad = torch.autograd.grad(
                    y_flat[b, i],
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0]
                J[b, i] = grad[b]
        
        # Compute metric tensor while preserving gradients
        G = torch.sum(J.unsqueeze(-2) * J.unsqueeze(-3), dim=1)
        
        # print("\nMetric computation debug:")
        # print(f"J max value: {J.abs().max().item():.6f}")
        # print(f"G requires grad: {G.requires_grad}")
        
        # Test metric derivatives directly
        try:
            dG = sum(torch.autograd.grad(
                G[0,i,j], 
                x,
                retain_graph=True,
                create_graph=True
            )[0].abs().max() for i in range(n_dims) for j in range(n_dims))
            # print(f"Total dG/dx magnitude: {dG.item():.6f}")
        except Exception as e:
            print(f"Gradient test failed: {e}")
        
        return G

    def _compute_christoffel(self, x):
        """Direct computation of Christoffel symbols using metric derivatives"""
        x = x.clone().detach().requires_grad_(True)
        
        # Get Jacobian directly from model output
        y = self.model(x)
        y_flat = y.view(y.size(0), -1)
        n_batch, n_output = y_flat.shape
        n_dims = x.view(n_batch, -1).shape[1]
        
        # print("\nModel output debug:")
        # print(f"y shape: {y.shape}")
        
        # Compute metric components using list accumulation
        G_components = []
        for i in range(n_dims):
            row_components = []
            for j in range(n_dims):
                # Compute metric as inner product of Jacobian rows
                gradi = torch.autograd.grad(
                    y_flat[:, i].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0].view(n_batch, -1)
                
                gradj = torch.autograd.grad(
                    y_flat[:, j].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0].view(n_batch, -1)
                
                component = torch.sum(gradi * gradj, dim=1)
                row_components.append(component)
            
            G_components.append(torch.stack(row_components, dim=1))
        
        # Stack to form metric tensor
        G = torch.stack(G_components, dim=1)
        
        # print(f"\nMetric debug:")
        # print(f"G[0]:\n{G[0]}")

        # Initialize Gamma
        Gamma = torch.zeros(n_batch, n_dims, n_dims, n_dims, device=x.device)
        
        for k in range(n_dims):
            for i in range(n_dims):
                for j in range(n_dims):
                    # Take derivatives and reshape properly
                    dGij_k = torch.autograd.grad(
                        G[:, i, j].sum(),
                        x,
                        create_graph=True,
                        retain_graph=True
                    )[0].view(n_batch, n_dims)  # [batch, dims]
                    
                    dGik_j = torch.autograd.grad(
                        G[:, i, k].sum(),
                        x,
                        create_graph=True,
                        retain_graph=True
                    )[0].view(n_batch, n_dims)  # [batch, dims]
                    
                    dGjk_i = torch.autograd.grad(
                        G[:, j, k].sum(),
                        x,
                        create_graph=True,
                        retain_graph=True
                    )[0].view(n_batch, n_dims)  # [batch, dims]
                    
                    # Debug shapes and values before combining
                    # print(f"\nComponent {k},{i},{j} debug:")
                    # print(f"dGij_k shape: {dGij_k.shape}")
                    # print(f"dGij_k[:, k] shape: {dGij_k[:, k].shape}")
                    # print(f"dG max: {max(dGij_k.abs().max().item(), dGik_j.abs().max().item(), dGjk_i.abs().max().item()):.6f}")
                    
                    # Extract correct components and combine
                    term1 = dGij_k[:, k]  # [batch]
                    term2 = dGik_j[:, j]  # [batch] 
                    term3 = dGjk_i[:, i]  # [batch]
                    
                    Gamma[:, k, i, j] = 0.5 * (term1 + term2 - term3)
        
        return Gamma

    def _geodesic_equations(self, t, state, Gamma):
        """
        Geodesic ODE system in local coordinates.
        state = (x, v), where
        x(t): position,
        v(t): velocity = dx/dt

        d/dt x^μ = v^μ
        d/dt v^μ = - Γ^μ_{νσ} v^ν v^σ

        For batch computations:
        x and v each has shape [n_batch, n_dims].
        Gamma has shape [n_batch, n_dims, n_dims, n_dims].
        
        We'll return (dx, dv) with the same shapes.
        """
        x, v = state
        n_batch, n_dims = x.shape
        
        # We need to compute dv^μ/dt = - Σ_{ν,σ} Γ^μ_{ν σ} v^ν v^σ
        # We'll gather Γ^μ_{ν σ} from Gamma for each batch
        # Then v^ν v^σ is an outer product. We can do broadcasted multiplications.
        
        # v_outer: [n_batch, n_dims, n_dims]
        v_outer = v.unsqueeze(2) * v.unsqueeze(1)
        
        # Then for each μ, we want sum_{ν,σ} Γ^μ_{νσ} v^ν v^σ
        # Gamma: [n_batch, n_dims, n_dims, n_dims]
        # We can do something like:
        #   dv[:, μ] = - sum_{ν,σ} Gamma[:, ν, σ, μ] * v_outer[:, ν, σ]
        # because the last dimension of Gamma is μ, but the two middle dims are ν, σ
        # We'll rearrange Gamma so that μ is the first dimension in that grouping.
        
        # For clarity, we might loop over μ, but we can also do it with batch-wise
        # Einstein summation if we like. For example:
        #   dv[:, μ] = -(Gamma[:, :, :, μ] * v_outer).sum(dim=(1, 2))
        
        dv = torch.zeros_like(v)
        for mu in range(n_dims):
            # sum over nu, sigma
            dv[:, mu] = - (Gamma[:, :, :, mu] * v_outer).sum(dim=(1,2))
        
        dx = v  # dx/dt = v
        return dx, dv

    def integrate_batch(self, baseline_tensor, input_tensor, n_steps=10):
        """Generate geodesic paths as deviations from straight paths"""
        x0 = baseline_tensor.detach().requires_grad_(True)
        xT = input_tensor.detach().requires_grad_(True)
        n_batch = x0.shape[0]
        
        # Initialize paths list with starting points
        paths = torch.zeros(n_steps, n_batch, x0.shape[-1], device=x0.device)
        t = torch.linspace(0, 1, n_steps, device=x0.device)
        
        # Generate straight line paths for all batches at once
        for step in range(n_steps):
            # Compute points for all batches at this step
            points = x0 + t[step] * (xT - x0)
            paths[step] = points
            
            # Debug output for first batch
            if step < 3 or step % 10 == 0:
                print(f"\nStep {step}:")
                print(f"Position: {points[0]}")
                print(f"Distance from start: {torch.norm(points[0] - x0[0]):.6f}")
                print(f"Distance to target: {torch.norm(xT[0] - points[0]):.6f}")
        
        # Reshape to [n_steps * n_batch, n_features]
        path_tensor = paths.reshape(-1, x0.shape[-1])
        

        # For each intermediate point
        for step in range(1, n_steps-1):
            # Current interpolated points for all batches
            x_current = paths[step]  # [n_batch, n_features]
            
            # Compute geometry
            G = self._compute_metric(x_current)
            Gamma = self._compute_christoffel(x_current)
            
            # Straight path velocity
            v_straight = xT - x0
            v_straight = v_straight / torch.norm(v_straight, dim=1, keepdim=True)
            
            # Compute and apply correction
            correction = torch.zeros_like(x_current)
            for mu in range(x_current.shape[1]):
                correction[:, mu] = -(Gamma[:, :, :, mu] * 
                                    v_straight.unsqueeze(2) * 
                                    v_straight.unsqueeze(1)).sum(dim=(1,2))
            
            # Scale correction
            t_factor = t[step] * (1 - t[step])
            scale = torch.clamp(torch.norm(correction, dim=1, keepdim=True), max=0.1)
            correction = correction * t_factor * scale * 0.1
            
            # Update points
            paths[step] = x_current + correction
            
            # Update flattened tensor
            path_tensor = paths.reshape(-1, x0.shape[-1])
            
        return path_tensor
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
            paths = paths.reshape(self.n_steps * input_tensor.shape[0], *input_tensor.shape[1:])
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
            input.shape[1:] for input in inputs
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