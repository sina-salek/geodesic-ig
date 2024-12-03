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
    
    def _get_approx_paths(self, inputs: Tensor, baselines: Tensor, augmentation_data: Tensor, n_steps: int, n_neighbors: int) -> Tensor:
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
        import networkx as nx
        from scipy.interpolate import interp1d
        import numpy as np

        device = inputs.device
        dtype = inputs.dtype

        inputs_np = inputs.detach().cpu().numpy()
        baselines_np = baselines.detach().cpu().numpy()
        aug_np = augmentation_data.detach().cpu().numpy()
        
        def edge_weight(p1, p2):
            # Convert points to tensors with correct dtype
            p1_tensor = torch.tensor(p1, device=device, dtype=dtype, requires_grad=True)
            p2_tensor = torch.tensor(p2, device=device, dtype=dtype, requires_grad=True)
            
            alpha = torch.linspace(0, 1, steps=10, device=device, dtype=dtype)
            path_points = p1_tensor + alpha.view(-1, 1) * (p2_tensor - p1_tensor)
            
            outputs = self.forward_func(path_points)
            grads = torch.autograd.grad(
                outputs.sum(), path_points,
                create_graph=True, retain_graph=True
            )[0]
            
            return float(torch.norm(grads, p=2, dim=-1).sum().cpu())
        
        batch_paths = []
        for i in range(len(inputs)):
            # Create graph with augmentation points and current input/baseline
            G = nx.Graph()
            
            # Add all points including current input/baseline
            all_points = np.vstack([aug_np, baselines_np[i:i+1], inputs_np[i:i+1]])  # Changed order: baseline first, then input
            for j in range(len(all_points)):
                G.add_node(j, pos=all_points[j])
            
            # Add edges between nearby points with gradient-based weights
            for j in range(len(all_points)):
                distances = np.linalg.norm(all_points - all_points[j], axis=1)
                nearest = np.argsort(distances)[1:n_neighbors+1]  # Exclude self
                
                for n in nearest:
                    if not G.has_edge(j, n):
                        weight = edge_weight(all_points[j], all_points[n])
                        G.add_edge(j, n, weight=weight)
            
            # Find shortest path using A*
            start_node = len(aug_np)  # Index of baseline point (first after augmentation)
            end_node = len(aug_np) + 1  # Index of input point (second after augmentation)
            
            try:
                path_indices = nx.astar_path(G, start_node, end_node, weight='weight')
                path_points = np.array([G.nodes[i]['pos'] for i in path_indices])
                
                # Interpolate to get n_steps points
                path_length = np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
                path_length = np.insert(path_length, 0, 0)
                
                kind = 'linear'
                if len(path_points) >= 4:
                    kind = 'cubic'
                    
                try:
                    interp_funcs = [
                        interp1d(path_length, path_points[:, j], kind=kind)
                        for j in range(path_points.shape[1])
                    ]
                    
                    even_spacing = np.linspace(0, path_length[-1], n_steps)
                    interpolated_path = np.column_stack([
                        f(even_spacing) for f in interp_funcs
                    ])
                    
                    batch_paths.append(interpolated_path)
                    
                except ValueError as e:
                    print(f"Interpolation failed for sample {i}, using straight line. Error: {e}")
                    straight_line = self._get_straight_line(
                        inputs[i:i+1], baselines[i:i+1], n_steps
                    )
                    batch_paths.append(straight_line.squeeze(1).cpu().numpy())
                    
            except nx.NetworkXNoPath:
                print(f"No path found for sample {i}, using straight line")
                straight_line = self._get_straight_line(
                    inputs[i:i+1], baselines[i:i+1], n_steps
                )
                batch_paths.append(straight_line.squeeze(1).cpu().numpy())
        
        # Stack [batch_size, n_steps, features] 
        paths = torch.tensor(np.stack(batch_paths, axis=0), device=device, dtype=dtype)  
        paths = paths.transpose(0, 1)  # Transpose to get [n_steps, batch_size, features]
        
        assert torch.allclose(paths[0], baselines), f"Paths don't start at baselines. Got {paths[0]} vs {baselines}"
        assert torch.allclose(paths[-1], inputs), f"Paths don't end at inputs. Got {paths[-1]} vs {inputs}"
        
        return paths

    def potential_energy(self, path: Tensor, initial_paths: Tensor, beta: float) -> Tensor:
        """
        The potential energy encourages the path between input and baseline pairs to be as short as possible, whilst
        avoiding regions of high curvature. 
        """
        # Distance penalty: How far the path deviates from the initial path
        # TODO: Not necessarily the best choice of distance metric for non-straight paths. Change in the next version.
        distance_penalty = torch.norm(path - initial_paths, p=2, dim=-1)  
        
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
    
    def model(self, inputs: Tensor, baselines: Tensor, initial_paths: Tensor, beta: float):
        input_tensor = inputs[0]
        dtype = input_tensor.dtype
        batch_size = input_tensor.size(0)

        if self.last_batch_size is not None and self.last_batch_size != batch_size:
            print(f"Batch size changed from {self.last_batch_size} to {batch_size}, clearing param store")
            pyro.clear_param_store()
        self.last_batch_size = batch_size
        
        alphas = torch.linspace(0, 1, steps=self.n_steps, 
                        device=input_tensor.device,
                        dtype=dtype).view(-1, 1, 1)
        
        # Sample path deviations
        delta = pyro.sample(
            "path_delta",
            dist.Normal(torch.zeros_like(initial_paths), 1.0).to_event(initial_paths.dim())
        )
        
        # Zero out the deviations at start and end points
        delta = delta * (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        
        # The optimized path is the initial path plus deviations
        path = initial_paths + delta
        path.requires_grad_()
        
        # Compute the potential energy of the path
        energy = self.potential_energy(path, initial_paths, beta)
        
        # Include the potential energy in the model's log probability
        pyro.factor("energy", -energy.sum())

    def guide(self, inputs: Tensor, baselines: Tensor, initial_paths: Tensor, beta: float):
        input_tensor = inputs[0]
        dtype = input_tensor.dtype
        batch_size = input_tensor.shape[0] 

        if hasattr(self, 'last_batch_size') and self.last_batch_size != batch_size:
            pyro.clear_param_store()
        self.last_batch_size = batch_size
        
        alphas = torch.linspace(0, 1, steps=self.n_steps, 
                        device=input_tensor.device,
                        dtype=dtype).view(-1, 1, 1)
        
        # Initialize parameters with appropriate shapes
        delta_loc = pyro.param(
            "delta_loc",
            lambda: torch.zeros_like(initial_paths)
        )
        
        # Zero out the location parameters at start and end points
        delta_loc = delta_loc * (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        
        # For scale, use a small positive value at endpoints
        delta_scale = pyro.param(
            "delta_scale",
            lambda: 0.1 * torch.ones_like(initial_paths),
            constraint=dist.constraints.positive
        )
        
        # Instead of zeroing scale, make it very small at endpoints
        endpoint_mask = (1 - (alphas == 0).float()) * (1 - (alphas == 1).float())
        delta_scale = delta_scale * endpoint_mask + 1e-6 * (1 - endpoint_mask)
        
        # Sample the path deviations
        pyro.sample(
            "path_delta",
            dist.Normal(delta_loc, delta_scale).to_event(initial_paths.dim())
        )

    def _optimize_paths(
        self,
        inputs: Tensor,
        baselines: Tensor,
        augmentation_data: Tensor,
        n_neighbors: int,
        beta: float,
        num_iterations: int = 1000,
        lr: float = 1e-2,
    ) -> Tensor:
        input_tensor = inputs[0]
        baseline_tensor = baselines[0]
        
        if augmentation_data is not None:
            initial_paths = self._get_approx_paths(input_tensor, baseline_tensor, augmentation_data, self.n_steps, n_neighbors)
            beta = 1/beta if beta > 1 else beta
            current_beta = beta * 10
            decay_rate = (current_beta * beta) ** (1 / num_iterations)
        else:
            initial_paths = self._get_straight_line(input_tensor, baseline_tensor, self.n_steps)
            current_beta = beta
            decay_rate = 1
        
        
        optimizer = Adam({"lr": lr})
        svi = SVI(
            lambda inputs, baselines, initial_paths, beta: self.model(inputs, baselines, initial_paths, beta),
            lambda inputs, baselines, initial_paths, beta: self.guide(inputs, baselines, initial_paths, beta),
            optimizer, 
            loss=Trace_ELBO()
        )
        
        for i in range(num_iterations):
            current_beta *= decay_rate
            loss = svi.step(inputs, baselines, initial_paths, current_beta)
            
            if i % 100 == 0:
                print(f"Iteration {i} - Loss: {loss} - Beta: {current_beta}")
        
        delta_opt = pyro.param("delta_loc").detach()
        optimized_path = initial_paths + delta_opt

        # Whilst experimenting, I kept getting paths that didn't start or end at the correct points. I fixed it, but also 
        # added some debugging information to help me understand what was going on, in case it happens again.
        rtol = 1e-4  # Relative tolerance
        atol = 1e-4  # Absolute tolerance

        # Debug information with more detail
        if not torch.allclose(optimized_path[-1], input_tensor, rtol=rtol, atol=atol):
            max_diff = (optimized_path[-1] - input_tensor).abs().max()
            mean_diff = (optimized_path[-1] - input_tensor).abs().mean()
            print(f"Maximum difference: {max_diff}")
            print(f"Mean difference: {mean_diff}")
            print(f"Shape of input_tensor: {input_tensor.shape}")
            print(f"Shape of optimized_path: {optimized_path.shape}")

        assert torch.allclose(optimized_path[0], baseline_tensor, rtol=rtol, atol=atol), "Path doesn't start at baseline"
        assert torch.allclose(optimized_path[-1], input_tensor, rtol=rtol, atol=atol), "Path doesn't end at input"

        return optimized_path

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
            raise ValueError("Augmentation data is provided, but no n_neighbors is given. Please provide a n_neighbors.")
        if augmentation_data is None and n_neighbors is not None:
            warnings.warn("n_neighbors is provided, but no augmentation data is given. Ignoring n_neighbors.")
        
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

        optimized_paths = self._optimize_paths(inputs, baselines, augmentation_data, n_neighbors, beta, num_iterations, lr=learning_rate)

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
            



        

