import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import torch as th
import numpy as np
import warnings

from argparse import ArgumentParser
from matplotlib.colors import ListedColormap
from pytorch_lightning import Trainer, seed_everything
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
)

from geodesic.attr.geodesic_knn_ig import GeodesicIntegratedGradients
from geodesic.attr.geodesic_svi_ig import SVI_IG


import pyro

from geodesic.models.mlp import MLP
from geodesic.models.net import Net


cm_bright = ListedColormap(["#FF0000", "#0000FF"])
warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_path, "weights")
data_path = os.path.join(current_path, "data")
figure_path = os.path.join(current_path, "figures")

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists(figure_path):
    os.makedirs(figure_path)


def main(
    explainers: List[str],
    n_samples: int,
    noises: List[float],
    softplus: bool = False,
    device: str = None,
    seed: int = 42,
    deterministic: bool = False,
    beta: float = 0.3,
    num_iterations: int = 1000,
    n_steps: int = 100,
    learning_rate: float = 0.01,
):
    # Set seed if deterministic
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Device setup
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create lock for multiprocessing
    lock = mp.Lock()

    for noise in noises:
        # Create model
        net = Net(
            MLP(units=[2, 10, 10, 2], activation_final="log_softmax"),
            loss="nll",
        ).to(device)

        if softplus:
            _net = Net(
                MLP(
                    units=[2, 10, 10, 2],
                    activations="softplus",
                    activation_final="log_softmax",
                ),
                loss="nll",
            ).to(device)
            _net.load_state_dict(net.state_dict())
            net = _net

        if len(os.listdir(model_path)) == 0:
            # Create and process dataset
            x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)

            # Convert to tensors on device
            x_train = th.from_numpy(x_train).float().to(device)
            x_test = th.from_numpy(x_test).float().to(device)
            y_train = th.from_numpy(y_train).long().to(device)
            y_test = th.from_numpy(y_test).long().to(device)

            # Create dataloaders
            train = TensorDataset(x_train, y_train)
            test = TensorDataset(x_test, y_test)
            train_loader = DataLoader(train, batch_size=32, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test, batch_size=32, shuffle=False, pin_memory=True)

            # Train model
            print(f"trainer device: {device}")
            trainer = Trainer(
                max_epochs=50,
                accelerator="gpu" if device.startswith("cuda") else "cpu",
                devices=1,
                deterministic=deterministic,
            )
            trainer.fit(net, train_loader)

            # Save model and data
            th.save(net.cpu().state_dict(), os.path.join(model_path, "net.pth"))
            th.save(x_train.cpu(), os.path.join(data_path, "x_train.pth"))
            th.save(x_test.cpu(), os.path.join(data_path, "x_test.pth"))
            th.save(y_train.cpu(), os.path.join(data_path, "y_train.pth"))
            th.save(y_test.cpu(), os.path.join(data_path, "y_test.pth"))
            th.save(test_loader, os.path.join(data_path, "test_loader.pth"))
            th.save(trainer, os.path.join(model_path, "trainer.pth"))
            
            # Move model back to device
            net = net.to(device)
        else:
            # Load saved model and data
            net.load_state_dict(th.load(os.path.join(model_path, "net.pth")))
            x_test = th.load(os.path.join(data_path, "x_test.pth")).to(device)
            y_test = th.load(os.path.join(data_path, "y_test.pth")).to(device)
            test_loader = th.load(os.path.join(data_path, "test_loader.pth"))
            trainer = th.load(os.path.join(model_path, "trainer.pth"))

            # Prepare model
            net.eval().to(device)

            # # Disable cudnn for CUDA if needed
            # if device.startswith("cuda"):
            #     th.backends.cudnn.enabled = False

        print(f"device at prediction: {device}. \n net device at prediction: {net.device}")
        # Get predictions (already on GPU)
        pred = trainer.predict(net, test_loader)

        attr = dict()
        # Create baselines on same device
        baselines = th.zeros_like(x_test).to(device)
        baselines[:, 0] = -0.5
        baselines[:, 1] = -0.5

        # # Get predictions with proper device management
        # with th.no_grad():  # Add for inference
        #     data_probs = net(x_test.to(device)).detach().cpu().numpy()
        #     baseline_probs = net(baselines.to(device)).detach().cpu().numpy()

        # # Calculate probability differences (now on CPU)
        # prob_diff = (data_probs - baseline_probs)[:, 0]

        # # Create scatter plot with CPU tensors
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(
        #     x_test.cpu().detach().numpy()[:, 0], 
        #     x_test.cpu().detach().numpy()[:, 1], 
        #     c=prob_diff,
        #     cmap="viridis",
        #     s=50
        # )
        # plt.colorbar(scatter, label="Probability Difference")
        # plt.xlabel("Feature 1")
        # plt.ylabel("Feature 2")
        # plt.title("Data Points Colored by Model-Baseline Probability Difference")
        # plt.savefig(os.path.join(figure_path, f"prob_diff_{noise}.png"))
        # plt.close()

        if "svi_integrated_gradients" in explainers:
            print("Running SVI-IG")
            linear_interpolation = [True, False]
            endpoint_matching = [True, False]
            for li in linear_interpolation:
                for em in endpoint_matching:
                        
                    explainer = SVI_IG(net)
                    _attr = th.zeros_like(x_test)
                    paths = []
                    predictions = net(x_test).argmax(-1)
                    attribution, gig_path = explainer.attribute(
                        x_test,
                        baselines=baselines,
                        target=predictions,
                        num_iterations=num_iterations,
                        learning_rate=learning_rate,
                        beta=beta,
                        n_steps=n_steps,
                        do_linear_interp=li,
                        use_endpoints_matching=em,
                        return_paths=True,
                    )


                    if gig_path is not None:
                        gig_path = gig_path[0]
                        paths.append(gig_path)
                    else:
                        paths = None
                    _attr = attribution.float()
                    attr[f"svi_integrated_gradients_{em}_{li}"] = (_attr, paths)

        if "ode_integrated_gradients" in explainers:
            raise NotImplementedError("ODE-IG is not implemented yet.")
        if "geodesic_integrated_gradients" in explainers:
            for n in [5]:
                geodesic_ig = GeodesicIntegratedGradients(net)

                _attr = th.zeros_like(x_test)
                paths = []
                predictions = net(x_test).argmax(-1)
                for target in range(2):

                    pyro.clear_param_store()

                    target_mask = predictions == target
                    gig_path = None
                    attribution = geodesic_ig.attribute(
                        x_test[target_mask],
                        baselines=baselines[target_mask],
                        target=target,
                        n_steps=n_steps,
                        n_neighbors=n,
                    )
                    if gig_path is not None:
                        gig_path = gig_path[0]
                        paths.append(gig_path)
                    else:
                        paths = None
                    _attr[target_mask] = attribution.float()

                attr[f"geodesic_integrated_gradients_{str(n)}"] = (_attr, paths)

        if "enhanced_integrated_gradients" in explainers:
            for n in range(5, 20, 5):
                explainer = GeodesicIntegratedGradients(net)
                _attr = th.zeros_like(x_test)

                for target in range(2):
                    predictions = net(x_test[y_test == target])
                    _attr[y_test == target] = explainer.attribute(
                        x_test[y_test == target],
                        baselines=baselines[y_test == target],
                        target=predictions.argmax(-1),
                        n_neighbors=n,
                        internal_batch_size=200,
                        distance="euclidean",
                    ).float()

                attr[f"enhanced_integrated_gradients_{n}"] = _attr

        if "gradient_shap" in explainers:
            explainer = GradientShap(net)
            attr["gradient_shap"] = explainer.attribute(
                x_test,
                target=y_test,
                baselines=baselines,
                n_samples=50,
            )

        if "integrated_gradients" in explainers:
            explainer = IntegratedGradients(net)
            attr["integrated_gradients"] = explainer.attribute(
                x_test,
                baselines=baselines,
                target=y_test,
                internal_batch_size=200,
            )

        if "smooth_grad" in explainers:
            explainer = NoiseTunnel(IntegratedGradients(net))
            attr["smooth_grad"] = explainer.attribute(
                x_test,
                baselines=baselines,
                target=y_test,
                internal_batch_size=200,
                nt_samples=10,
                stdevs=0.1,
            )

        # Eval
        with lock:
            for k, v in attr.items():
                if type(v) is tuple:
                    _attr, paths = v
                else:
                    _attr = v
                    paths = None

                scatter = plt.scatter(
                    x_test[:, 0].detach().cpu(),
                    x_test[:, 1].detach().cpu(),
                    c=_attr.abs().sum(-1).detach().cpu(),
                )
                cbar = plt.colorbar(scatter)
                cbar.ax.tick_params(labelsize=20)

                x_min, x_max = x_test[:, 0].min().item(), x_test[:, 0].max().item()
                y_min, y_max = x_test[:, 1].min().item(), x_test[:, 1].max().item()

                # Create mesh grid
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
                )

                # Stack coordinates
                mesh_coords = th.FloatTensor(
                    np.column_stack([xx.ravel(), yy.ravel()])
                ).requires_grad_(True)

                # Get gradients across mesh

                mesh_preds = net(mesh_coords)
                mesh_grads = th.autograd.grad(mesh_preds.sum(), mesh_coords)[0]

                grad_magnitude = mesh_grads.norm(dim=1).reshape(100, 100)

                # Add contour plot
                plt.contour(
                    xx, yy, grad_magnitude.cpu(), levels=10, cmap="viridis", alpha=0.5
                )

                if paths is not None:

                    n_paths_to_plot = 10

                    # Get number of samples for this class
                    n_samples = x_test.shape[0]

                    # Sample random indices
                    path_indices = th.randint(0, n_samples, (n_paths_to_plot,))

                    for i, idx in enumerate(path_indices):
                        # Extract single path [n_steps, features]
                        single_path = (
                            paths[0]
                            .view(n_steps, n_samples, -1)[:, idx, :]
                            .detach()
                            .cpu()
                        )

                        # Plot path
                        plt.plot(
                            single_path[:, 0],
                            single_path[:, 1],
                            linestyle="--",
                            color="gray",
                            marker="o",
                            alpha=0.7,
                        )

                        # Plot endpoints
                        baseline_label = "Baseline" if i == 0 else None
                        input_label = "Input" if i == 0 else None

                        plt.scatter(
                            single_path[0, 0],
                            single_path[0, 1],
                            color="red",
                            marker="x",
                            s=100,
                            label=baseline_label,
                        )
                        plt.scatter(
                            single_path[-1, 0],
                            single_path[-1, 1],
                            color="blue",
                            marker="o",
                            s=100,
                            label=input_label,
                        )
                # save figure
                plt.savefig(os.path.join(figure_path, f"{k}_{noise}.png"))
                plt.close()

        with open("results.csv", "a") as fp, lock:
            fp.write(str(seed) + ",")
            fp.write(str(noise) + ",")
            fp.write("softplus," if softplus else "relu,")
            fp.write("\n")

            # Write purity
            # for k, v in attr.items():
            #     if type(v) is tuple:
            #         _attr, _ = v
            #     else:
            #         _attr = v

            #     topk_idx = th.topk(
            #         _attr.abs().sum(-1),
            #         int(len(_attr.abs().sum(-1)) * 0.5),
            #         sorted=False,
            #         largest=False,
            #     ).indices

            #     fp.write(str(seed) + ",")
            #     fp.write(str(noise) + ",")
            #     fp.write("softplus," if softplus else "relu,")
            #     fp.write(k + ",")
            #     fp.write(f"{th.cat(pred).argmax(-1)[topk_idx].float().mean():.4},")
            #     fp.write(f"{_attr.abs().sum(-1)[y_test == 0].std():.4},")
            #     fp.write(f"{_attr.abs().sum(-1)[y_test == 1].std():.4}")
            #     fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            # "integrated_gradients",
            # "geodesic_integrated_gradients",
            "svi_integrated_gradients",
        ],
        nargs="+",
        metavar="N",
        help="List of explainers to use.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples in the dataset.",
    )
    parser.add_argument(
        "--noises",
        type=float,
        default=[0.1],
        nargs="+",
        metavar="N",
        help="List of noises to use.",
    )
    parser.add_argument(
        "--softplus",
        action="store_true",
        help="Whether to replace relu with softplus or not.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta parameter for the potential energy. Used in the SVI-IG.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of iterations for the optimization. Used in the SVI-IG.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="Number of points generated along the geodesic path. Used in the SVI-IG.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimization. Used in the SVI-IG.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        n_samples=args.n_samples,
        noises=args.noises,
        softplus=args.softplus,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
        beta=args.beta,
        num_iterations=args.num_iterations,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
    )
