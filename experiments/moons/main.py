import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import torch as th
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

from geodesic.geodesic_ig import GeodesicIntegratedGradients
from geodesic.svi_ig import SVI_IG

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
    device: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
    beta: float = 0.3,
    num_iterations: int = 1000,
    n_steps: int = 100,
    learning_rate: float = 0.01,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Loop over noises
    for noise in noises:
        # Create model
        net = Net(
            MLP(units=[2, 10, 10, 2], activation_final="log_softmax"),
            loss="nll",
        )
        if softplus:
            _net = Net(
                MLP(
                    units=[2, 10, 10, 2],
                    activations="softplus",
                    activation_final="log_softmax",
                ),
                loss="nll",
            )
            _net.load_state_dict(net.state_dict())
            net = _net

        if len(os.listdir(model_path)) == 0:
            # Create dataset
            x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)

            # Convert to tensors
            x_train = th.from_numpy(x_train).float()
            x_test = th.from_numpy(x_test).float()
            y_train = th.from_numpy(y_train).long()
            y_test = th.from_numpy(y_test).long()

            # Create dataset and batchify
            train = TensorDataset(x_train, y_train)
            test = TensorDataset(x_test, y_test)

            train_loader = DataLoader(train, batch_size=32, shuffle=True)
            test_loader = DataLoader(test, batch_size=32, shuffle=False)

            
            # Fit model
            trainer = Trainer(
                max_epochs=50,
                accelerator=accelerator,
                devices=device_id,
                deterministic=deterministic,
            )
            trainer.fit(net, train_loader)
            
            # Save model
            th.save(net.state_dict(), os.path.join(model_path, "net.pth"))
            # save data
            th.save(x_train, os.path.join(data_path, "x_train.pth"))
            th.save(x_test, os.path.join(data_path, "x_test.pth"))

            th.save(y_train, os.path.join(data_path, "y_train.pth"))
            th.save(y_test, os.path.join(data_path, "y_test.pth"))
            # save data loader
            th.save(test_loader, os.path.join(data_path, "test_loader.pth"))

            th.save(trainer, os.path.join(model_path, "trainer.pth"))

        else:
            
            net.load_state_dict(th.load(os.path.join(model_path, "net.pth")))
            x_test = th.load(os.path.join(data_path, "x_test.pth"))
            y_test = th.load(os.path.join(data_path, "y_test.pth"))
            test_loader = th.load(os.path.join(data_path, "test_loader.pth"))
            trainer = th.load(os.path.join(model_path, "trainer.pth"))


            # Set model to eval
            net.eval()

            # Set model to device
            net.to(device)

            # Disable cudnn if using cuda accelerator.
            # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
            # for more information.
            if accelerator == "cuda":
                th.backends.cudnn.enabled = False

            # Set data to device
            x_test = x_test.to(device)
            y_test = y_test.to(device)

        # Get predictions
        pred = trainer.predict(net, test_loader)

        # Print accuracy
        acc = (th.cat(pred).argmax(-1) == y_test).float().mean()
        print("acc: ", acc)


        # Create dict of attr
        attr = dict()

        # Set baseline as (-0.5, -0.5)
        baselines = th.zeros_like(x_test).to(device)
        baselines[:, 0] = -0.5
        baselines[:, 1] = -0.5

        if "geodesic_integrated_gradients" in explainers:
            import pyro
            # from geodesic.svi_ig_copy import SVI_IG
            for n in [5]:
                # explainer = GeodesicIntegratedGradients(net)
                # svi_ig = IntegratedGradients(net)

                svi_ig = SVI_IG(net)
                _attr = th.zeros_like(x_test)
                paths = []

                for target in range(2):

                    pyro.clear_param_store()
                    
                    target_mask = y_test == target
                    predictions = net(x_test[target_mask])
                    attribution, gig_path = svi_ig.attribute(
                        x_test[target_mask],
                        baselines=baselines[target_mask],
                        # augmentation_data=x_test[target_mask], # uncomment to use augmentation data
                        target=predictions.argmax(-1), 
                        n_steps=n_steps,
                        n_neighbors=n,
                        beta=beta,
                        num_iterations=num_iterations,
                        learning_rate=learning_rate,
                        method='riemann_trapezoid'
                    )
                    gig_path = gig_path[0]
                    _attr[target_mask] = attribution.float()
                    paths.append(gig_path)

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
                # internal_batch_size=200,
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
                if "geodesic_integrated_gradients" in k:
                    _attr, paths = v
                else:
                    _attr = v
                    paths = None

                scatter = plt.scatter(
                    x_test[:, 0].cpu(),
                    x_test[:, 1].cpu(),
                    c=_attr.abs().sum(-1).detach().cpu(),
                )
                cbar = plt.colorbar(scatter)
                cbar.ax.tick_params(labelsize=20)

                class_index = 0
                
                if paths is not None:
                    class_path = paths[class_index]
                    n_paths_to_plot = 10
                    
                    # Get number of samples for this class
                    trg_mask = y_test == class_index
                    n_samples = x_test[trg_mask].shape[0]
                    
                    # Sample random indices
                    path_indices = th.randint(0, n_samples, (n_paths_to_plot,))

                    for i, idx in enumerate(path_indices):
                        # Extract single path [n_steps, features]
                        n_samples = x_test[trg_mask].shape[0]
                        single_path = class_path.view(n_steps, n_samples, -1)[:, idx, :].detach().numpy()

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
                            label=baseline_label
                        )
                        plt.scatter(
                            single_path[-1, 0], 
                            single_path[-1, 1],
                            color="blue", 
                            marker="o", 
                            s=100,
                            label=input_label
                        )
                # save figure
                plt.savefig(os.path.join(figure_path, f"{k}_{noise}.png"))
                plt.close()

        with open("results.csv", "a") as fp, lock:
            # Write acc
            fp.write(str(seed) + ",")
            fp.write(str(noise) + ",")
            fp.write("softplus," if softplus else "relu,")
            fp.write("acc,")
            fp.write(f"{acc:.4}")
            fp.write("\n")

            # Write purity
            for k, v in attr.items():
                if "geodesic_integrated_gradients" in k:
                    _attr, _ = v
                else:
                    _attr = v

                topk_idx = th.topk(
                    _attr.abs().sum(-1),
                    int(len(_attr.abs().sum(-1)) * 0.5),
                    sorted=False,
                    largest=False,
                ).indices

                fp.write(str(seed) + ",")
                fp.write(str(noise) + ",")
                fp.write("softplus," if softplus else "relu,")
                fp.write(k + ",")
                fp.write(f"{th.cat(pred).argmax(-1)[topk_idx].float().mean():.4},")
                fp.write(f"{_attr.abs().sum(-1)[y_test == 0].std():.4},")
                fp.write(f"{_attr.abs().sum(-1)[y_test == 1].std():.4}")
                fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "integrated_gradients",
            "geodesic_integrated_gradients",
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
        default=0,
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
