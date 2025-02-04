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
    IntegratedGradients,
    KernelShap,
    GradientShap,
    InputXGradient,
    NoiseTunnel,
)
import saliency.core as saliency

from geodesic.attr import GeodesicIGKNN, GeodesicIGSVI, Occlusion, AugmentedOcclusion
from geodesic.models import MLP, Net
from geodesic.utils.tqdm import get_progress_bars


cm_bright = ListedColormap(["#FF0000", "#0000FF"])
warnings.filterwarnings("ignore")


def main(
    explainers: List[str],
    n_samples: int,
    noises: List[float],
    n_steps: int,
    softplus: bool = False,
    device: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
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

        # Create model
        net = Net(
            MLP(units=[2, 10, 10, 2], activation_final="log_softmax"),
            loss="nll",
        )

        # Fit model
        trainer = Trainer(
            max_epochs=50,
            accelerator=accelerator,
            devices=device_id,
            deterministic=deterministic,
        )
        trainer.fit(net, train_loader)

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
        pred_class = th.cat(pred).argmax(-1)

        # Print accuracy
        acc = (th.cat(pred).argmax(-1) == y_test).float().mean()
        print("acc: ", acc)

        # Create dir to save figures
        with lock:
            path = f"plots/{'softplus' if softplus else 'relu'}/{str(seed)}"
            os.makedirs(path, exist_ok=True)

            # Set ticks size
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)

            # Save plots of true values and predictions
            scatter = plt.scatter(
                x_test[:, 0].cpu(),
                x_test[:, 1].cpu(),
                c=y_test.cpu(),
                cmap=cm_bright,
                edgecolors="k",
            )
            plt.legend(*scatter.legend_elements(), fontsize=20)
            plt.savefig(f"{path}/true_labels_{str(noise)}.pdf")
            plt.close()

            scatter = plt.scatter(
                x_test[:, 0].cpu(),
                x_test[:, 1].cpu(),
                c=th.cat(pred).argmax(-1).cpu(),
                cmap=cm_bright,
                edgecolors="k",
            )
            plt.legend(*scatter.legend_elements(), fontsize=20)
            plt.savefig(f"{path}/preds_{str(noise)}.pdf")
            plt.close()

        # Create dict of attr
        attr = dict()

        # Set baseline as (-0.5, -0.5)
        baselines = th.zeros_like(x_test).to(device)
        baselines[:, 0] = -0.5
        baselines[:, 1] = -0.5

        if "geodesic_integrated_gradients" in explainers:
            for n in range(5, 20, 5):
                explainer = GeodesicIGKNN(net)
                _attr = th.zeros_like(x_test)

                for target in range(2):
                    _attr[pred_class == target] = explainer.attribute(
                        x_test[pred_class == target],
                        baselines=baselines[pred_class == target],
                        target=target,
                        n_neighbors=n,
                        n_steps=n_steps,
                        internal_batch_size=200,
                    ).float()

                attr[f"geodesic_integrated_gradients_{str(n)}"] = _attr

        if "enhanced_integrated_gradients" in explainers:
            for n in range(5, 20, 5):
                explainer = GeodesicIGKNN(net)
                _attr = th.zeros_like(x_test)

                for target in range(2):
                    _attr[pred_class == target] = explainer.attribute(
                        x_test[pred_class == target],
                        baselines=baselines[pred_class == target],
                        target=target,
                        n_neighbors=n,
                        n_steps=n_steps,
                        internal_batch_size=200,
                        distance="euclidean",
                    ).float()

                attr[f"enhanced_integrated_gradients_{n}"] = _attr

        if "gradient_shap" in explainers:
            explainer = GradientShap(net)
            attr["gradient_shap"] = explainer.attribute(
                x_test,
                target=pred_class,
                baselines=baselines,
                n_samples=50,
            )

        if "integrated_gradients" in explainers:
            explainer = IntegratedGradients(net)
            attr["integrated_gradients"] = explainer.attribute(
                x_test,
                baselines=baselines,
                target=pred_class,
                internal_batch_size=200,
                n_steps=n_steps,
            )

        if "smooth_grad" in explainers:
            explainer = NoiseTunnel(IntegratedGradients(net))
            attr["smooth_grad"] = explainer.attribute(
                x_test,
                baselines=baselines,
                target=pred_class,
                internal_batch_size=200,
                nt_samples=10,
                stdevs=0.1,
            )
        if "input_x_gradient" in explainers:
            explainer = InputXGradient(net)
            attr["input_x_gradient"] = explainer.attribute(
                x_test,
                target=pred_class,
            )
        if "kernel_shap" in explainers:
            explainer = KernelShap(net)
            attr["kernel_shap"] = explainer.attribute(
                x_test,
                target=pred_class,
                baselines=baselines,
            )
        if "svi_integrated_gradients" in explainers:
            explainer = GeodesicIGSVI(net)

            attr["svi_integrated_gradients"] = explainer.attribute(
                x_test,
                target=pred_class,
                baselines=baselines,
                num_iterations=1000,
                beta=0.1,
                n_steps=n_steps,
                do_linear_interp=True,
                use_endpoints_matching=True,
                learning_rate=0.001,
            )
        if "guided_integrated_gradients" in explainers:
            guided_ig = saliency.GuidedIG()
            def PreprocessData(data):
                data = np.array(data)
                data = th.tensor(data, dtype=th.float32)
                return data.requires_grad_(True)


            def call_model_function(data, call_model_args=None, expected_keys=None):
                data = PreprocessData(data)
                target_class_idx = call_model_args[class_idx_str]
                output = net(data)
                m = th.nn.Softmax(dim=1)
                output = m(output)
                if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
                    outputs = output[:, target_class_idx]
                    grads = th.autograd.grad(
                        outputs, data, grad_outputs=th.ones_like(outputs)
                    )[0]
                    gradients = grads.detach().numpy()
                    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

            _attr = list()
            for i, (x, c, b) in get_progress_bars()(
                enumerate(zip(x_test, pred_class, baselines)),
                total=len(x_test),
                desc="Guided Integrated Gradients",
            ):
                predictions = net(x.unsqueeze(0))
                predictions = predictions.detach().numpy()
                prediction_class = np.argmax(predictions[0])
                class_idx_str = "class_idx_str"
                call_model_args = {class_idx_str: prediction_class}

                _attr.append(
                    th.from_numpy(
                        guided_ig.GetMask(
                                x.numpy(),  
                                call_model_function,
                                call_model_args,
                                x_steps=n_steps,
                                x_baseline=b,  
                                max_dist=1.0,
                                fraction=0.5,
                            )
                    )
                )
            attr["guided_integrated_gradients"] = th.stack(_attr)

        if "occlusion" in explainers:
            explainer = Occlusion(net)
            attr["occlusion"] = explainer.attribute(
                x_test,
                target=pred_class,
                baselines=baselines,
                sliding_window_shapes=(2,),
                strides=(1,),
                attributions_fn=abs,
            )

        if "augmented_occlusion" in explainers:
            explainer = AugmentedOcclusion(net, data=x_test)
            attr["augmented_occlusion"] = explainer.attribute(
                x_test,
                target=pred_class,
                sliding_window_shapes=(2,),
                strides=(1,),
                attributions_fn=abs,
            )

        if "random" in explainers:
            attr["random"] = th.randn_like(x_test)

        # Eval
        with lock:
            for k, v in attr.items():
                scatter = plt.scatter(
                    x_test[:, 0].cpu(),
                    x_test[:, 1].cpu(),
                    c=v.abs().sum(-1).detach().cpu(),
                )
                cbar = plt.colorbar(scatter)
                cbar.ax.tick_params(labelsize=20)
                plt.savefig(f"{path}/{k}_{str(noise)}.pdf")
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
                topk_idx = th.topk(
                    v.abs().sum(-1),
                    int(len(v.abs().sum(-1)) * 0.5),
                    sorted=False,
                    largest=False,
                ).indices

                fp.write(str(seed) + ",")
                fp.write(str(noise) + ",")
                fp.write("softplus," if softplus else "relu,")
                fp.write(k + ",")
                fp.write(f"{th.cat(pred).argmax(-1)[topk_idx].float().mean():.4},")
                fp.write(f"{v.abs().sum(-1)[y_test == 0].std():.4},")
                fp.write(f"{v.abs().sum(-1)[y_test == 1].std():.4}")
                fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "geodesic_integrated_gradients",
            "enhanced_integrated_gradients",
            "gradient_shap",
            "integrated_gradients",
            "smooth_grad",
            "input_x_gradient",
            "kernel_shap",
            "svi_integrated_gradients",
            "guided_integrated_gradients",
            "augmented_occlusion",
            "occlusion",
            "random",
        ],
        nargs="+",
        metavar="N",
        help="List of explainers to use.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples in the dataset.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of steps for the IG family of methods.",
    )
    parser.add_argument(
        "--noises",
        type=float,
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        noises=args.noises,
        softplus=args.softplus,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
