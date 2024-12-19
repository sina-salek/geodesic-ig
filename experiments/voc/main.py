import multiprocessing as mp
import os
import torch as th
import torchvision.transforms as T
import warnings

from captum.attr import (
    IntegratedGradients,
)
from captum.metrics import sensitivity_max

from argparse import ArgumentParser
from pytorch_lightning import  seed_everything
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

from geodesic.svi_ig import SVI_IG
from geodesic.geodesic_ig import GeodesicIntegratedGradients

from geodesic.utils.tqdm import get_progress_bars

from experiments.voc.classifier import VocClassifier

file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")

def plot_and_save(tensor, filename, is_attribution=False):

    save_path = os.path.join(
        os.path.join(file_dir, "plots", filename)
    )
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))


    # Denormalize if it's the original image
    if not is_attribution:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor = tensor.cpu().numpy().transpose(1, 2, 0)
        tensor = std * tensor + mean
        tensor = np.clip(tensor, 0, 1)
    else:
        tensor = tensor.cpu().numpy().transpose(1, 2, 0)
        tensor = np.mean(tensor, axis=2)  # Average across channels for attributions

    plt.figure(figsize=(8, 8))
    plt.imshow(tensor)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def generate_augmented_points(x, n_base_points=2500, n_noise_points=2500, noise_std=0.1, device="cuda"):
    # Base interpolation points
    base_points = th.rand((n_base_points,) + x.shape, device=device)
    base_points, _ = base_points.sort(dim=0)
    
    # Generate noise points around interpolation
    noise_points = base_points.unsqueeze(1) + th.randn((n_base_points, n_noise_points//n_base_points) + x.shape, device=device) * noise_std
    noise_points = noise_points.view(-1, *x.shape)
    
    # Combine and sort all points
    all_points = th.cat([base_points, noise_points], dim=0)
    all_points, _ = all_points.sort(dim=0)
    
    # Create augmented data
    x_aug = x.unsqueeze(0) * all_points
    
    return x_aug

def main(
    explainers: List[str],
    device: str = "cpu",
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    # Get accelerator and device
    accelerator = device.split(":")[0]

    
    # Get data transform
    image_size = 32
    centre_crop = 32
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(centre_crop),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    target_transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(centre_crop),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 255).long()),
        ]
    )

    # Load test data
    voc = VOCSegmentation(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "voc",
        ),
        image_set="val",
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    voc_loader = voc_loader = DataLoader(
        voc, 
        batch_size=1, 
        shuffle=True,
        generator=th.Generator().manual_seed(2) 
    )

    # Load model
    resnet = VocClassifier()

    # Switch to eval
    resnet.eval()

    # Set model to device
    resnet.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Get data as tensors
    # we only load 100 images
    x_test = list()
    seg_test = list()
    i = 0
    for data, seg in voc_loader:
        if i == 100:
            break

        seg_ids = seg.unique()
        if len(seg_ids) <= 1:
            continue

        seg_ = seg.clone()
        for j, seg_id in enumerate(seg_ids):
            seg_[seg_ == seg_id] = j

        x_test.append(data)
        seg_test.append(seg_)
        break

    print(f"Number of images loaded: {len(x_test)}")
    print(f"Final x_test shape: {x_test[0].shape}")

    x_test = th.cat(x_test).to(device)
    seg_test = th.cat(seg_test).to(device)

    # Target is the model prediction
    y_test = resnet(x_test).argmax(-1).to(device)

    # Create dict of attributions, explainers, sensitivity max
    # and lipschitz max
    attr = dict()
    expl = dict()

    plot_and_save(x_test[0], "original_image.png")

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()

        for i, (x, y) in get_progress_bars()(
            enumerate(zip(x_test, y_test)),
            total=len(x_test),
            desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
        ):
            x_aug = generate_augmented_points(x, n_base_points=2500, n_noise_points=2500, device=device) 
            explainer = GeodesicIntegratedGradients(
                resnet, 
                data=x_aug, 
                n_neighbors=5
            )

            _attr.append(
                explainer.attribute(
                    x.unsqueeze(0),
                    target=y.item(),
                    internal_batch_size=100,
                )
            )

        
        attr["geodesic_integrated_gradients"] = th.cat(_attr)
        expl["geodesic_integrated_gradients"] = explainer

        plot_and_save(attr["geodesic_integrated_gradients"][0], f"attribution_geodesic_integrated_gradients.png", is_attribution=True)

    if "integrated_gradients" in explainers:
        explainer = IntegratedGradients(resnet)
        _attr = explainer.attribute(
            x_test,
            target=y_test,
            internal_batch_size=200,
        )
        attr["integrated_gradients"] = _attr
        expl["integrated_gradients"] = explainer

        plot_and_save(_attr[0], f"attribution_integrated_gradients.png", is_attribution=True)


    if "svi_integrated_gradients" in explainers:
        explainer = SVI_IG(resnet)
        _attr = explainer.attribute(
            x_test,
            target=y_test,
        )
        attr["svi_integrated_gradients"] = _attr
        expl["svi_integrated_gradients"] = explainer

        plot_and_save(_attr[0], f"attribution_svi_integrated_gradients.png", is_attribution=True)

    if "ode_integrated_gradients" in explainers:
        from geodesic.ode_ig import OdeIG
        explainer = OdeIG(resnet)
        baseline = th.zeros_like(x_test)
        _attr = explainer.attribute(
            x_test,
            baselines=baseline,
            target=y_test,
        )
        attr["ode_integrated_gradients"] = _attr
        expl["ode_integrated_gradients"] = explainer

        plot_and_save(_attr[0], f"attribution_ode_integrated_gradients.png", is_attribution=True)

    # Save attributions
    attr_path = os.path.join(
        os.path.join(file_dir, "plots", "attributions")
    )
    if not os.path.exists(attr_path):
        os.makedirs(attr_path)

    th.save(attr, os.path.join(attr_path, "attributions.pt"))
    th.save(expl, os.path.join(attr_path, "explainers.pt"))



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            # "geodesic_integrated_gradients",
            # "svi_integrated_gradients",
            # "integrated_gradients",
            "ode_integrated_gradients"

        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
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
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )