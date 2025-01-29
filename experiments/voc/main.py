from typing import Any, List

import multiprocessing as mp
import subprocess
import os
import logging
import torch as th
import torchvision.transforms as T
import warnings
from torch import nn

from captum.attr import (
    IntegratedGradients,
    KernelShap,
    GradientShap,
    InputXGradient,
    NoiseTunnel,
)
import saliency.core as saliency

from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

import numpy as np

from geodesic.attr import GeodesicIGSVI, Occlusion, AugmentedOcclusion

from geodesic.utils.tqdm import get_progress_bars

from experiments.voc.constants import VALID_BACKBONE_NAMES
from experiments.voc.train_classifier import setup_model

from geodesic.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)
warnings.filterwarnings("ignore")


class ModelWithSoftmax(nn.Module):
    def __init__(self, base_model, input_dim=20, output_dim=20, add_linear=False):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.add_linear = add_linear

    def forward(self, x):
        logits = self.base_model(x)
        if self.add_linear:
            logits = self.linear(logits)
        return self.sigmoid(logits)


def main(
    explainers: List[str],
    areas: List[float],
    n_steps: int = 50,
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
    # TODO: Transformations in the final experiment should 224x224. Here we use 128x128 for speed.
    image_size = 128
    centre_crop = 128
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(centre_crop),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test data
    voc = VOCDetection(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "geodesic-ig",
            "data",
            "voc",
        ),
        image_set="val",
        transform=transform,
        download=True,
    )

    voc_loader = DataLoader(
        voc, batch_size=1, shuffle=True, generator=th.Generator().manual_seed(2)
    )

    def load_model(checkpoint_path, add_softmax, add_linear=False):
        # Initialise model
        model = setup_model()

        # Load checkpoint
        checkpoint = (
            th.load(checkpoint_path, map_location=th.device("cpu"))
            if accelerator == "cpu"
            else th.load(checkpoint_path)
        )

        # Extract model state dict from checkpoint
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if add_softmax:
            if add_linear:
                model = ModelWithSoftmax(model, add_linear=True)
            else:
                model = ModelWithSoftmax(model)

        model.eval()

        # Move to GPU if available
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model = model.to(device)
        return model

    models = dict()
    # Load models
    for model_name in VALID_BACKBONE_NAMES:
        model_path = os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "voc",
            "checkpoints",
            model_name,
            "best_model.pt",
        )

        # if file does not exist, train a new model
        if not os.path.exists(model_path):
            print(f"Model {model_name} does not exist. Training a new model...")
            subprocess.run(
                ["python", "train_classifier.py", "--model_name", model_name]
            )

        else:
            model_with_softmax = (
                load_model(model_path, add_softmax=True).eval().to(device)
            )
            model_without_softmax = (
                load_model(model_path, add_softmax=False).eval().to(device)
            )

        heads = {
            "model_with_softmax": model_with_softmax,
            "model_without_softmax": model_without_softmax,
        }
        models[model_name] = heads

        print(f"Loaded model: {model_name}")

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Get data as tensors
    x_test = list()
    i = 0
    for data, target in voc_loader:
        if i == 1:
            break

        x_test.append(data)
        i += 1

    print(f"Number of test samples: {len(x_test)}")

    x_test = th.cat(x_test).to(device)
    # Baseline is a normalised black image
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    baselines = normalizer(th.zeros_like(x_test))

    # Create dict of attributions, explainers
    attr = dict()

    now = np.datetime64("now").astype(str)

    if "kernel_shap" in explainers:
        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = KernelShap(model)
            _attr = list()
            for i, (x, y, b) in get_progress_bars()(
                enumerate(zip(x_test, y_test, baselines)),
                total=len(x_test),
                desc=f"{KernelShap.get_name()} attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        baselines=b.unsqueeze(0),
                        target=y.item(),
                    ).squeeze(0)
                )
            attr[f"kernel_shap_{model_name}"] = th.stack(_attr)

    if "gradient_shap" in explainers:
        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = GradientShap(model)
            _attr = list()
            for i, (x, y, b) in get_progress_bars()(
                enumerate(zip(x_test, y_test, baselines)),
                total=len(x_test),
                desc=f"{GradientShap.get_name()} attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        baselines=b.unsqueeze(0),
                        target=y.item(),
                    ).squeeze(0)
                )
            attr[f"gradient_shap_{model_name}"] = th.stack(_attr)

    if "integrated_gradients" in explainers:

        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = IntegratedGradients(model)
            _attr = list()
            for i, (x, y, b) in get_progress_bars()(
                enumerate(zip(x_test, y_test, baselines)),
                total=len(x_test),
                desc=f"{IntegratedGradients.get_name()} attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        baselines=b.unsqueeze(0),
                        target=y.item(),
                        n_steps=n_steps,
                    ).squeeze(0)
                )
            attr[f"integrated_gradients_{model_name}"] = th.stack(_attr)

    if "guided_integrated_gradients" in explainers:
        guided_ig = saliency.GuidedIG()

        transformer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def PreprocessImages(images):
            # assumes input is 4-D, with range [0,255]
            #
            # torchvision have color channel as first dimension
            # with normalization relative to mean/std of ImageNet:
            #    https://pytorch.org/vision/stable/models.html
            images = np.array(images)
            images = images / 255
            images = np.transpose(images, (0, 3, 1, 2))
            images = th.tensor(images, dtype=th.float32)
            images = transformer.forward(images)
            return images.requires_grad_(True)

        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]

            def call_model_function(images, call_model_args=None, expected_keys=None):
                images = PreprocessImages(images)
                target_class_idx = call_model_args[class_idx_str]
                output = model(images)
                m = th.nn.Softmax(dim=1)
                output = m(output)
                if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
                    outputs = output[:, target_class_idx]
                    grads = th.autograd.grad(
                        outputs, images, grad_outputs=th.ones_like(outputs)
                    )
                    grads = th.movedim(grads[0], 1, 3)
                    gradients = grads.detach().numpy()
                    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            _attr = list()
            for i, (x, y, b) in get_progress_bars()(
                enumerate(zip(x_test, y_test, baselines)),
                total=len(x_test),
                desc=f"guided IG attribution",
            ):

                predictions = model(x.unsqueeze(0))
                predictions = predictions.detach().numpy()
                prediction_class = np.argmax(predictions[0])
                class_idx_str = "class_idx_str"
                call_model_args = {class_idx_str: prediction_class}
                x = x.permute(1, 2, 0)
                baseline = b.permute(1, 2, 0)
                _attr.append(
                    th.from_numpy(
                        guided_ig.GetMask(
                            x,
                            call_model_function,
                            call_model_args,
                            x_steps=n_steps,
                            x_baseline=baseline,
                            max_dist=1.0,
                            fraction=0.5,
                        )
                    ).permute(2, 0, 1)
                )
            attr[f"guided_integrated_gradients_{model_name}"] = th.stack(_attr)

    if "svi_integrated_gradients" in explainers:
        # TODO: move the hyperparameters to argparse
        num_iterations = 500
        beta = 0.1
        linear_interpolation = [False]
        endpoint_matching = [True]
        learning_rate_decay = True
        learning_rate = 0.01
        for li in linear_interpolation:
            for em in endpoint_matching:
                for model_name, model_with_head in models.items():
                    model = model_with_head["model_with_softmax"]

                    # Ensure input data is on same device
                    x_test = x_test.to(device)
                    model = model.to(device)
                    y_test = model(x_test).argmax(-1)

                    _attr = list()
                    explainer = GeodesicIGSVI(model)
                    for i, (x, y, b) in get_progress_bars()(
                        enumerate(zip(x_test, y_test, baselines)),
                        total=len(x_test),
                        desc="GeodesicIGSVI attribution",
                    ):
                        _attr.append(
                            explainer.attribute(
                                x.unsqueeze(0),
                                baselines=b.unsqueeze(0),
                                target=y.item(),
                                num_iterations=num_iterations,
                                beta=beta,
                                n_steps=n_steps,
                                do_linear_interp=li,
                                use_endpoints_matching=em,
                                learning_rate=learning_rate,
                            ).squeeze(0)
                        )
                    attr[
                        f"svi_integrated_gradients_{model_name}_{em}_{li}_{num_iterations}_{n_steps}_{beta}_{learning_rate}_{learning_rate_decay}"
                    ] = th.stack(_attr)

    if "input_x_gradient" in explainers:
        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            x_test = x_test.to(device)
            model = model.to(device)
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = InputXGradient(model)
            _attr = list()
            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc="InputX attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        target=y.item(),
                    ).squeeze(0)
                )
            attr[f"input_x_gradient_{model_name}"] = th.stack(_attr)

    if "augmented_occlusion" in explainers:
        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            # Target is the model prediction
            x_test = x_test.to(device)
            model = model.to(device)
            y_test = model(x_test).argmax(-1).to(device)

            explainer = AugmentedOcclusion(model, data=x_test)
            _attr = list()
            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc="Augmented Occlusion attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        sliding_window_shapes=(3, 15, 15),
                        strides=(3, 8, 8),
                        target=y.item(),
                        attributions_fn=abs,
                    ).squeeze(0)
                )
            attr[f"augmented_occlusion_{model_name}"] = th.stack(_attr)

    if "occlusion" in explainers:
        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            # Target is the model prediction
            x_test = x_test.to(device)
            model = model.to(device)
            y_test = model(x_test).argmax(-1).to(device)

            explainer = Occlusion(model)
            _attr = list()
            for i, (x, y, b) in get_progress_bars()(
                enumerate(zip(x_test, y_test, baselines)),
                total=len(x_test),
                desc="Occlusion attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        baselines=b.unsqueeze(0),
                        sliding_window_shapes=(3, 15, 15),
                        strides=(3, 8, 8),
                        target=y.item(),
                        attributions_fn=abs,
                    ).squeeze(0)
                )
            attr[f"occlusion_{model_name}"] = th.stack(_attr)

    if "smooth_grad" in explainers:
        for model_name, model_with_head in models.items():
            model = model_with_head["model_with_softmax"]
            # Target is the model prediction
            x_test = x_test.to(device)
            model = model.to(device)
            y_test = model(x_test).argmax(-1).to(device)

            explainer = NoiseTunnel(IntegratedGradients(model))
            _attr = list()
            for i, (x, y, b) in get_progress_bars()(
                enumerate(zip(x_test, y_test, baselines)),
                total=len(x_test),
                desc="smoothGrad attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        baselines=b.unsqueeze(0),
                        target=y.item(),
                        internal_batch_size=200,
                        nt_samples=10,
                        stdevs=1.0,
                        nt_type="smoothgrad_sq",
                    ).squeeze(0)
                )
            attr["smooth_grad"] = th.stack(_attr)

    if "random" in explainers:
        for model_name, model in models.items():

            _attr = list()
            for i, x in get_progress_bars()(
                enumerate(x_test),
                total=len(x_test),
                desc="Random attribution",
            ):
                _attr.append(th.rand_like(x))
            attr[f"random_{model_name}"] = th.stack(_attr)

    # Save attributions
    attr_path = os.path.join(os.path.join(file_dir, "plots", now, "attributions"))
    if not os.path.exists(attr_path):
        os.makedirs(attr_path)

    attr_cpu = {k: v.cpu() if th.is_tensor(v) else v for k, v in attr.items()}

    # Save CPU tensors
    th.save(attr_cpu, os.path.join(attr_path, "attributions.pt"))

    lock = mp.Lock()

    results_path = os.path.join(os.path.join(file_dir, "results", now, "results.csv"))
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path))

    with open(results_path, "a") as fp, lock:

        for mode in get_progress_bars()(
            ["zeros", "aug"], total=2, desc="Mode", leave=False
        ):
            for topk in get_progress_bars()(areas, desc="Topk", leave=False):
                for k, v in get_progress_bars()(attr.items(), desc="Attr", leave=False):
                    for model_name, model_with_head in models.items():
                        model = model_with_head["model_without_softmax"]
                        device = th.device("cuda" if th.cuda.is_available() else "cpu")
                        model = model.to(device)
                        x_test = x_test.to(device)
                        if v is not None:
                            v = v.to(device)

                        acc_comp = accuracy(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                            mask_largest=True,
                        )
                        acc_suff = accuracy(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                            mask_largest=False,
                        )
                        comp = comprehensiveness(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                        )
                        ce_comp = cross_entropy(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                            mask_largest=True,
                        )
                        ce_suff = cross_entropy(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                            mask_largest=False,
                        )
                        l_odds = log_odds(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                        )
                        suff = sufficiency(
                            model,
                            x_test,
                            attributions=v.abs(),
                            baselines=x_test if mode == "aug" else None,
                            n_samples=10 if mode == "aug" else 1,
                            n_samples_batch_size=1 if mode == "aug" else None,
                            stdevs=0.1 if mode == "aug" else 0.0,
                            draw_baseline_from_distrib=True if mode == "aug" else False,
                            topk=topk,
                        )

                        fp.write(str(seed) + ",")
                        fp.write(mode + ",")
                        fp.write(str(topk) + ",")
                        fp.write(k + ",")
                        fp.write(model_name + ",")
                        fp.write(f"{acc_comp:.4},")
                        fp.write(f"{acc_suff:.4},")
                        fp.write(f"{comp:.4},")
                        fp.write(f"{ce_comp:.4},")
                        fp.write(f"{ce_suff:.4},")
                        fp.write(f"{l_odds:.4},")
                        fp.write(f"{suff:.4},")
                        fp.write("None,")
                        fp.write("None")
                        fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Number of steps for methods that require it, such as Integrated Gradients.",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "geodesic_integrated_gradients",
            "input_x_gradient",
            "kernel_shap",
            "svi_integrated_gradients",
            "guided_integrated_gradients",
            "integrated_gradients",
            "gradient_shap",
            "augmented_occlusion",
            "occlusion",
            "random",
            "smooth_grad",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.01,
            0.02,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
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
        areas=args.areas,
        n_steps=args.n_steps,
        seed=args.seed,
        deterministic=args.deterministic,
    )
