import multiprocessing as mp
import os
import torch as th
import torchvision.transforms as T
import warnings

from captum.attr import (
    IntegratedGradients,
    KernelShap,
    GradientShap
)
import saliency.core as saliency

from argparse import ArgumentParser
from pytorch_lightning import  seed_everything
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection , VOCSegmentation
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

from geodesic.svi_ig import SVI_IG
from geodesic.geodesic_ig import GeodesicIntegratedGradients

from geodesic.utils.tqdm import get_progress_bars

from experiments.voc.classifier import VocClassifier
from experiments.voc.constants import VALID_BACKBONE_NAMES
from experiments.voc.train_classifier import setup_model

from geodesic.metrics import accuracy, comprehensiveness, cross_entropy, log_odds, sufficiency


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
        tensor = tensor.detach().numpy().transpose(1, 2, 0)
        tensor = std * tensor + mean
        tensor = np.clip(tensor, 0, 1)
    else:
        device = tensor.device
        tensor = tensor.detach().numpy().transpose(1, 2, 0) if device == "cpu" else tensor.cpu().detach().numpy().transpose(1, 2, 0)
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
    areas: List[float],
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
    # TODO: Transformations in the final experiment should 224x224
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

    target_transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(centre_crop),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 255).long()),
        ]
    )
  

    # Load test data
    voc = VOCDetection(
        root=os.path.join(
            os.path.split(os.path.split(file_dir)[0])[0],
            "tint",
            "data",
            "voc",
        ),
        image_set="val",
        transform=transform,
        download=True,
    )

    # voc = VOCSegmentation(
    #     root=os.path.join(
    #         os.path.split(os.path.split(file_dir)[0])[0],
    #         "tint",
    #         "data",
    #         "voc",
    #     ),
    #     image_set="val",
    #     transform=transform,
    #     target_transform=target_transform,
    #     download=True,
    # )

    voc_loader = DataLoader(
        voc, 
        batch_size=1, 
        shuffle=True,
        generator=th.Generator().manual_seed(2) 
    )

    def load_model(checkpoint_path):
        # Initialize model
        model = setup_model()
        
        # Load checkpoint
        checkpoint = th.load(checkpoint_path, map_location=th.device('cpu')) if accelerator == "cpu" else th.load(checkpoint_path)
        
        # Extract model state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Move to GPU if available
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        model = model.to(device)
        return model
    
    models = dict()
    # Load models
    for model_name in VALID_BACKBONE_NAMES:
        # TODO: tidy this up
        models[model_name] = load_model('/Users/sinasalek/VSCodeProjects/geodesic-ig/experiments/voc/attributions202501100055235/best_model.pt')
    

        # Switch to eval
        models[model_name].eval()

        # Set model to device
        models[model_name].to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # VOC_CLASSES = [
    #     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    #     'bus', 'car', 'cat', 'chair', 'cow',
    #     'diningtable', 'dog', 'horse', 'motorbike', 'person',
    #     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]

    # Get data as tensors
    x_test = list()
    y_test = list()
    i = 0
    for data, target in voc_loader:
        if i == 1:
            break
        
        # # Extract first object class as the target
        # class_name = list(target['annotation']['object'][0]['name'])
        # class_idx = VOC_CLASSES.index(class_name[0])  
        
        x_test.append(data)
        # y_test.append(class_idx)
        i += 1
        
    print(f"Number of test samples: {len(x_test)}")

    x_test = th.cat(x_test).to(device)
    # y_test = th.tensor(y_test).to(device)

    # x_test = list()
    # seg_test = list()
    # i = 0
    # for data, seg in voc_loader:
    #     if i == 1:
    #         break

    #     seg_ids = seg.unique()
    #     if len(seg_ids) <= 1:
    #         continue

    #     seg_ = seg.clone()
    #     for j, seg_id in enumerate(seg_ids):
    #         seg_[seg_ == seg_id] = j

    #     x_test.append(data)
    #     seg_test.append(seg_)
    #     i += 1

    # x_test = th.cat(x_test).to(device)
    # seg_test = th.cat(seg_test).to(device)


    # Create dict of attributions, explainers
    attr = dict()
    expl = dict()

    # Save the first 10 images
    for i in range(min(10, len(x_test))):
        image_extended_path = os.path.join("original_images", f"original_image_{i}.png")
        plot_and_save(x_test[i], image_extended_path)
    now = np.datetime64("now").astype(str)

    if "geodesic_integrated_gradients" in explainers:
        _attr = list()

        for model_name, model in models.items():
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc=f"{GeodesicIntegratedGradients.get_name()} attribution",
            ):
                x_aug = generate_augmented_points(x, n_base_points=50, n_noise_points=0, device=device) 
                explainer = GeodesicIntegratedGradients(
                    model, 
                    data=x_aug, 
                    n_neighbors=5
                )

                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        target=y.item(),
                        internal_batch_size=100,
                    ).squeeze(0)
                )

            attr[f"geodesic_integrated_gradients_{model_name}"] = th.stack(_attr)
            expl[f"geodesic_integrated_gradients_{model_name}"] = explainer
            for i in range(min(10, len(x_test))):
                image_extended_path = os.path.join(
                    now, "attribution_geodesic_integrated_gradients", f"attribution_geodesic_integrated_gradients_{model_name}_{i}.png"
                )
                plot_and_save(attr[f"geodesic_integrated_gradients_{model_name}"][i], image_extended_path, is_attribution=True)
    if "kernel_shap" in explainers:
        for model_name, model in models.items():
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = KernelShap(model)
            _attr = list()
            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc=f"{KernelShap.get_name()} attribution",
            ):
                _attr.append(
                    explainer.attribute(
                    x.unsqueeze(0),
                    target=y.item(),
                    ).squeeze(0)
                )
            attr[f"kernel_shap_{model_name}"] = th.stack(_attr)
            expl[f"kernel_shap_{model_name}"] = explainer
            for i in range(min(10, len(x_test))):
                image_extended_path = os.path.join(
                    now, "attribution_kernel_shap", f"attribution_kernel_shap_{model_name}_{i}.png"
                )
                plot_and_save(attr[f"kernel_shap_{model_name}"][i], image_extended_path, is_attribution=True)


    if "gradient_shap" in explainers:
        for model_name, model in models.items():
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = GradientShap(model)
            _attr = list()
            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc=f"{GradientShap.get_name()} attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        baselines=th.zeros_like(x_test),
                        target=y.item(),
                    ).squeeze(0)
                )
            attr[f"gradient_shap_{model_name}"] = th.stack(_attr)
            expl[f"gradient_shap_{model_name}"] = explainer
            for i in range(min(10, len(x_test))):
                image_extended_path = os.path.join(
                    now, "attribution_gradient_shap", f"attribution_gradient_shap_{model_name}_{i}.png"
                )
                plot_and_save(attr[f"gradient_shap_{model_name}"][i], image_extended_path, is_attribution=True)

    if "integrated_gradients" in explainers:
        
        n_steps = 50
        for model_name, model in models.items():
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            explainer = IntegratedGradients(model)
            _attr = list()
            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc=f"{IntegratedGradients.get_name()} attribution",
            ):
                _attr.append(
                    explainer.attribute(
                        x.unsqueeze(0),
                        target=y.item(),
                        n_steps=n_steps,
                    ).squeeze(0)
                )
            attr[f"integrated_gradients_{model_name}"] = th.stack(_attr)
            expl[f"integrated_gradients_{model_name}"] = explainer

            for i in range(min(10, len(x_test))):
                image_extended_path = os.path.join(
                    now, "attribution_integrated_gradients", f"attribution_integrated_gradients_{model_name}_{i}.png"
                )
                plot_and_save(attr[f"integrated_gradients_{model_name}"][i], image_extended_path, is_attribution=True)

    if "guided_integrated_gradients" in explainers:
        n_steps = 50
        guided_ig = saliency.GuidedIG()



        # transformer = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transformer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        def PreprocessImages(images):
            # assumes input is 4-D, with range [0,255]
            #
            # torchvision have color channel as first dimension
            # with normalization relative to mean/std of ImageNet:
            #    https://pytorch.org/vision/stable/models.html
            images = np.array(images)
            images = images/255
            images = np.transpose(images, (0,3,1,2))
            images = th.tensor(images, dtype=th.float32)
            images = transformer.forward(images)
            return images.requires_grad_(True)
        
        for model_name, model in models.items():

            # conv_layer = model.Mixed_7c
            # conv_layer_outputs = {}
            # def conv_layer_forward(m, i, o):
            #     # move the RGB dimension to the last dimension
            #     conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = th.movedim(o, 1, 3).detach().numpy()
            # def conv_layer_backward(m, i, o):
            #     # move the RGB dimension to the last dimension
            #     conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = th.movedim(o[0], 1, 3).detach().numpy()
            # conv_layer.register_forward_hook(conv_layer_forward)
            # conv_layer.register_full_backward_hook(conv_layer_backward)
            def call_model_function(images, call_model_args=None, expected_keys=None):
                images = PreprocessImages(images)
                target_class_idx =  call_model_args[class_idx_str]
                output = model(images)
                m = th.nn.Softmax(dim=1)
                output = m(output)
                if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
                    outputs = output[:,target_class_idx]
                    grads = th.autograd.grad(outputs, images, grad_outputs=th.ones_like(outputs))
                    grads = th.movedim(grads[0], 1, 3)
                    gradients = grads.detach().numpy()
                    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
                else:
                    one_hot = th.zeros_like(output)
                    one_hot[:,target_class_idx] = 1
                    model.zero_grad()
                    output.backward(gradient=one_hot, retain_graph=True)
                    return conv_layer_outputs
            # Target is the model prediction
            y_test = model(x_test).argmax(-1).to(device)

            _attr = list()
            for i, (x, y) in get_progress_bars()(
                enumerate(zip(x_test, y_test)),
                total=len(x_test),
                desc=f"guided IG attribution",
            ):

                predictions = model(x.unsqueeze(0))
                predictions = predictions.detach().numpy()
                prediction_class = np.argmax(predictions[0])
                class_idx_str = 'class_idx_str'
                call_model_args = {class_idx_str: prediction_class}
                x = x.permute(1, 2, 0)
                baseline = np.zeros(x.shape)
                _attr.append(
                    th.from_numpy(
                        guided_ig.GetMask(
                        x, call_model_function, call_model_args, x_steps=n_steps, x_baseline=baseline, max_dist=1.0, fraction=0.5
                        )
                    ).permute(2, 0, 1)
                )
            attr[f"guided_integrated_gradients_{model_name}"] = th.stack(_attr)
            expl[f"guided_integrated_gradients_{model_name}"] = guided_ig
            for i in range(min(10, len(x_test))):
                image_extended_path = os.path.join(
                    now, "attribution_guided_integrated_gradients", f"attribution_guided_integrated_gradients_{model_name}_{i}.png"
                )
                plot_and_save(attr[f"guided_integrated_gradients_{model_name}"][i], image_extended_path, is_attribution=True)
    if "svi_integrated_gradients" in explainers:
        num_iterations = 1000
        beta = 0.3
        linear_interpolation = [False, True]
        endpoint_matching = [True, False]
        learning_rate_decay = True
        n_steps = 50
        learning_rate = 0.01
        for li in linear_interpolation:
            for em in endpoint_matching:
                for model_name, model in models.items():
                    # Ensure input data is on same device
                    x_test = x_test.to(device)  
                    model = model.to(device)
                    y_test = model(x_test).argmax(-1)  # Result already on GPU
                    
                    _attr = list()
                    explainer = SVI_IG(model)
                    for i, (x, y) in get_progress_bars()(
                        enumerate(zip(x_test, y_test)),
                        total=len(x_test),
                        desc=f"{SVI_IG.get_name()} attribution",
                    ):
                        _attr.append(
                            explainer.attribute(
                                x.unsqueeze(0),
                                target=y.item(),
                                num_iterations=num_iterations,
                                beta=beta,
                                n_steps=n_steps,
                                do_linear_interp=li,
                                use_endpoints_matching=em,
                                learning_rate = learning_rate
                            ).squeeze(0)
                        )
                    attr[f"svi_integrated_gradients_{model_name}_{em}_{li}_{num_iterations}_{n_steps}_{beta}_{learning_rate}_{learning_rate_decay}"] = th.stack(_attr)
                    expl[f"svi_integrated_gradients_{model_name}_{em}_{li}_{num_iterations}_{n_steps}_{beta}_{learning_rate}_{learning_rate_decay}"] = explainer
                    for i in range(min(10, len(x_test))):
                        image_extended_path = os.path.join(
                            now, "attribution_svi_integrated_gradients", f"attribution_svi_integrated_gradients_{model_name}_{em}_{li}_{num_iterations}_{n_steps}_{beta}_{learning_rate}_{learning_rate_decay}_{i}.png"
                        )
                        plot_and_save(attr[f"svi_integrated_gradients_{model_name}_{em}_{li}_{num_iterations}_{n_steps}_{beta}_{learning_rate}_{learning_rate_decay}"][i], image_extended_path, is_attribution=True)


    # Save attributions
    attr_path = os.path.join(
        os.path.join(file_dir, "plots", now, "attributions")
    )
    if not os.path.exists(attr_path):
        os.makedirs(attr_path)

    th.save(attr, os.path.join(attr_path, "attributions.pt"))
    th.save(expl, os.path.join(attr_path, "explainers.pt"))

    
    lock = mp.Lock()
    
    results_path = os.path.join(
        os.path.join(file_dir, "results",now, "results.csv")
    )
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path))

    with open(results_path, "a") as fp, lock:
  
        for mode in get_progress_bars()(
            ["zeros", "aug"], total=2, desc="Mode", leave=False
        ):
            for topk in get_progress_bars()(areas, desc="Topk", leave=False):
                for k, v in get_progress_bars()(
                    attr.items(), desc="Attr", leave=False
                ):
                    for model_name, model in models.items():
                        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
                            draw_baseline_from_distrib=True
                            if mode == "aug"
                            else False,
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
        "--explainers",
        type=str,
        default=[
            # "geodesic_integrated_gradients",
            # "svi_integrated_gradients",
            "guided_integrated_gradients",
            # "integrated_gradients",
            # "kernel_shap",
            # "gradient_shap",
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
            0.2,
            0.5,
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
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="resnet18",
        help="Name of the classifier to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        device=args.device,
        areas=args.areas,
        seed=args.seed,
        deterministic=args.deterministic,
    )