import pytest
import torch as th

from captum.attr import Saliency

from contextlib import nullcontext

from geodesic.metrics import log_odds

from tests.basic_models import BasicModel, BasicModel5_MultiArgs


@pytest.mark.parametrize(
    [
        "forward_func",
        "inputs",
        "baselines",
        "additional_forward_args",
        "target",
        "n_samples",
        "n_samples_batch_size",
        "stdevs",
        "draw_baseline_from_distrib",
        "topk",
        "weight_fn",
        "fails",
    ],
    [
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 1, None, 0.0, False, 0.2, None, False),
        (
            BasicModel5_MultiArgs(),
            th.rand(8, 5, 3),
            None,
            (th.rand(8, 5, 3), th.rand(8, 5, 3)),
            None,
            1,
            None,
            0.0,
            False,
            0.2,
            None,
            False,
        ),
        (BasicModel(), th.rand(8, 5, 3), 0, None, None, 1, None, 0.0, False, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), th.rand(8, 5, 3), None, None, 1, None, 0.0, False, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, 0, 1, None, 0.0, False, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 5, None, 0.0, False, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 5, 3, 0.0, False, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 5, None, 0.1, False, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), th.rand(16, 5, 3), None, None, 5, None, 0.0, True, 0.2, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 1, None, 0.0, False, 0.6, None, False),
        (BasicModel(), th.rand(8, 5, 3), None, None, None, 1, None, 0.0, False, 1.2, None, True),
    ],
)
def test_log_odds(
    forward_func,
    inputs,
    baselines,
    additional_forward_args,
    target,
    n_samples,
    n_samples_batch_size,
    stdevs,
    draw_baseline_from_distrib,
    topk,
    weight_fn,
    fails,
):
    with pytest.raises(Exception) if fails else nullcontext():
        explainer = Saliency(forward_func=forward_func)
        attr = explainer.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )

        log_odds_ = log_odds(
            forward_func=forward_func,
            inputs=inputs,
            attributions=attr,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
            n_samples=n_samples,
            n_samples_batch_size=n_samples_batch_size,
            stdevs=stdevs,
            draw_baseline_from_distrib=draw_baseline_from_distrib,
            topk=topk,
            weight_fn=weight_fn,
        )

        assert isinstance(log_odds_, float)
