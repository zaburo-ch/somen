from typing import Callable, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from pytorch_pfn_extras.runtime import runtime_registry
from torch import Tensor
from tqdm import tqdm


def predict(
    model: Union[torch.nn.Module, Callable],
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 0,
    device: str = "cuda",
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    progress: bool = True,
) -> Union[Mapping[str, np.ndarray], Sequence[np.ndarray]]:

    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if progress:
        data_loader = tqdm(data_loader)

    runtime = runtime_registry.get_runtime_class_for_device_spec(device)(device, {})

    with torch.no_grad():

        def _get_non_ref_array(x):
            return x.detach().cpu().numpy().copy()

        y_preds: Union[Dict[str, List[np.ndarray]], List[List[np.ndarray]], None] = None
        for batch in data_loader:
            batch = runtime.convert_batch(batch)

            if isinstance(batch, dict):
                y_pred = model(**batch)
            elif isinstance(batch, (list, tuple)):
                y_pred = model(*batch)
            else:
                y_pred = model(batch)

            if isinstance(y_pred, dict):
                y_pred = {key: _get_non_ref_array(value) for key, value in y_pred.items()}
            if isinstance(y_pred, (list, tuple)):
                y_pred = [_get_non_ref_array(e) for e in y_pred]
            else:
                y_pred = [_get_non_ref_array(y_pred)]

            if isinstance(y_pred, dict):
                if y_preds is None:
                    y_preds = {key: [] for key in y_pred}

                assert isinstance(y_preds, dict)
                for key, value in y_pred.items():
                    y_preds[key].append(value)
            else:
                assert isinstance(y_pred, list)
                if y_preds is None:
                    y_preds = [[] for _ in range(len(y_pred))]

                assert isinstance(y_preds, list)
                for k in range(len(y_pred)):
                    y_preds[k].append(y_pred[k])

        assert y_preds is not None
        if isinstance(y_preds, dict):
            return {key: np.concatenate(value, axis=0) for key, value in y_preds.items()}
        else:
            return [np.concatenate(value, axis=0) for value in y_preds]


def slide_average(
    f: Callable[[Tensor], Tensor],
    inputs: Tensor,
    crop_size: int,
    max_stride: int,
    weight_type: Literal["uniform", "distance_to_border"] = "uniform",
    up_scale: int = 1,
    down_scale: int = 1,
) -> Tensor:
    assert up_scale == 1 or down_scale == 1

    B, _, H, W = inputs.shape

    nh = (H - crop_size + max_stride) // max_stride
    nw = (W - crop_size + max_stride) // max_stride

    if down_scale > 1:
        assert H % down_scale == 0 and W % down_scale == 0
        assert crop_size % down_scale == 0
        # Get indices divisible by down_scale
        ys = np.linspace(0, (H - crop_size) // down_scale, nh).round().astype(int) * down_scale
        xs = np.linspace(0, (W - crop_size) // down_scale, nw).round().astype(int) * down_scale
    else:
        ys = np.linspace(0, H - crop_size, nh).astype(int)
        xs = np.linspace(0, W - crop_size, nw).astype(int)

    starts = [(y, x) for y in ys for x in xs]

    outputs = [f(inputs[..., y : y + crop_size, x : x + crop_size]) for y, x in starts]
    output_channels = outputs[0].shape[1]

    if down_scale > 1:
        crop_size = crop_size // down_scale
        starts = [(y // down_scale, x // down_scale) for y, x in starts]
        H, W = H // down_scale, W // down_scale
        assert all([o.shape == (B, output_channels, crop_size, crop_size) for o in outputs])
    else:
        crop_size = crop_size * up_scale
        starts = [(y * up_scale, x * up_scale) for y, x in starts]
        H, W = H * up_scale, W * up_scale
        assert all([o.shape == (B, output_channels, crop_size, crop_size) for o in outputs])

    if weight_type == "uniform":
        weights = torch.ones((crop_size, crop_size)).to(inputs)
    elif weight_type == "distance_to_border":
        weights = (1 - torch.linspace(-1, 1, crop_size + 2)[1:-1].abs()) / 2
        weights = torch.minimum(weights[:, np.newaxis], weights[np.newaxis, :])
        weights = weights.to(inputs)
    else:
        raise RuntimeError()  # TODO: message

    merged = torch.zeros((B, output_channels, H, W)).to(inputs)
    normalize = torch.zeros((H, W)).to(inputs)

    for (y, x), o in zip(starts, outputs):
        merged[:, :, y : y + crop_size, x : x + crop_size] += weights * o
        normalize[y : y + crop_size, x : x + crop_size] += weights

    return merged / normalize
