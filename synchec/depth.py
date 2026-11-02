from __future__ import annotations

import torch


def fuse_metric_depth(raw: torch.Tensor, predicted: torch.Tensor):
    if raw.shape != predicted.shape or raw.ndim not in (2, 3):
        raise ValueError("Depth arrays must share shape (H, W) or (N, H, W)")
    if not raw.is_floating_point() or not predicted.is_floating_point():
        raise ValueError("Depth arrays must use floating-point meters")
    single = raw.ndim == 2
    if single:
        raw, predicted = raw.unsqueeze(0), predicted.unsqueeze(0)
    predictions = predicted.to(raw)
    valid_raw = torch.isfinite(raw) & (raw > 0)
    valid_pred = torch.isfinite(predictions) & (predictions > 0)
    completed, validity, scales = [], [], []
    for sensor, prediction, sensor_valid, prediction_valid in zip(
        raw, predictions, valid_raw, valid_pred
    ):
        overlap = sensor_valid & prediction_valid
        if not overlap.any():
            raise ValueError(
                "Each frame needs valid sensor/prediction overlap for metric alignment"
            )
        scale = torch.quantile(sensor[overlap] / prediction[overlap], 0.5)
        filled = torch.where(prediction_valid, prediction * scale, torch.zeros_like(prediction))
        completed.append(torch.where(sensor_valid, sensor, filled))
        validity.append(sensor_valid | prediction_valid)
        scales.append(scale)
    result = torch.stack(completed), torch.stack(validity), torch.stack(scales)
    return tuple(x[0] for x in result) if single else result
