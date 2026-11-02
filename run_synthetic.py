from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from synchec import lie
from synchec.config import Config
from synchec.data import CalibDataset
from synchec.optimize import CalibrationProblem, calibrate, calibrate_joint, evaluate
from synchec.render import Rasterizer
from synchec.synthetic import load_ground_truth, make_synthetic_recording


def main():
    parser = argparse.ArgumentParser(description="SyncHeC synthetic pose and time-offset recovery")
    parser.add_argument("--out", type=Path, default=Path("runs/synthetic"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/synthetic.json"
    )
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--offset-ms", type=float, default=87.0)
    parser.add_argument("--init-delta-t-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constant-velocity", action="store_true")
    parser.add_argument("--joint-descent", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = Config.from_json(args.config)
    if cfg.device != "cpu" or cfg.render.backend != "soft":
        parser.error("The synthetic example uses the CPU soft renderer")
    cfg.seed = args.seed
    cfg.data.width, cfg.data.height, cfg.data.n_keep = args.width, args.height, args.frames
    if args.no_depth:
        cfg.loss.lambda_depth = 0.0
    cfg.validate()
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    recording = args.out / "recording"
    make_synthetic_recording(
        recording,
        delta_t_gt=args.offset_ms * 1e-3,
        seed=cfg.seed,
        n_frames=args.frames,
        width=args.width,
        height=args.height,
        constant_velocity=args.constant_velocity,
        verbose=not args.quiet,
    )
    ds = CalibDataset(recording)
    rasterizer = Rasterizer.from_config(ds.camera, cfg.render)
    problem = CalibrationProblem(
        ds.robot,
        ds.camera,
        rasterizer,
        ds.joint_log,
        ds.cam_stamps,
        ds.masks,
        ds.depths,
        ds.depth_valid,
        cfg,
        mount=ds.mount,
    )
    gt = load_ground_truth(recording)
    T_gt = torch.as_tensor(gt["extrinsic"], dtype=ds.dtype)
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    perturbation = lie.se3_exp(
        torch.randn(3, generator=generator) * 0.01,
        torch.randn(3, generator=generator) * 0.02,
    )
    T_init = T_gt @ perturbation
    dt_init = args.init_delta_t_ms * 1e-3
    runner = calibrate_joint if args.joint_descent else calibrate
    result = runner(problem, T_init, dt_init, cfg, verbose=not args.quiet)
    metrics = evaluate(result.T, T_gt, result.delta_t_s, gt["delta_t_s"])
    payload = {
        "method": "SyncHeC",
        "benchmark": "synthetic",
        "initialization": "perturbed reference pose",
        "initial": evaluate(T_init, T_gt, dt_init, gt["delta_t_s"]),
        "final": metrics,
        "extrinsic": result.T.tolist(),
        "delta_t_ms": result.delta_t_ms,
        "renderer": rasterizer.backend,
        "config": cfg.to_dict(),
        "history": asdict(result.history),
    }
    path = args.out / "report.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(result.history.summary())
    print(json.dumps(metrics))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
