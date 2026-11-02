from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import torch

from synchec.config import Config
from synchec.data import CalibDataset, select_high_flow_frames
from synchec.kinematics import q_sync
from synchec.optimize import CalibrationProblem, calibrate, calibrate_joint
from synchec.render import Rasterizer
from synchec.stage1 import refine_anchor, select_anchor


def main():
    parser = argparse.ArgumentParser(description="SyncHeC recording calibration")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/calibration"))
    parser.add_argument("--init-extrinsic", type=Path)
    parser.add_argument("--init-delta-t-ms", type=float, default=0.0)
    parser.add_argument("--base-mask", type=Path)
    parser.add_argument("--static-frame", type=int, default=0)
    parser.add_argument("--mount", choices=("eth", "eih"))
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=("soft", "nvdiffrast"))
    parser.add_argument("--joint-descent", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = Config.from_json(args.config) if args.config else Config()
    if args.device:
        cfg.device = args.device
    if args.backend:
        cfg.render.backend = args.backend
    cfg.validate()
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.seed)
    ds = CalibDataset(args.recording, dtype=getattr(torch, cfg.dtype), device=cfg.device)
    if args.mount:
        ds.mount = args.mount
    cfg.data.width, cfg.data.height = ds.camera.width, ds.camera.height
    cfg.data.mount, cfg.data.depth_scale = ds.mount, ds.depth_scale
    if ds.fps is not None:
        cfg.data.camera_fps = ds.fps
    if ds.joint_rate_hz is not None:
        cfg.data.joint_rate_hz = ds.joint_rate_hz
    robot = ds.robot
    rasterizer = Rasterizer.from_config(ds.camera, cfg.render)
    if args.init_extrinsic:
        payload = json.loads(args.init_extrinsic.read_text())
        T_init = torch.as_tensor(payload["extrinsic"], dtype=ds.dtype, device=ds.device)
    else:
        if args.base_mask is None:
            parser.error("Stage 1 requires --base-mask; alternatively provide --init-extrinsic")
        if not 0 <= args.static_frame < ds.n_frames:
            parser.error("--static-frame is outside the recording")
        base = cv2.imread(str(args.base_mask), cv2.IMREAD_GRAYSCALE)
        if base is None:
            raise FileNotFoundError(args.base_mask)
        base = torch.as_tensor(base > 127, device=ds.device)
        i = args.static_frame
        q_static = q_sync(ds.joint_log, ds.cam_stamps[i : i + 1], ds.masks.new_tensor(0.0))[0]
        anchor = select_anchor(
            robot,
            ds.camera,
            rasterizer,
            q_static,
            ds.masks[i],
            ds.depths[i],
            cfg,
            base_mask=base,
            depth_valid=ds.depth_valid[i],
            mount=ds.mount,
            verbose=not args.quiet,
        )
        T_init = refine_anchor(
            robot,
            ds.camera,
            rasterizer,
            q_static,
            ds.masks[i],
            anchor["T"],
            cfg,
            mount=ds.mount,
            verbose=not args.quiet,
        )
    keep = select_high_flow_frames(ds, robot, cfg.data.n_keep)
    sub = ds.subset(keep)
    problem = CalibrationProblem(
        robot,
        sub.camera,
        rasterizer,
        sub.joint_log,
        sub.cam_stamps,
        sub.masks,
        sub.depths,
        sub.depth_valid,
        cfg,
        mount=sub.mount,
    )
    runner = calibrate_joint if args.joint_descent else calibrate
    result = runner(problem, T_init, args.init_delta_t_ms * 1e-3, cfg, verbose=not args.quiet)
    payload = {
        "method": "SyncHeC",
        "extrinsic": result.T.cpu().tolist(),
        "transform": "T_cb" if sub.mount == "eth" else "T_ce",
        "delta_t_s": result.delta_t_s,
        "delta_t_ms": result.delta_t_ms,
        "frame_indices": keep.cpu().tolist(),
        "renderer": rasterizer.backend,
        "optimizer": "joint" if args.joint_descent else "alternating",
        "config": cfg.to_dict(),
        "history": asdict(result.history),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "calibration.json"
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(result.history.summary())
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
