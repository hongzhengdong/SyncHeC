from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from synchec.config import Config
from synchec.data import CalibDataset
from synchec.planner import BSplineTrajectory, LinkProjector, candidate_cameras, plan


def main():
    parser = argparse.ArgumentParser(description="SyncHeC landmark-based trajectory refinement")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initial-trajectory", type=Path)
    parser.add_argument(
        "--constraints", help="Importable module:function returning constraint callbacks"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/trajectory"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = Config.from_json(args.config) if args.config else Config()
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.seed)
    ds = CalibDataset(args.recording, eager=False, dtype=getattr(torch, cfg.dtype))
    robot = ds.robot
    T = torch.as_tensor(json.loads(args.extrinsic.read_text())["extrinsic"], dtype=ds.dtype)
    projector = LinkProjector.from_robot(robot, ds.camera, mount=ds.mount, seed=cfg.seed)
    candidates = candidate_cameras(T, cfg.planner.n_candidate_cameras, seed=cfg.seed)
    initial = None
    if args.initial_trajectory:
        payload = json.loads(args.initial_trajectory.read_text())
        initial = BSplineTrajectory(
            torch.as_tensor(payload["control_points"], dtype=ds.dtype),
            duration=float(payload["duration_s"]),
            degree=int(payload.get("degree", 3)),
        )
    constraints = {}
    if args.constraints:
        module, function = args.constraints.split(":", 1)
        constraints = getattr(importlib.import_module(module), function)(robot, ds.camera, T)
        if not isinstance(constraints, dict) or set(constraints) - {"collision_free", "visibility"}:
            raise ValueError(
                "Constraint factory must return collision_free and/or visibility callbacks"
            )
    trajectory, report = plan(
        projector,
        candidates,
        alpha=cfg.planner.alpha,
        beta=cfg.planner.beta,
        init=initial,
        n_knots=cfg.planner.n_knots,
        dof=robot.dof,
        duration=cfg.planner.duration_s,
        limits=robot.joint_limits,
        n_steps=cfg.planner.opt_steps,
        lr=cfg.planner.lr,
        seed=cfg.seed,
        verbose=not args.quiet,
        **constraints,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stamps, positions = trajectory.resample(cfg.data.joint_rate_hz)
    path = args.out / "trajectory.csv"
    array = np.column_stack((stamps.cpu().numpy(), positions.cpu().numpy()))
    np.savetxt(
        path,
        array,
        delimiter=",",
        header="t_seconds," + ",".join(f"q{i}" for i in range(robot.dof)),
        comments="",
    )
    payload = {
        "method": "SyncHeC",
        "objective": "projected_landmarks",
        "control_points": trajectory.control_points.cpu().tolist(),
        "duration_s": trajectory.duration,
        "degree": trajectory.degree,
        "joint_names": [j.name for j in robot.joints if j.q_index is not None],
        "config": cfg.to_dict(),
        "report": asdict(report),
    }
    (args.out / "trajectory.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(report.summary())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
