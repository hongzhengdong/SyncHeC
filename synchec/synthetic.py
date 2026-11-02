from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch
from . import data as data_mod
from . import lie
from .data import Intrinsics
from .kinematics import LinkPoseModel, q_sync
from .planner import BSplineTrajectory

DELTA_T_GT_DEFAULT = 0.087
FPS_DEFAULT = 30.0
JOINT_RATE_DEFAULT = 250.0
DURATION_DEFAULT = 6.5
N_KNOTS_DEFAULT = 25


@dataclass
class SyntheticSpec:
    delta_t_gt: float = DELTA_T_GT_DEFAULT
    seed: int = 0
    n_links: int = 4
    mount: str = "eth"
    width: int = 848
    height: int = 480
    fps: float = FPS_DEFAULT
    joint_rate_hz: float = JOINT_RATE_DEFAULT
    duration_s: float = DURATION_DEFAULT
    n_knots: int = N_KNOTS_DEFAULT
    depth_sigma_m: float = 0.0
    invalid_depth_frac: float = 0.0
    mask_morph_px: int = 0
    constant_velocity: bool = False
    extrinsic: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_t_seconds": float(self.delta_t_gt),
            "extrinsic": self.extrinsic,
            "mount": self.mount,
            "seed": int(self.seed),
            "trajectory": {
                "constant_velocity": bool(self.constant_velocity),
                "duration_s": float(self.duration_s),
                "n_knots": int(self.n_knots),
            },
            "robot": {"n_links": int(self.n_links)},
            "sensor": {
                "width": int(self.width),
                "height": int(self.height),
                "fps": float(self.fps),
                "joint_rate_hz": float(self.joint_rate_hz),
            },
            "noise": {
                "depth_sigma_m": float(self.depth_sigma_m),
                "invalid_depth_frac": float(self.invalid_depth_frac),
                "mask_morph_px": int(self.mask_morph_px),
            },
        }


def exciting_trajectory(
    dof: int,
    n_knots: int = N_KNOTS_DEFAULT,
    duration: float = DURATION_DEFAULT,
    limits: tuple[torch.Tensor, torch.Tensor] | None = None,
    amplitude: float = 0.9,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> BSplineTrajectory:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    probe = BSplineTrajectory(
        torch.zeros(n_knots, max(dof, 1), dtype=torch.float64), duration=duration
    )
    tg = torch.stack([probe.knots[i + 1 : i + probe.degree + 1].mean() for i in range(n_knots)]).to(
        torch.float64
    )
    cp = torch.zeros(n_knots, dof, dtype=torch.float64)
    base = 2.0 * math.pi / float(duration)
    for j in range(dof):
        for m, mult in enumerate((1.0, math.sqrt(2.0) * 1.7, math.pi / 1.3)):
            freq = base * mult * (1.0 + 0.23 * j)
            phase = float(torch.rand((), generator=g)) * 2.0 * math.pi
            weight = amplitude / (m + 1.4)
            cp[:, j] += weight * torch.sin(freq * tg + phase)
    if limits is not None:
        lo = torch.as_tensor(limits[0], dtype=torch.float64).reshape(-1)
        hi = torch.as_tensor(limits[1], dtype=torch.float64).reshape(-1)
        mid = 0.5 * (lo + hi)
        half = 0.9 * 0.5 * (hi - lo)
        span = cp.abs().amax(dim=0).clamp_min(1e-09)
        cp = mid + cp / span * torch.minimum(half, span)
        cp = cp.clamp(lo, hi)
    return BSplineTrajectory(cp.to(dtype), duration=duration)


def constant_velocity_trajectory(
    dof: int,
    n_knots: int = N_KNOTS_DEFAULT,
    duration: float = DURATION_DEFAULT,
    speed: float = 0.25,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> BSplineTrajectory:
    probe = BSplineTrajectory(
        torch.zeros(n_knots, max(dof, 1), dtype=torch.float64), duration=duration
    )
    tg = torch.stack([probe.knots[i + 1 : i + probe.degree + 1].mean() for i in range(n_knots)]).to(
        torch.float64
    )
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.randn(dof, generator=g, dtype=torch.float64)
    direction = direction / direction.norm().clamp_min(1e-12)
    cp = tg.unsqueeze(1) * direction.unsqueeze(0) * float(speed)
    return BSplineTrajectory(cp.to(dtype), duration=duration)


def default_intrinsics(width: int = 848, height: int = 480, hfov_deg: float = 69.4) -> Intrinsics:
    fx = 0.5 * width / math.tan(0.5 * math.radians(hfov_deg))
    return Intrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def default_extrinsic(
    mount: str = "eth", distance: float = 1.5, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if mount == "eih":
        R = torch.eye(3, dtype=dtype)
        t = torch.tensor([0.0, 0.0, 0.12], dtype=dtype)
        return lie.compose(R, t)
    eye = torch.tensor([distance * 0.8, -distance * 0.55, distance * 0.55], dtype=dtype)
    target = torch.tensor([0.18, 0.0, 0.18], dtype=dtype)
    fwd = target - eye
    fwd = fwd / fwd.norm()
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    right = torch.linalg.cross(world_up, fwd)
    right = right / right.norm().clamp_min(1e-09)
    down = torch.linalg.cross(fwd, right)
    R_cw = torch.stack([right, down, fwd], dim=1)
    return lie.inverse(lie.compose(R_cw, eye))


def _render_frames(
    robot: Any,
    rasterizer: Any,
    T_extrinsic: torch.Tensor,
    q_true: torch.Tensor,
    mount: str,
    chunk: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    pose_model = LinkPoseModel(robot, mount=mount)
    masks: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, q_true.shape[0], chunk):
            q = q_true[s : s + chunk]
            link_T = pose_model(T_extrinsic, q)
            (verts, faces) = robot.posed_mesh(link_T)
            out = rasterizer(verts, faces)
            masks.append(out["mask"].detach().cpu().numpy())
            depths.append(out["depth"].detach().cpu().numpy())
    return (
        np.concatenate(masks, axis=0).astype(np.float32),
        np.concatenate(depths, axis=0).astype(np.float32),
    )


def _apply_realism(
    masks: np.ndarray,
    depths: np.ndarray,
    *,
    depth_sigma_m: float,
    invalid_depth_frac: float,
    mask_morph_px: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if mask_morph_px:
        import cv2

        k = int(abs(mask_morph_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        op = cv2.erode if mask_morph_px > 0 else cv2.dilate
        masks = np.stack([op(m, kernel) for m in masks], axis=0).astype(np.float32)
    if depth_sigma_m > 0:
        hit = depths > 0
        noise = rng.normal(0.0, float(depth_sigma_m), size=depths.shape)
        depths = np.where(hit, depths + noise, depths).astype(np.float32)
        depths = np.clip(depths, 0.0, None)
    if invalid_depth_frac > 0:
        if not 0.0 <= invalid_depth_frac < 1.0:
            raise ValueError(
                "invalid_depth_frac must lie in [0, 1), got %r" % (invalid_depth_frac,)
            )
        (n, h, w) = depths.shape
        (yy, xx) = np.mgrid[0:h, 0:w]
        out = depths.copy()
        for i in range(n):
            hit = depths[i] > 0
            n_hit = int(hit.sum())
            if n_hit == 0:
                continue
            target = invalid_depth_frac * n_hit
            (ys, xs) = np.nonzero(hit)
            rad = max(1, int(round(math.sqrt(max(target, 1.0) / (8.0 * math.pi)))))
            keep = np.ones((h, w), dtype=bool)
            for _ in range(64):
                if int((hit & ~keep).sum()) >= target:
                    break
                j = int(rng.integers(0, n_hit))
                keep &= (yy - ys[j]) ** 2 + (xx - xs[j]) ** 2 > rad**2
            out[i] = np.where(keep, depths[i], 0.0)
        depths = out.astype(np.float32)
    return (masks, depths)


def make_synthetic_recording(
    out_dir: str | Path,
    delta_t_gt: float = DELTA_T_GT_DEFAULT,
    seed: int = 0,
    *,
    n_links: int = 4,
    mount: str = "eth",
    width: int = 848,
    height: int = 480,
    fps: float = FPS_DEFAULT,
    joint_rate_hz: float = JOINT_RATE_DEFAULT,
    duration_s: float = DURATION_DEFAULT,
    n_knots: int = N_KNOTS_DEFAULT,
    n_frames: int | None = None,
    t0_camera: float = 0.0,
    t0_joints: float = 0.0,
    depth_sigma_m: float = 0.0,
    invalid_depth_frac: float = 0.0,
    mask_morph_px: int = 0,
    constant_velocity: bool = False,
    T_extrinsic: torch.Tensor | None = None,
    trajectory: BSplineTrajectory | None = None,
    robot: Any = None,
    rasterizer: Any = None,
    write_rgb: bool = False,
    overwrite: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    if delta_t_gt < 0:
        raise ValueError("delta_t_gt must be >= 0 seconds, got %r" % (delta_t_gt,))
    if mount not in ("eth", "eih"):
        raise ValueError("mount must be 'eth' or 'eih', got %r" % (mount,))
    out_dir = Path(out_dir)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    if robot is None:
        from .fk import Robot

        robot = Robot.from_primitives(n_links=n_links)
    dof = int(robot.dof)
    intr = default_intrinsics(width, height)
    if rasterizer is None:
        from .render import Rasterizer

        rasterizer = Rasterizer(_camera_from(intr))
    if T_extrinsic is None:
        T_extrinsic = default_extrinsic(mount)
    T_extrinsic = T_extrinsic.to(torch.float32)
    if trajectory is None:
        limits = getattr(robot, "joint_limits", None)
        trajectory = (
            constant_velocity_trajectory(dof, n_knots, duration_s, seed=seed)
            if constant_velocity
            else exciting_trajectory(dof, n_knots, duration_s, limits=limits, seed=seed)
        )
    (t_r_rel, q_log) = trajectory.resample(joint_rate_hz)
    t_r = t_r_rel.to(torch.float64) + float(t0_joints)
    n_max = int(round(fps * duration_s))
    stamps = torch.arange(n_max + 1, dtype=torch.float64) / float(fps) + float(t0_camera)
    tau_all = stamps - float(delta_t_gt)
    inside = (tau_all >= float(t_r[0])) & (tau_all <= float(t_r[-1]))
    stamps = stamps[inside]
    if n_frames is not None:
        stamps = stamps[: int(n_frames)]
    if stamps.numel() < 2:
        raise ValueError(
            "only %d frame(s) have t_c - delta_t_gt inside the joint log; increase duration_s or lower delta_t_gt (%.3f s) / t0_camera"
            % (int(stamps.numel()), delta_t_gt)
        )
    from .kinematics import JointLog

    log = JointLog(t_r, q_log.to(torch.float32))
    dt_gt = torch.tensor(float(delta_t_gt), dtype=torch.float64)
    q_true = q_sync(log, stamps, dt_gt)
    q_naive = q_sync(log, stamps, torch.zeros((), dtype=torch.float64))
    moved = float((q_true - q_naive).abs().max())
    if delta_t_gt > 0:
        assert moved > 1e-09, (
            "the latency offset had no effect on the rendered joints (max |dq| = %.3e): the recording would be perfectly synchronized and delta_t_gt unrecoverable"
            % (moved,)
        )
    if verbose:
        print(
            "  rendering %d frames; latency shifts joints by up to %.4f rad"
            % (stamps.numel(), moved)
        )
    (masks, depths) = _render_frames(robot, rasterizer, T_extrinsic, q_true, mount)
    depth_clean = depths.copy()
    (masks, depths) = _apply_realism(
        masks,
        depths,
        depth_sigma_m=depth_sigma_m,
        invalid_depth_frac=invalid_depth_frac,
        mask_morph_px=mask_morph_px,
        rng=rng,
    )
    rgb = None
    if write_rgb:
        rgb = np.repeat((np.clip(masks, 0, 1) * 200).astype(np.uint8)[..., None], 3, axis=-1)
    spec = SyntheticSpec(
        delta_t_gt=float(delta_t_gt),
        seed=int(seed),
        n_links=int(n_links),
        mount=mount,
        width=int(width),
        height=int(height),
        fps=float(fps),
        joint_rate_hz=float(joint_rate_hz),
        duration_s=float(duration_s),
        n_knots=int(n_knots),
        depth_sigma_m=float(depth_sigma_m),
        invalid_depth_frac=float(invalid_depth_frac),
        mask_morph_px=int(mask_morph_px),
        constant_velocity=bool(constant_velocity),
        extrinsic=[[float(v) for v in row] for row in T_extrinsic.tolist()],
    )
    data_mod.write_dataset(
        out_dir,
        joint_stamps=t_r,
        joint_q=q_log,
        cam_stamps=stamps,
        masks=masks,
        depths=depths,
        intrinsics=intr,
        mount=mount,
        fps=fps,
        joint_rate_hz=joint_rate_hz,
        depth_scale=0.001,
        rgb=rgb,
        urdf=None,
        joint_names=["j%d" % i for i in range(dof)],
        ground_truth=spec.to_dict(),
        overwrite=overwrite,
    )
    meta_path = out_dir / "meta.yaml"
    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    meta["robot_model"] = "box_links"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return {
        "path": Path(out_dir).resolve(),
        "T_extrinsic": T_extrinsic,
        "delta_t_gt": float(delta_t_gt),
        "spec": spec,
        "n_frames": int(stamps.numel()),
        "cam_stamps": stamps,
        "q_true": q_true,
        "trajectory": trajectory,
        "robot": robot,
        "masks": masks,
        "depths": depths,
        "depth_noiseless": depth_clean,
    }


def constant_velocity_recording(
    out_dir: str | Path, delta_t_gt: float = DELTA_T_GT_DEFAULT, seed: int = 0, **kwargs: Any
) -> dict[str, Any]:
    kwargs.pop("constant_velocity", None)
    return make_synthetic_recording(
        out_dir, delta_t_gt=delta_t_gt, seed=seed, constant_velocity=True, **kwargs
    )


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    gt = dict(data_mod.read_ground_truth(path))
    if "delta_t_seconds" in gt:
        gt.setdefault("delta_t_s", float(gt["delta_t_seconds"]))
    elif "delta_t_s" in gt:
        gt.setdefault("delta_t_seconds", float(gt["delta_t_s"]))
    else:
        raise KeyError(
            "%r has a ground_truth.yaml with no latency field; expected 'delta_t_seconds' (as written by SyntheticSpec)"
            % (str(path),)
        )
    gt["delta_t_s"] = float(gt["delta_t_s"])
    if "extrinsic" not in gt:
        raise KeyError("%r has a ground_truth.yaml with no 'extrinsic'" % (str(path),))
    return gt


def _camera_from(intr: Intrinsics) -> Any:
    from .render import PinholeCamera

    return PinholeCamera(intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height)
