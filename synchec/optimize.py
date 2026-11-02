from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from . import lie
from .config import Config
from .kinematics import JointLog, LinkPoseModel, q_sync
from .render import losses


@dataclass
class History:
    delta_t_ms: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    mask: list[float] = field(default_factory=list)
    depth: list[float] = field(default_factory=list)
    reg: list[float] = field(default_factory=list)
    trans_step_mm: list[float] = field(default_factory=list)
    rot_step_deg: list[float] = field(default_factory=list)
    dt_step_ms: list[float] = field(default_factory=list)
    n_subset: list[int] = field(default_factory=list)
    temporal_updates: list[int] = field(default_factory=list)
    spatial_updates: list[int] = field(default_factory=list)
    stopped_early: bool = False
    n_cycles: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        if not self.total:
            return "cycles=0"
        return (
            f"cycles={self.n_cycles} loss={self.total[-1]:.6g} "
            f"offset={self.delta_t_ms[-1]:.3f} ms "
            f"temporal_updates={sum(self.temporal_updates)} "
            f"spatial_updates={sum(self.spatial_updates)} "
            f"seconds={self.seconds:.1f}"
        )


@dataclass
class Result:
    T: torch.Tensor
    delta_t_s: float
    history: History

    @property
    def delta_t_ms(self) -> float:
        return self.delta_t_s * 1e3


class CalibrationProblem:
    def __init__(
        self,
        robot,
        camera,
        rasterizer,
        joint_log: JointLog,
        cam_stamps: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        depth_valid: torch.Tensor | None,
        cfg: Config,
        mount: str = "eth",
    ):
        cfg.validate()
        n = cam_stamps.numel()
        if n == 0 or cam_stamps.ndim != 1:
            raise ValueError("cam_stamps must be a nonempty vector")
        if masks.shape != (n, camera.height, camera.width) or depths.shape != masks.shape:
            raise ValueError("Observations must have shape (N, height, width)")
        if depth_valid is not None and depth_valid.shape != depths.shape:
            raise ValueError("depth_valid must match depth observations")
        if not torch.isfinite(cam_stamps).all() or not torch.isfinite(masks).all():
            raise ValueError("Camera timestamps and masks must be finite")
        if not ((masks >= 0) & (masks <= 1)).all():
            raise ValueError("Mask values must lie in [0, 1]")
        if joint_log.dof != robot.dof:
            raise ValueError("Robot and joint log dimensions differ")
        tensors = [cam_stamps, masks, depths, joint_log.q, joint_log.stamps]
        if len({x.device for x in tensors}) != 1:
            raise ValueError("All recording tensors must use the same device")
        self.robot = robot
        self.camera = camera
        self.rasterizer = rasterizer
        self.log = joint_log
        self.cam_stamps = cam_stamps
        self.masks = masks
        self.depths = depths
        self.depth_valid = depth_valid
        self.cfg = cfg
        self.pose_model = LinkPoseModel(robot, mount=mount)
        self.offset_bounds = (
            float(cam_stamps.max() - joint_log.stamps[-1]),
            float(cam_stamps.min() - joint_log.stamps[0]),
        )
        if self.offset_bounds[0] >= self.offset_bounds[1]:
            raise ValueError("The joint log must cover the camera sequence with a temporal margin")

    def loss(self, phi, rho, delta_t, frames=None):
        stamps = self.cam_stamps if frames is None else self.cam_stamps[frames]
        mask = self.masks if frames is None else self.masks[frames]
        depth = self.depths if frames is None else self.depths[frames]
        valid = self.depth_valid
        if valid is not None and frames is not None:
            valid = valid[frames]
        q = q_sync(self.log, stamps, delta_t)
        link_T = self.pose_model(lie.se3_exp(phi, rho), q)
        verts, faces = self.robot.posed_mesh(link_T)
        return losses(
            self.rasterizer(verts, faces),
            mask,
            depth,
            lambda_depth=self.cfg.loss.lambda_depth,
            lambda_reg=self.cfg.loss.lambda_reg,
            delta_t=delta_t,
            epsilon=self.cfg.loss.epsilon,
            depth_valid=valid,
        )


def high_velocity_subset(log, cam_stamps, delta_t, tau_vel):
    speed = log.speed_at(cam_stamps.to(torch.float64) - delta_t.detach().to(torch.float64))
    subset = torch.nonzero(speed > tau_vel, as_tuple=False).squeeze(-1)
    if subset.numel() == 0:
        raise ValueError(f"No frame exceeds the temporal velocity threshold {tau_vel:g} rad/s")
    return subset


def _batches(indices, cfg):
    for _ in range(cfg.optim.passes_per_block):
        yield from indices.split(cfg.optim.batch_size)


def _average_loss(problem, phi, rho, delta_t, cfg):
    indices = torch.arange(problem.cam_stamps.numel(), device=phi.device)
    out = {}
    with torch.no_grad():
        for batch in indices.split(cfg.optim.batch_size):
            terms = problem.loss(phi, rho, delta_t, batch)
            weight = batch.numel() / indices.numel()
            for key in ("total", "mask", "depth", "reg"):
                out[key] = out.get(key, 0.0) + float(terms[key]) * weight
    return out


def _parameters(problem, T_init, delta_t_init):
    if T_init.shape != (4, 4) or not torch.isfinite(T_init).all():
        raise ValueError("The initial extrinsic must be a finite 4x4 transform")
    R = T_init[:3, :3]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    if not torch.allclose(R.T @ R, eye, atol=1e-4) or not torch.allclose(
        torch.det(R), R.new_tensor(1.0), atol=1e-4
    ):
        raise ValueError("The initial extrinsic must contain a proper rotation")
    if not torch.allclose(T_init[3], T_init.new_tensor([0, 0, 0, 1]), atol=1e-6):
        raise ValueError("The initial extrinsic must have homogeneous last row [0, 0, 0, 1]")
    if T_init.device != problem.masks.device or T_init.dtype != problem.masks.dtype:
        raise ValueError("The initial extrinsic must match observation dtype and device")
    lo, hi = problem.offset_bounds
    if not math.isfinite(delta_t_init) or not lo <= delta_t_init <= hi:
        raise ValueError(f"The initial offset must lie in [{lo:g}, {hi:g}] seconds")
    phi, rho = lie.se3_log(T_init)
    return (
        phi.detach().clone().requires_grad_(True),
        rho.detach().clone().requires_grad_(True),
        T_init.new_tensor(delta_t_init, requires_grad=True),
    )


def _step(optimizer, terms, parameters):
    if not torch.isfinite(terms["total"]):
        raise FloatingPointError("Calibration loss is nonfinite")
    terms["total"].backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Calibration gradient is missing or nonfinite")
    optimizer.step()


def _record(hist, problem, phi, rho, dt, previous_T, previous_dt, cfg):
    values = _average_loss(problem, phi, rho, dt, cfg)
    for key, value in values.items():
        getattr(hist, key).append(value)
    T = lie.se3_exp(phi.detach(), rho.detach())
    d_trans = float((T[:3, 3] - previous_T[:3, 3]).norm())
    d_rot = float(lie.so3_log(previous_T[:3, :3].T @ T[:3, :3]).norm())
    d_time = abs(float(dt.detach()) - previous_dt)
    hist.delta_t_ms.append(float(dt.detach()) * 1e3)
    hist.trans_step_mm.append(d_trans * 1e3)
    hist.rot_step_deg.append(math.degrees(d_rot))
    hist.dt_step_ms.append(d_time * 1e3)
    hist.n_cycles += 1
    tol = cfg.derived()
    return d_trans < tol["tol_trans_m"] and d_rot < tol["tol_rot_rad"] and d_time < tol["tol_dt_s"]


def calibrate(problem, T_init, delta_t_init=0.0, cfg=None, verbose=True):
    cfg = cfg or problem.cfg
    cfg.validate()
    phi, rho, dt = _parameters(problem, T_init, delta_t_init)
    spatial = torch.optim.Adam(
        [
            {"params": [phi], "lr": cfg.optim.eta_rot},
            {"params": [rho], "lr": cfg.optim.eta_trans},
        ]
    )
    temporal = torch.optim.Adam([dt], lr=cfg.optim.eta_t)
    all_frames = torch.arange(problem.cam_stamps.numel(), device=phi.device)
    hist = History()
    start = time.perf_counter()
    for _ in range(cfg.optim.k_max):
        previous_T = lie.se3_exp(phi.detach(), rho.detach()).clone()
        previous_dt = float(dt.detach())
        subset = high_velocity_subset(problem.log, problem.cam_stamps, dt, cfg.optim.tau_vel)
        hist.n_subset.append(subset.numel())
        temporal_count = spatial_count = 0
        for batch in _batches(subset, cfg):
            temporal.zero_grad(set_to_none=True)
            _step(temporal, problem.loss(phi.detach(), rho.detach(), dt, batch), [dt])
            with torch.no_grad():
                dt.clamp_(*problem.offset_bounds)
            temporal_count += 1
        for batch in _batches(all_frames, cfg):
            spatial.zero_grad(set_to_none=True)
            _step(spatial, problem.loss(phi, rho, dt.detach(), batch), [phi, rho])
            spatial_count += 1
        hist.temporal_updates.append(temporal_count)
        hist.spatial_updates.append(spatial_count)
        stopped = _record(hist, problem, phi, rho, dt, previous_T, previous_dt, cfg)
        if verbose:
            print(
                f"cycle={hist.n_cycles} loss={hist.total[-1]:.6g} offset_ms={hist.delta_t_ms[-1]:.3f} temporal_updates={temporal_count} spatial_updates={spatial_count}"
            )
        if cfg.optim.early_stopping and stopped:
            hist.stopped_early = True
            break
    hist.seconds = time.perf_counter() - start
    return Result(lie.se3_exp(phi.detach(), rho.detach()), float(dt.detach()), hist)


def calibrate_joint(problem, T_init, delta_t_init=0.0, cfg=None, verbose=True):
    cfg = cfg or problem.cfg
    cfg.validate()
    phi, rho, dt = _parameters(problem, T_init, delta_t_init)
    optimizer = torch.optim.Adam(
        [
            {"params": [phi], "lr": cfg.optim.eta_rot},
            {"params": [rho], "lr": cfg.optim.eta_trans},
            {"params": [dt], "lr": cfg.optim.eta_t},
        ]
    )
    all_frames = torch.arange(problem.cam_stamps.numel(), device=phi.device)
    hist = History()
    start = time.perf_counter()
    for _ in range(cfg.optim.k_max):
        previous_T = lie.se3_exp(phi.detach(), rho.detach()).clone()
        previous_dt = float(dt.detach())
        subset = high_velocity_subset(problem.log, problem.cam_stamps, dt, cfg.optim.tau_vel)
        hist.n_subset.append(subset.numel())
        count = 0
        for active in (subset, all_frames):
            for batch in _batches(active, cfg):
                optimizer.zero_grad(set_to_none=True)
                _step(optimizer, problem.loss(phi, rho, dt, batch), [phi, rho, dt])
                with torch.no_grad():
                    dt.clamp_(*problem.offset_bounds)
                count += 1
        hist.temporal_updates.append(count)
        hist.spatial_updates.append(count)
        stopped = _record(hist, problem, phi, rho, dt, previous_T, previous_dt, cfg)
        if verbose:
            print(
                f"cycle={hist.n_cycles} loss={hist.total[-1]:.6g} offset_ms={hist.delta_t_ms[-1]:.3f} joint_updates={count}"
            )
        if cfg.optim.early_stopping and stopped:
            hist.stopped_early = True
            break
    hist.seconds = time.perf_counter() - start
    return Result(lie.se3_exp(phi.detach(), rho.detach()), float(dt.detach()), hist)


def evaluate(T_est, T_gt, delta_t_est_s, delta_t_gt_s):
    return {
        "E_t_cm": float((T_est[:3, 3] - T_gt[:3, 3]).norm()) * 100.0,
        "E_r_deg": math.degrees(float(lie.geodesic_angle(T_est[:3, :3], T_gt[:3, :3]))),
        "E_lat_ms": abs(delta_t_est_s - delta_t_gt_s) * 1e3,
    }


def calibrate_multicamera(
    problems, initial_extrinsics, delta_t_init=0.0, shared_offset=False, cfg=None, verbose=True
):
    if len(problems) < 2 or len(problems) != len(initial_extrinsics):
        raise ValueError("Provide at least two problems and one initial extrinsic per camera")
    cfg = cfg or problems[0].cfg
    cfg.validate()
    if any(p.cfg.loss != problems[0].cfg.loss for p in problems):
        raise ValueError("Camera objectives must use the same loss configuration")
    if not shared_offset:
        return [
            calibrate(p, T, delta_t_init, cfg, verbose)
            for p, T in zip(problems, initial_extrinsics)
        ]
    first = problems[0]
    for problem in problems[1:]:
        if not torch.equal(problem.log.stamps, first.log.stamps) or not torch.equal(
            problem.log.q, first.log.q
        ):
            raise ValueError("Shared-offset calibration requires a common joint log")
    parameters = [_parameters(p, T, delta_t_init) for p, T in zip(problems, initial_extrinsics)]
    dt = parameters[0][2]
    lo = max(p.offset_bounds[0] for p in problems)
    hi = min(p.offset_bounds[1] for p in problems)
    if not lo <= delta_t_init <= hi:
        raise ValueError("The shared offset is outside the common joint-log coverage")
    temporal = torch.optim.Adam([dt], lr=cfg.optim.eta_t)
    spatial = [
        torch.optim.Adam(
            [
                {"params": [phi], "lr": cfg.optim.eta_rot},
                {"params": [rho], "lr": cfg.optim.eta_trans},
            ]
        )
        for phi, rho, _ in parameters
    ]
    histories = [History() for _ in problems]
    start = time.perf_counter()
    for cycle in range(cfg.optim.k_max):
        previous = [lie.se3_exp(phi.detach(), rho.detach()).clone() for phi, rho, _ in parameters]
        previous_dt = float(dt.detach())
        subsets = [
            high_velocity_subset(p.log, p.cam_stamps, dt, cfg.optim.tau_vel) for p in problems
        ]
        chunks = [list(s.split(cfg.optim.batch_size)) for s in subsets]
        time_updates = 0
        for _ in range(cfg.optim.passes_per_block):
            for index in range(max(map(len, chunks))):
                temporal.zero_grad(set_to_none=True)
                objective = cfg.loss.lambda_reg * dt.square()
                for problem, (phi, rho, _), batches in zip(problems, parameters, chunks):
                    batch = batches[index % len(batches)]
                    values = problem.loss(phi.detach(), rho.detach(), dt, batch)
                    objective = objective + values["mask"] + cfg.loss.lambda_depth * values["depth"]
                _step(temporal, {"total": objective}, [dt])
                with torch.no_grad():
                    dt.clamp_(lo, hi)
                time_updates += 1
        stopping = []
        for problem, (phi, rho, _), optimizer, history, old_T, subset in zip(
            problems, parameters, spatial, histories, previous, subsets
        ):
            all_frames = torch.arange(problem.cam_stamps.numel(), device=phi.device)
            space_updates = 0
            for batch in _batches(all_frames, cfg):
                optimizer.zero_grad(set_to_none=True)
                _step(optimizer, problem.loss(phi, rho, dt.detach(), batch), [phi, rho])
                space_updates += 1
            history.n_subset.append(subset.numel())
            history.temporal_updates.append(time_updates)
            history.spatial_updates.append(space_updates)
            stopping.append(_record(history, problem, phi, rho, dt, old_T, previous_dt, cfg))
        if verbose:
            print(
                f"cycle={cycle + 1} cameras={len(problems)} shared_offset_ms={float(dt.detach()) * 1e3:.3f} temporal_updates={time_updates}"
            )
        if cfg.optim.early_stopping and all(stopping):
            for history in histories:
                history.stopped_early = True
            break
    elapsed = time.perf_counter() - start
    for history in histories:
        history.seconds = elapsed
    return [
        Result(lie.se3_exp(phi.detach(), rho.detach()), float(dt.detach()), history)
        for (phi, rho, _), history in zip(parameters, histories)
    ]
