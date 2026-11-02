from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Callable
import torch
from . import lie

Projector = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BSplineTrajectory:
    def __init__(self, control_points: torch.Tensor, duration: float = 6.5, degree: int = 3):
        if control_points.ndim != 2:
            raise ValueError(
                "control_points must be (K, D), got shape %s" % (tuple(control_points.shape),)
            )
        (k, _) = control_points.shape
        if degree < 1:
            raise ValueError("degree must be >= 1, got %d" % (degree,))
        if k < degree + 1:
            raise ValueError(
                "a degree-%d spline needs at least %d control points, got %d"
                % (degree, degree + 1, k)
            )
        if not duration > 0:
            raise ValueError("duration must be positive, got %r" % (duration,))
        self.control_points = control_points
        self.duration = float(duration)
        self.degree = int(degree)
        self.knots = self._make_knots()

    @property
    def n_knots(self) -> int:
        return int(self.control_points.shape[0])

    @property
    def dof(self) -> int:
        return int(self.control_points.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.control_points.dtype

    @property
    def device(self) -> torch.device:
        return self.control_points.device

    def __repr__(self) -> str:
        return "BSplineTrajectory(K=%d, dof=%d, duration=%.3f s, degree=%d)" % (
            self.n_knots,
            self.dof,
            self.duration,
            self.degree,
        )

    def _make_knots(self) -> torch.Tensor:
        (p, k) = (self.degree, self.n_knots)
        n_interior = k - p - 1
        interior = torch.linspace(0.0, self.duration, n_interior + 2, dtype=torch.float64)[1:-1]
        return torch.cat(
            [
                torch.zeros(p + 1, dtype=torch.float64),
                interior,
                torch.full((p + 1,), self.duration, dtype=torch.float64),
            ]
        )

    def sample_times(self, n: int | None = None) -> torch.Tensor:
        n = self.n_knots if n is None else int(n)
        if n < 1:
            raise ValueError("n must be >= 1, got %d" % (n,))
        if n == 1:
            return torch.zeros(1, dtype=torch.float64, device=self.device)
        return torch.linspace(0.0, self.duration, n, dtype=torch.float64, device=self.device)

    def _basis_levels(self, t: torch.Tensor) -> list[torch.Tensor]:
        kn = self.knots.to(t.device)
        m = kn.shape[0]
        t = t.reshape(-1)
        n0 = torch.zeros(t.shape[0], m - 1, dtype=torch.float64, device=t.device)
        spans = kn[1:] > kn[:-1]
        for i in range(m - 1):
            if bool(spans[i]):
                n0[:, i] = ((kn[i] <= t) & (t < kn[i + 1])).to(torch.float64)
        last = int(torch.nonzero(spans, as_tuple=False)[-1])
        n0[:, last] = torch.where(t == kn[-1], torch.ones_like(n0[:, last]), n0[:, last])
        levels = [n0]
        for k in range(1, self.degree + 1):
            prev = levels[-1]
            cur = torch.zeros(t.shape[0], m - k - 1, dtype=torch.float64, device=t.device)
            for i in range(m - k - 1):
                d1 = kn[i + k] - kn[i]
                d2 = kn[i + k + 1] - kn[i + 1]
                if float(d1) > 0:
                    cur[:, i] = cur[:, i] + (t - kn[i]) / d1 * prev[:, i]
                if float(d2) > 0:
                    cur[:, i] = cur[:, i] + (kn[i + k + 1] - t) / d2 * prev[:, i + 1]
            levels.append(cur)
        return levels

    def _derivative_basis(self, lower: torch.Tensor, p: int) -> torch.Tensor:
        kn = self.knots.to(lower.device)
        n_out = lower.shape[1] - 1
        out = torch.zeros(lower.shape[0], n_out, dtype=torch.float64, device=lower.device)
        for i in range(n_out):
            d1 = kn[i + p] - kn[i]
            d2 = kn[i + p + 1] - kn[i + 1]
            if float(d1) > 0:
                out[:, i] = out[:, i] + p * lower[:, i] / d1
            if float(d2) > 0:
                out[:, i] = out[:, i] - p * lower[:, i + 1] / d2
        return out

    def basis(self, t: torch.Tensor, order: int = 0) -> torch.Tensor:
        if order < 0:
            raise ValueError("order must be >= 0, got %d" % (order,))
        t = torch.as_tensor(t, dtype=torch.float64, device=self.device).reshape(-1)
        if order > self.degree:
            return torch.zeros(t.shape[0], self.n_knots, dtype=self.dtype, device=self.device)
        levels = self._basis_levels(t)
        b = levels[self.degree - order]
        for j in range(order):
            b = self._derivative_basis(b, self.degree - order + j + 1)
        return b[:, : self.n_knots].to(self.dtype)

    def q(self, t: torch.Tensor) -> torch.Tensor:
        return self.basis(t, 0) @ self.control_points

    def qdot(self, t: torch.Tensor) -> torch.Tensor:
        return self.basis(t, 1) @ self.control_points

    def qddot(self, t: torch.Tensor) -> torch.Tensor:
        return self.basis(t, 2) @ self.control_points

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.q(t), self.qdot(t))

    def resample(self, rate_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(math.floor(self.duration * float(rate_hz))) + 1
        t = torch.arange(n, dtype=torch.float64, device=self.device) / float(rate_hz)
        t = t.clamp(max=self.duration)
        return (t, self.q(t))

    def speed_profile(self, n: int = 200) -> torch.Tensor:
        return self.qdot(self.sample_times(n)).norm(dim=-1)

    def direction_spread(self, n: int = 200) -> float:
        v = self.qdot(self.sample_times(n))
        nrm = v.norm(dim=-1, keepdim=True)
        keep = (nrm > 1e-09).squeeze(-1)
        if int(keep.sum()) == 0:
            return 0.0
        u = v[keep] / nrm[keep]
        return float(1.0 - u.mean(dim=0).norm())

    @classmethod
    def from_waypoints(
        cls, waypoints: torch.Tensor, duration: float = 6.5, degree: int = 3
    ) -> "BSplineTrajectory":
        return cls(waypoints, duration=duration, degree=degree)

    @classmethod
    def random(
        cls,
        n_knots: int = 25,
        dof: int = 7,
        duration: float = 6.5,
        limits: tuple[torch.Tensor, torch.Tensor] | None = None,
        scale: float = 0.6,
        seed: int | None = None,
        degree: int = 3,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "BSplineTrajectory":
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        step = torch.randn(n_knots, dof, generator=gen, dtype=torch.float64) * scale
        cp = torch.cumsum(step, dim=0) / max(1.0, math.sqrt(n_knots) * 0.5)
        cp = cp - cp.mean(dim=0, keepdim=True)
        if limits is not None:
            (lo, hi) = (x.to(torch.float64) for x in limits)
            mid = 0.5 * (lo + hi)
            half = 0.45 * (hi - lo)
            span = cp.abs().amax(dim=0).clamp_min(1e-09)
            cp = mid + cp / span * half
        return cls(cp.to(dtype=dtype, device=device), duration=duration, degree=degree)


def link_landmarks(
    robot: Any, n_per_link: int = 24, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    meshes = getattr(robot, "link_meshes", None)
    if not meshes:
        raise ValueError(
            "robot has no non-empty .link_meshes, so landmarks cannot be sampled; pass an explicit points/link_index pair instead"
        )
    names = list(getattr(robot, "link_names", list(meshes.keys())))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pts: list[torch.Tensor] = []
    idx: list[torch.Tensor] = []
    for li, name in enumerate(names):
        entry = meshes.get(name)
        if entry is None:
            continue
        verts = entry[0] if isinstance(entry, (tuple, list)) else entry
        verts = torch.as_tensor(verts, dtype=torch.float32).reshape(-1, 3)
        if verts.shape[0] == 0:
            continue
        take = min(int(n_per_link), verts.shape[0])
        sel = torch.randperm(verts.shape[0], generator=gen)[:take]
        pts.append(verts[sel])
        idx.append(torch.full((take,), li, dtype=torch.long))
    if not pts:
        raise ValueError("every link mesh was empty; nothing to project")
    return (torch.cat(pts, dim=0), torch.cat(idx, dim=0))


class LinkProjector:
    def __init__(
        self,
        robot: Any,
        camera: Any,
        points: torch.Tensor,
        link_index: torch.Tensor,
        mount: str = "eth",
    ):
        from .kinematics import LinkPoseModel

        self.robot = robot
        self.camera = camera
        self.points = points
        self.link_index = link_index
        self.pose_model = LinkPoseModel(robot, mount=mount)

    @classmethod
    def from_robot(
        cls, robot: Any, camera: Any, n_per_link: int = 24, mount: str = "eth", seed: int = 0
    ) -> "LinkProjector":
        (pts, idx) = link_landmarks(robot, n_per_link=n_per_link, seed=seed)
        return cls(robot, camera, pts, idx, mount=mount)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def __call__(self, T_extrinsic: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        link_T = self.pose_model(T_extrinsic, q)
        T_pts = link_T[:, self.link_index]
        p = self.points.to(dtype=T_pts.dtype, device=T_pts.device)
        cam = (T_pts[..., :3, :3] @ p.unsqueeze(-1)).squeeze(-1) + T_pts[..., :3, 3]
        return self.camera.project(cam)


def candidate_cameras(
    T_anchor: torch.Tensor, n: int = 8, angle_deg: float = 8.0, trans_m: float = 0.05, seed: int = 0
) -> torch.Tensor:
    if n < 2:
        raise ValueError("T_cand needs at least 2 cameras for a variance, got %d" % (n,))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    (dtype, device) = (T_anchor.dtype, T_anchor.device)
    out = [T_anchor]
    for _ in range(n - 1):
        phi = torch.randn(3, generator=gen, dtype=torch.float64)
        phi = phi / phi.norm().clamp_min(1e-12) * math.radians(angle_deg)
        rho = torch.randn(3, generator=gen, dtype=torch.float64)
        rho = rho / rho.norm().clamp_min(1e-12) * float(trans_m)
        dT = lie.se3_exp(phi.to(dtype=dtype, device=device), rho.to(dtype=dtype, device=device))
        out.append(dT @ T_anchor)
    return torch.stack(out, dim=0)


def spatial_excitation(
    project: Projector,
    q: torch.Tensor,
    T_cand: torch.Tensor,
    in_view_only: bool = True,
    camera: Any = None,
    soft_px: float = 20.0,
) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError("q must be (N, D), got %s" % (tuple(q.shape),))
    if T_cand.ndim != 3 or T_cand.shape[-2:] != (4, 4):
        raise ValueError("T_cand must be (C, 4, 4), got %s" % (tuple(T_cand.shape),))
    c = T_cand.shape[0]
    if c < 2:
        raise ValueError("a variance needs C >= 2 candidate cameras, got %d" % (c,))
    px = torch.stack([project(T_cand[i], q) for i in range(c)], dim=0)
    var = px.var(dim=0, unbiased=False)
    if in_view_only:
        cam = camera if camera is not None else getattr(project, "camera", None)
        if cam is not None:
            w = float(getattr(cam, "width"))
            h = float(getattr(cam, "height"))
            s = max(float(soft_px), 1e-06)
            (u, v) = (px[..., 0], px[..., 1])
            wu = torch.sigmoid(u / s) * torch.sigmoid((w - u) / s)
            wv = torch.sigmoid(v / s) * torch.sigmoid((h - v) / s)
            weight = (wu * wv).mean(dim=0)
            var = var * weight.unsqueeze(-1)
    return var.sum(dim=(1, 2))


def temporal_excitation(
    project: Projector,
    q: torch.Tensor,
    qdot: torch.Tensor,
    T_ref: torch.Tensor,
    method: str = "double-backward",
    eps: float = 1e-12,
) -> torch.Tensor:
    if q.shape != qdot.shape:
        raise ValueError(
            "q and qdot must have the same shape, got %s and %s"
            % (tuple(q.shape), tuple(qdot.shape))
        )
    if method not in ("jvp", "double-backward"):
        raise ValueError("method must be jvp or double-backward")

    def f(x: torch.Tensor) -> torch.Tensor:
        return project(T_ref, x)

    if method == "jvp":
        _, flow = torch.func.jvp(f, (q,), (qdot,))
    else:
        flow = _jvp_double_backward(f, q, qdot)
    return flow.pow(2).sum(dim=tuple(range(1, flow.ndim))).clamp_min(eps).sqrt()


def _jvp_double_backward(
    f: Callable[[torch.Tensor], torch.Tensor], q: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    q_in = q if q.requires_grad else q.detach().requires_grad_(True)
    y = f(q_in)
    u = torch.zeros_like(y, requires_grad=True)
    (gq,) = torch.autograd.grad((y * u).sum(), q_in, create_graph=True)
    (jv,) = torch.autograd.grad((gq * v).sum(), u, create_graph=True)
    return jv


@dataclass
class PlanReport:
    objective: list[float] = field(default_factory=list)
    v_space: list[float] = field(default_factory=list)
    v_time: list[float] = field(default_factory=list)
    penalty: list[float] = field(default_factory=list)
    n_steps: int = 0
    alpha: float = 5e-4
    beta: float = 2.5e-3
    collision_hook: bool = False
    visibility_hook: bool = False
    limits_enforced: bool = False
    collision_rejections: int = 0
    seconds: float = 0.0

    @property
    def improved(self) -> bool:
        return bool(self.objective) and self.objective[-1] >= self.objective[0]

    def constraint_summary(self) -> str:
        return f"joint_limits={self.limits_enforced} collision_check={self.collision_hook} visibility_penalty={self.visibility_hook}"

    def summary(self) -> str:
        if not self.objective:
            return "no steps ran"
        return (
            "steps=%d  objective %.4g -> %.4g  V_space %.4g -> %.4g  V_time %.4g -> %.4g  (%.1f s)\n  constraints: %s"
            % (
                self.n_steps,
                self.objective[0],
                self.objective[-1],
                self.v_space[0],
                self.v_space[-1],
                self.v_time[0],
                self.v_time[-1],
                self.seconds,
                self.constraint_summary(),
            )
        )


def excitation_objective(
    traj: BSplineTrajectory,
    project: Projector,
    T_cand: torch.Tensor,
    alpha: float = 5e-4,
    beta: float = 2.5e-3,
    n_eval: int | None = None,
    T_ref: torch.Tensor | None = None,
    jvp_method: str = "double-backward",
    camera: Any = None,
) -> dict[str, torch.Tensor]:
    t = traj.sample_times(n_eval)
    (q, qdot) = traj(t)
    v_s = spatial_excitation(project, q, T_cand, camera=camera)
    v_t = temporal_excitation(
        project, q, qdot, T_cand[0] if T_ref is None else T_ref, method=jvp_method
    )
    total = alpha * v_s.sum() + beta * v_t.sum()
    return {
        "total": total,
        "v_space": v_s.sum(),
        "v_time": v_t.sum(),
        "per_knot_space": v_s,
        "per_knot_time": v_t,
        "q": q,
        "qdot": qdot,
    }


def plan(
    project: Projector,
    T_cand: torch.Tensor,
    *,
    alpha: float = 5e-4,
    beta: float = 2.5e-3,
    init: BSplineTrajectory | None = None,
    n_knots: int = 25,
    dof: int | None = None,
    duration: float = 6.5,
    limits: tuple[torch.Tensor, torch.Tensor] | None = None,
    n_steps: int = 100,
    lr: float = 0.01,
    n_eval: int | None = None,
    collision_free: Callable[[torch.Tensor], bool] | None = None,
    visibility: Callable[[torch.Tensor], torch.Tensor] | None = None,
    limit_penalty: float = 0.0,
    qdot_max: float | None = None,
    qdot_penalty: float = 0.0,
    jvp_method: str = "double-backward",
    camera: Any = None,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[BSplineTrajectory, PlanReport]:
    import time

    if alpha == 0.0 and beta == 0.0:
        raise ValueError(
            "alpha and beta cannot both be zero; Eq. (12) would be identically flat and the optimizer would wander on numerical noise"
        )
    if init is None:
        if dof is None:
            if limits is not None:
                dof = int(torch.as_tensor(limits[0]).reshape(-1).shape[0])
            else:
                raise ValueError("pass init=, or dof=, or limits= so dof is known")
        init = BSplineTrajectory.random(
            n_knots=n_knots, dof=dof, duration=duration, limits=limits, seed=seed
        )
    cp = init.control_points.detach().clone()
    if limits is not None:
        lo = torch.as_tensor(limits[0], dtype=cp.dtype, device=cp.device).reshape(-1)
        hi = torch.as_tensor(limits[1], dtype=cp.dtype, device=cp.device).reshape(-1)
        if lo.shape[0] != cp.shape[1] or hi.shape[0] != cp.shape[1]:
            raise ValueError(
                "limits must each be (D,) with D = %d, got %d and %d"
                % (cp.shape[1], lo.shape[0], hi.shape[0])
            )
        if not bool((hi > lo).all()):
            raise ValueError("every joint limit must satisfy hi > lo")
        cp = cp.clamp(lo, hi)
    else:
        lo = hi = None
    cp = cp.requires_grad_(True)
    traj = BSplineTrajectory(cp, duration=init.duration, degree=init.degree)
    opt = torch.optim.Adam([cp], lr=lr)
    rep = PlanReport(
        alpha=alpha,
        beta=beta,
        collision_hook=collision_free is not None,
        visibility_hook=visibility is not None,
        limits_enforced=limits is not None,
    )
    t0 = time.time()
    check_count = max(200, n_eval or n_knots)
    if collision_free is not None:
        with torch.no_grad():
            if not bool(collision_free(traj.q(traj.sample_times(check_count)))):
                raise ValueError("The initial spline violates collision constraints")
    for _ in range(int(n_steps)):
        prev = cp.detach().clone()
        opt.zero_grad(set_to_none=True)
        terms = excitation_objective(
            traj,
            project,
            T_cand,
            alpha=alpha,
            beta=beta,
            n_eval=n_eval,
            jvp_method=jvp_method,
            camera=camera,
        )
        obj = terms["total"]
        pen = torch.zeros((), dtype=obj.dtype, device=obj.device)
        if visibility is not None:
            pen = pen + visibility(terms["q"]).reshape(())
        if limit_penalty and lo is not None:
            over = (cp - hi).clamp_min(0.0) + (lo - cp).clamp_min(0.0)
            pen = pen + limit_penalty * over.pow(2).sum()
        if qdot_penalty and qdot_max is not None:
            excess = (terms["qdot"].norm(dim=-1) - float(qdot_max)).clamp_min(0.0)
            pen = pen + qdot_penalty * excess.pow(2).sum()
        if not torch.isfinite(obj - pen):
            raise FloatingPointError("Trajectory objective is nonfinite")
        (-(obj - pen)).backward()
        if cp.grad is None or not torch.isfinite(cp.grad).all():
            raise FloatingPointError("Trajectory gradient is missing or nonfinite")
        opt.step()
        with torch.no_grad():
            if lo is not None:
                cp.clamp_(lo, hi)
        rejected = False
        if collision_free is not None:
            with torch.no_grad():
                q_chk = traj.q(traj.sample_times(check_count))
            if not bool(collision_free(q_chk)):
                with torch.no_grad():
                    cp.copy_(prev)
                rep.collision_rejections += 1
                rejected = True
        rep.objective.append(float(obj.detach()))
        rep.v_space.append(float(terms["v_space"].detach()))
        rep.v_time.append(float(terms["v_time"].detach()))
        rep.penalty.append(float(pen.detach()))
        rep.n_steps += 1
        if verbose:
            print(
                "  step %3d  obj=%.6g  V_space=%.6g  V_time=%.6g  pen=%.3g%s"
                % (
                    rep.n_steps,
                    rep.objective[-1],
                    rep.v_space[-1],
                    rep.v_time[-1],
                    rep.penalty[-1],
                    "  [collision: rolled back]" if rejected else "",
                )
            )
    final_terms = excitation_objective(
        traj,
        project,
        T_cand,
        alpha=alpha,
        beta=beta,
        n_eval=n_eval,
        jvp_method=jvp_method,
        camera=camera,
    )
    final_penalty = torch.zeros_like(final_terms["total"])
    if visibility is not None:
        final_penalty = final_penalty + visibility(final_terms["q"]).reshape(())
    if qdot_penalty and qdot_max is not None:
        excess = (final_terms["qdot"].norm(dim=-1) - float(qdot_max)).clamp_min(0)
        final_penalty = final_penalty + qdot_penalty * excess.square().sum()
    rep.objective.append(float(final_terms["total"].detach()))
    rep.v_space.append(float(final_terms["v_space"].detach()))
    rep.v_time.append(float(final_terms["v_time"].detach()))
    rep.penalty.append(float(final_penalty.detach()))
    rep.seconds = time.time() - t0
    final = BSplineTrajectory(cp.detach().clone(), duration=traj.duration, degree=traj.degree)
    return (final, rep)
