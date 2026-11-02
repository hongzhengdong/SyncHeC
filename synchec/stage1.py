from __future__ import annotations
import math
import torch
from . import lie
from .config import Config


def robust_surface_depth(
    depth: torch.Tensor, base_mask: torch.Tensor, valid: torch.Tensor | None = None
) -> float:
    sel = base_mask > 0.5
    if valid is not None:
        sel = sel & valid
    sel = sel & (depth > 0) & torch.isfinite(depth)
    n = int(sel.sum())
    if n == 0:
        raise ValueError(
            "base-link mask and valid depth do not intersect: cannot anchor. Check the segmentation and that the depth frame is not all zero."
        )
    return float(torch.quantile(depth[sel], 0.5))


def solve_center_distance(
    d_surf: float, r_base: float, height: float, exact: bool = False, iters: int = 50
) -> float:
    if d_surf <= abs(height):
        raise ValueError(
            "d_surf (%.4f m) must exceed the camera height (%.4f m); the side-profile model has no real solution otherwise."
            % (d_surf, abs(height))
        )
    if not exact:
        l0 = d_surf
        return d_surf + r_base * l0 / math.sqrt(l0 * l0 - height * height)
    L = d_surf + r_base
    for _ in range(iters):
        denom = math.sqrt(max(L * L - height * height, 1e-12))
        nxt = d_surf + r_base * L / denom
        if abs(nxt - L) < 1e-12:
            return nxt
        L = nxt
    return L


def sample_view_sphere(
    radius: float,
    n: int,
    center: torch.Tensor,
    elevation_range: tuple[float, float] = (0.0, 1.2),
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive, got %r" % (radius,))
    if n < 1:
        raise ValueError("n must be positive")
    idx = torch.arange(n, dtype=torch.float64, device=center.device)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    (lo, hi) = elevation_range
    elev = lo + (hi - lo) * (idx + 0.5) / n
    azim = golden * idx
    x = radius * torch.cos(elev) * torch.cos(azim)
    y = radius * torch.cos(elev) * torch.sin(azim)
    z = radius * torch.sin(elev)
    eye = torch.stack([x, y, z], dim=-1).to(dtype) + center.to(dtype)
    return look_at(eye, center.to(dtype).expand_as(eye))


def look_at(
    eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor | None = None
) -> torch.Tensor:
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=eye.dtype, device=eye.device)
    up = up.expand_as(eye)
    fwd = target - eye
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    parallel = (fwd * up).sum(-1, keepdim=True).abs() > 1.0 - 1e-06
    alt = torch.tensor([1.0, 0.0, 0.0], dtype=eye.dtype, device=eye.device).expand_as(eye)
    up = torch.where(parallel, alt, up)
    right = torch.cross(up, fwd, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    down = torch.cross(fwd, right, dim=-1)
    R_wc = torch.stack([right, down, fwd], dim=-2)
    t = -(R_wc @ eye.unsqueeze(-1)).squeeze(-1)
    return lie.compose(R_wc, t)


def joint_fitness(
    rend_mask: torch.Tensor,
    rend_depth: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_depth: torch.Tensor,
    lambda_init: float,
    epsilon: float,
    depth_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    r = rend_mask > epsilon
    o = obs_mask > 0.5
    inter = (r & o).flatten(-2).sum(-1).to(rend_mask.dtype)
    union = (r | o).flatten(-2).sum(-1).to(rend_mask.dtype)
    iou = inter / union.clamp_min(1.0)
    omega = r & o
    if depth_valid is not None:
        omega = omega & depth_valid
    valid = torch.isfinite(obs_depth) & (obs_depth > 0)
    omega = omega & valid
    safe_depth = torch.where(valid, obs_depth, torch.zeros_like(obs_depth))
    resid = torch.where(omega, (rend_depth - safe_depth).abs(), torch.zeros_like(rend_depth))
    n = omega.flatten(-2).sum(-1).to(rend_depth.dtype)
    mean_resid = resid.flatten(-2).sum(-1) / n.clamp_min(1.0)
    return iou - lambda_init * mean_resid


def select_anchor(
    robot,
    camera,
    rasterizer,
    q_static: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_depth: torch.Tensor,
    cfg: Config,
    base_mask: torch.Tensor | None = None,
    depth_valid: torch.Tensor | None = None,
    mount: str = "eth",
    verbose: bool = True,
) -> dict:
    from .kinematics import LinkPoseModel

    dtype = obs_depth.dtype
    if base_mask is None:
        raise ValueError("Stage 1 requires a base-link mask")
    if base_mask.shape != obs_depth.shape:
        raise ValueError("Base-link mask and depth must have the same shape")
    d_surf = robust_surface_depth(obs_depth, base_mask, depth_valid)
    L = solve_center_distance(
        d_surf, cfg.stage_one.base_cylinder_radius_m, cfg.stage_one.camera_height_m
    )
    radius = L + cfg.stage_one.delta_r_m
    center = torch.zeros(3, dtype=dtype, device=obs_depth.device)
    cands = sample_view_sphere(radius, cfg.stage_one.n_sphere_samples, center, dtype=dtype)
    pose_model = LinkPoseModel(robot, mount=mount)
    q = q_static.reshape(1, -1).to(obs_depth)
    if mount == "eih":
        cands = cands @ robot.flange(q)[0]
    scores = torch.empty(cands.shape[0], dtype=dtype, device=obs_depth.device)
    with torch.no_grad():
        for i in range(cands.shape[0]):
            link_T = pose_model(cands[i], q)
            (verts, faces) = robot.posed_mesh(link_T)
            rend = rasterizer(verts, faces)
            scores[i] = joint_fitness(
                rend["mask"][0],
                rend["depth"][0],
                obs_mask,
                obs_depth,
                cfg.loss.lambda_init,
                cfg.loss.epsilon,
                depth_valid,
            )
    best = int(torch.argmax(scores))
    if verbose:
        print(
            "  d_surf=%.4f m  L=%.4f m  radius=%.4f m  best candidate %d/%d score=%.4f"
            % (d_surf, L, radius, best, cands.shape[0], float(scores[best]))
        )
    return {
        "T": cands[best].clone(),
        "score": float(scores[best]),
        "d_surf": d_surf,
        "L": L,
        "radius": radius,
        "scores": scores,
        "candidates": cands,
        "best_index": best,
    }


def refine_anchor(
    robot,
    camera,
    rasterizer,
    q_static: torch.Tensor,
    obs_mask: torch.Tensor,
    T_init: torch.Tensor,
    cfg: Config,
    mount: str = "eth",
    verbose: bool = True,
) -> torch.Tensor:
    from .kinematics import LinkPoseModel

    pose_model = LinkPoseModel(robot, mount=mount)
    (phi, rho) = lie.se3_log(T_init)
    phi = phi.clone().detach().requires_grad_(True)
    rho = rho.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam(
        [{"params": [phi], "lr": cfg.optim.eta_rot}, {"params": [rho], "lr": cfg.optim.eta_trans}]
    )
    q = q_static.reshape(1, -1).to(obs_mask.dtype)
    target = obs_mask.unsqueeze(0)
    for step in range(cfg.stage_one.refine_steps):
        opt.zero_grad(set_to_none=True)
        link_T = pose_model(lie.se3_exp(phi, rho), q)
        (verts, faces) = robot.posed_mesh(link_T)
        rend = rasterizer(verts, faces)
        loss = ((rend["mask"] - target) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (step + 1) % max(1, cfg.stage_one.refine_steps // 4) == 0:
            print(
                "    refine %3d/%d  mask MSE %.6f"
                % (step + 1, cfg.stage_one.refine_steps, float(loss.detach()))
            )
    return lie.se3_exp(phi.detach(), rho.detach())
