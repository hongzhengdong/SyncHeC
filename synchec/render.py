from __future__ import annotations
from typing import Any
import torch

_TINY = 1e-12
_DEPTH_MASK_FLOOR = 0.001


class PinholeCamera:
    __slots__ = ("fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        if not (fx > 0 and fy > 0):
            raise ValueError("focal lengths must be positive, got fx=%r fy=%r" % (fx, fy))
        if width <= 0 or height <= 0:
            raise ValueError("image size must be positive, got %rx%r" % (width, height))
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)

    def __repr__(self) -> str:
        return "PinholeCamera(fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, %dx%d)" % (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.width,
            self.height,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def matrix(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=dtype
        )

    def project(self, points_cam: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
        z = points_cam[..., 2:3]
        z = torch.where(z.abs() < eps, torch.full_like(z, eps) * torch.sign(z + eps), z)
        u = self.fx * points_cam[..., 0:1] / z + self.cx
        v = self.fy * points_cam[..., 1:2] / z + self.cy
        return torch.cat([u, v], dim=-1)

    def project_world(self, points_world: torch.Tensor, T_cw: torch.Tensor) -> torch.Tensor:
        R = T_cw[..., :3, :3]
        t = T_cw[..., :3, 3]
        cam = (R @ points_world.unsqueeze(-1)).squeeze(-1) + t
        return self.project(cam)

    def pixel_grid(
        self, dtype: torch.dtype = torch.float32, device: torch.device | None = None
    ) -> torch.Tensor:
        v = torch.arange(self.height, dtype=dtype, device=device)
        u = torch.arange(self.width, dtype=dtype, device=device)
        (vv, uu) = torch.meshgrid(v, u, indexing="ij")
        return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)


def _segment_distance_sq(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(-1).clamp_min(_TINY)
    t = ((ap * ab).sum(-1) / denom).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    d = p - closest
    return (d * d).sum(-1)


def _signed_distance_to_triangles(pix: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    p = pix.unsqueeze(1)
    a = tri[..., 0, :].unsqueeze(-3)
    b = tri[..., 1, :].unsqueeze(-3)
    c = tri[..., 2, :].unsqueeze(-3)

    def edge(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
        e = v1 - v0
        r = p - v0
        return e[..., 0] * r[..., 1] - e[..., 1] * r[..., 0]

    (e0, e1, e2) = (edge(a, b), edge(b, c), edge(c, a))
    inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0) | (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
    d2 = torch.minimum(
        torch.minimum(_segment_distance_sq(p, a, b), _segment_distance_sq(p, b, c)),
        _segment_distance_sq(p, c, a),
    )
    dist = torch.sqrt(d2.clamp_min(_TINY))
    return torch.where(inside, dist, -dist)


def _barycentric_depth(pix: torch.Tensor, tri: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = pix.unsqueeze(1)
    (a, b, c) = (tri[..., i, :].unsqueeze(-3) for i in range(3))
    (v0, v1, v2) = (b - a, c - a, p - a)
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    safe = denom.abs() > _TINY
    denom = torch.where(safe, denom, torch.ones_like(denom))
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w1 = torch.where(safe, w1, torch.zeros_like(w1)).clamp(0.0, 1.0)
    w2 = torch.where(safe, w2, torch.zeros_like(w2)).clamp(0.0, 1.0)
    s = (w1 + w2).clamp_min(1.0)
    (w1, w2) = (w1 / s, w2 / s)
    w0 = 1.0 - w1 - w2
    zi = 1.0 / z.clamp_min(_TINY)
    inv = (
        w0 * zi[..., 0].unsqueeze(-2)
        + w1 * zi[..., 1].unsqueeze(-2)
        + w2 * zi[..., 2].unsqueeze(-2)
    )
    return 1.0 / inv.clamp_min(_TINY)


class SoftRasterizer:
    def __init__(
        self,
        camera: PinholeCamera,
        sigma_px: float = 1.0,
        gamma: float = 0.0001,
        face_chunk: int = 2048,
        z_near: float = 0.001,
        cull_sigma: float | None = 16.0,
        tile: int = 24,
        depth_gate_sigma: float = 3.0,
    ):
        if sigma_px <= 0:
            raise ValueError("sigma_px must be positive, got %r" % (sigma_px,))
        if gamma <= 0:
            raise ValueError("gamma must be positive, got %r" % (gamma,))
        if cull_sigma is not None and cull_sigma <= 0:
            raise ValueError("cull_sigma must be positive or None, got %r" % (cull_sigma,))
        self.camera = camera
        self.sigma_px = float(sigma_px)
        self.gamma = float(gamma)
        self.face_chunk = max(1, int(face_chunk))
        self.z_near = float(z_near)
        self.cull_sigma = None if cull_sigma is None else float(cull_sigma)
        self.tile = max(1, int(tile))
        self.depth_gate_sigma = float(depth_gate_sigma)
        if self.cull_sigma is not None and self.depth_gate_sigma >= self.cull_sigma:
            raise ValueError(
                "depth_gate_sigma (%g) must be smaller than cull_sigma (%g), or tile culling would change the depth buffer"
                % (self.depth_gate_sigma, self.cull_sigma)
            )
        self.backend = "soft-torch"

    def __repr__(self) -> str:
        return "SoftRasterizer(%r, sigma_px=%.3f, gamma=%.2e, face_chunk=%d, cull_sigma=%s)" % (
            self.camera,
            self.sigma_px,
            self.gamma,
            self.face_chunk,
            self.cull_sigma,
        )

    def __call__(self, verts: torch.Tensor, faces: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward(verts, faces)

    def forward(self, verts: torch.Tensor, faces: torch.Tensor) -> dict[str, torch.Tensor]:
        if verts.ndim == 2:
            verts = verts.unsqueeze(0)
        if verts.ndim != 3 or verts.shape[-1] != 3:
            raise ValueError("verts must be (N, V, 3), got %s" % (tuple(verts.shape),))
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError("faces must be (F, 3), got %s" % (tuple(faces.shape),))
        if faces.numel() and int(faces.max()) >= verts.shape[1]:
            raise ValueError(
                "face index %d exceeds the %d vertices given" % (int(faces.max()), verts.shape[1])
            )
        cam = self.camera
        n = verts.shape[0]
        (h, w) = (cam.height, cam.width)
        (dtype, device) = (verts.dtype, verts.device)
        pix = cam.pixel_grid(dtype=dtype, device=device)
        if self.cull_sigma is None:
            (m, d) = self._render_batch(verts, faces, pix)
            return {"mask": m.reshape(n, h, w), "depth": d.reshape(n, h, w)}
        return self._render_tiled(verts, faces)

    def _render_tiled(self, verts: torch.Tensor, faces: torch.Tensor) -> dict[str, torch.Tensor]:
        cam = self.camera
        (n, h, w, t) = (verts.shape[0], cam.height, cam.width, self.tile)
        (dtype, device) = (verts.dtype, verts.device)
        mask = torch.zeros(n, h, w, dtype=dtype, device=device)
        depth = torch.zeros(n, h, w, dtype=dtype, device=device)
        with torch.no_grad():
            uv_d = cam.project(verts.detach())
            tri_d = uv_d[:, faces]
            lo = tri_d.amin(dim=2)
            hi = tri_d.amax(dim=2)
            z_ok = (verts.detach()[:, :, 2][:, faces] > self.z_near).all(dim=-1)
            pad = self.cull_sigma * self.sigma_px
        for y0 in range(0, h, t):
            y1 = min(y0 + t, h)
            for x0 in range(0, w, t):
                x1 = min(x0 + t, w)
                with torch.no_grad():
                    keep = (
                        z_ok
                        & (hi[..., 0] >= x0 - pad)
                        & (lo[..., 0] <= x1 - 1 + pad)
                        & (hi[..., 1] >= y0 - pad)
                        & (lo[..., 1] <= y1 - 1 + pad)
                    )
                    idx = torch.nonzero(keep.any(dim=0), as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                vv = torch.arange(y0, y1, dtype=dtype, device=device)
                uu = torch.arange(x0, x1, dtype=dtype, device=device)
                (gy, gx) = torch.meshgrid(vv, uu, indexing="ij")
                sub_pix = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
                (m, d) = self._render_batch(verts, faces[idx], sub_pix)
                mask[:, y0:y1, x0:x1] = m.reshape(n, y1 - y0, x1 - x0)
                depth[:, y0:y1, x0:x1] = d.reshape(n, y1 - y0, x1 - x0)
        return {"mask": mask, "depth": depth}

    def _render_batch(
        self, verts: torch.Tensor, faces: torch.Tensor, pix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (n, p) = (verts.shape[0], pix.shape[0])
        (dtype, device) = (verts.dtype, verts.device)
        uv = self.camera.project(verts)
        z = verts[..., 2]
        log_one_minus = torch.zeros(n, p, dtype=dtype, device=device)
        run_max = torch.full((n, p), -float("inf"), dtype=dtype, device=device)
        run_den = torch.zeros(n, p, dtype=dtype, device=device)
        run_num = torch.zeros(n, p, dtype=dtype, device=device)
        for start in range(0, faces.shape[0], self.face_chunk):
            fchunk = faces[start : start + self.face_chunk]
            tri_uv = uv[:, fchunk]
            tri_z = z[:, fchunk]
            visible = (tri_z > self.z_near).all(dim=-1)
            if not bool(visible.any()):
                continue
            sd = _signed_distance_to_triangles(pix, tri_uv)
            prob = torch.sigmoid(sd / self.sigma_px)
            prob = prob * visible.to(dtype).unsqueeze(-2)
            log_one_minus = log_one_minus + torch.log1p(-prob.clamp(max=1.0 - 1e-07)).sum(-1)
            zf = _barycentric_depth(pix, tri_uv, tri_z)
            inv_z = 1.0 / zf.clamp_min(self.z_near)
            near = sd > -(self.depth_gate_sigma * self.sigma_px)
            usable = visible.unsqueeze(-2) & near
            logits = torch.where(usable, inv_z / self.gamma, torch.full_like(inv_z, -float("inf")))
            chunk_max = logits.max(dim=-1).values
            new_max = torch.maximum(run_max, chunk_max)
            new_max_d = torch.where(
                torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
            ).detach()
            rescale = torch.exp(
                torch.where(torch.isfinite(run_max), run_max, torch.zeros_like(run_max)).detach()
                - new_max_d
            )
            rescale = torch.where(torch.isfinite(run_max), rescale, torch.zeros_like(rescale))
            wt = prob * torch.exp(logits - new_max_d.unsqueeze(-1))
            wt = torch.where(usable, wt, torch.zeros_like(wt))
            run_den = run_den * rescale + wt.sum(-1)
            run_num = run_num * rescale + (wt * zf).sum(-1)
            run_max = new_max
        mask = 1.0 - torch.exp(log_one_minus)
        depth = run_num / run_den.clamp_min(_TINY)
        covered = (run_den > _TINY) & (mask > _DEPTH_MASK_FLOOR)
        depth = torch.where(covered, depth, torch.zeros_like(depth))
        return (mask, depth)


class NvdiffrastRasterizer:
    def __init__(self, camera: PinholeCamera, z_near: float = 1e-3, z_far: float = 100.0):
        if not 0 < z_near < z_far:
            raise ValueError("Clipping planes must satisfy 0 < near < far")
        import nvdiffrast.torch as dr

        self.dr = dr
        self.camera = camera
        self.z_near = z_near
        self.z_far = z_far
        self.contexts = {}

    def clip_coordinates(self, verts):
        cam = self.camera
        x, y, z = verts.unbind(-1)
        near, far = self.z_near, self.z_far
        return torch.stack(
            (
                2 * cam.fx / cam.width * x + (2 * (cam.cx + 0.5) / cam.width - 1) * z,
                -2 * cam.fy / cam.height * y + (1 - 2 * (cam.cy + 0.5) / cam.height) * z,
                (far + near) / (far - near) * z - 2 * far * near / (far - near),
                z,
            ),
            dim=-1,
        )

    def __call__(self, verts, faces):
        if not verts.is_cuda or verts.dtype != torch.float32:
            raise ValueError("nvdiffrast requires CUDA float32 vertices")
        if verts.ndim == 2:
            verts = verts.unsqueeze(0)
        if verts.device not in self.contexts:
            self.contexts[verts.device] = self.dr.RasterizeCudaContext(device=verts.device)
        clip = self.clip_coordinates(verts).contiguous()
        triangles = faces.to(device=verts.device, dtype=torch.int32).contiguous()
        rast, _ = self.dr.rasterize(
            self.contexts[verts.device],
            clip,
            triangles,
            resolution=[self.camera.height, self.camera.width],
            grad_db=False,
        )
        coverage = (rast[..., 3:4] > 0).to(verts.dtype)
        mask = self.dr.antialias(coverage, rast, clip, triangles)
        depth, _ = self.dr.interpolate(verts[..., 2:3].contiguous(), rast, triangles)
        return {
            "mask": mask[..., 0].flip(1),
            "depth": (depth * coverage)[..., 0].flip(1),
        }


class Rasterizer:
    def __init__(
        self,
        camera: PinholeCamera,
        sigma_px: float = 1.0,
        gamma: float = 1e-4,
        face_chunk: int = 2048,
        prefer: str = "soft",
        cull_sigma: float | None = 16.0,
        tile: int = 24,
        depth_gate_sigma: float = 3.0,
    ):
        self.camera = camera
        self.backend = prefer
        if prefer == "soft":
            self._impl = SoftRasterizer(
                camera,
                sigma_px=sigma_px,
                gamma=gamma,
                face_chunk=face_chunk,
                cull_sigma=cull_sigma,
                tile=tile,
                depth_gate_sigma=depth_gate_sigma,
            )
        elif prefer == "nvdiffrast":
            self._impl = NvdiffrastRasterizer(camera)
        else:
            raise ValueError("Rasterizer backend must be soft or nvdiffrast")

    @classmethod
    def from_config(cls, camera: PinholeCamera, render_cfg: Any):
        return cls(
            camera,
            sigma_px=render_cfg.sigma_px,
            gamma=render_cfg.gamma,
            face_chunk=render_cfg.face_chunk,
            prefer=render_cfg.backend,
            cull_sigma=render_cfg.cull_sigma,
            tile=render_cfg.tile,
            depth_gate_sigma=render_cfg.depth_gate_sigma,
        )

    def __call__(self, verts, faces):
        return self._impl(verts, faces)


def losses(rend, obs_mask, obs_depth, lambda_depth, lambda_reg, delta_t, epsilon, depth_valid=None):
    mask, depth = rend["mask"], rend["depth"]
    if mask.shape != obs_mask.shape or depth.shape != obs_depth.shape or mask.shape != depth.shape:
        raise ValueError("Rendered and observed mask/depth shapes must match")
    if mask.ndim not in (2, 3):
        raise ValueError("Observations must have shape (H, W) or (N, H, W)")
    l_mask = (mask - obs_mask.to(mask)).square().mean()
    valid = torch.isfinite(obs_depth) & (obs_depth > 0)
    if depth_valid is not None:
        valid = valid & depth_valid.to(dtype=torch.bool, device=mask.device)
    omega = (mask > epsilon) & (obs_mask > 0.5) & valid
    safe_obs = torch.where(valid, obs_depth, torch.zeros_like(obs_depth))
    residual = torch.where(omega, (depth - safe_obs).abs(), torch.zeros_like(depth))
    counts = omega.sum(dim=(-2, -1))
    l_depth = (residual.sum(dim=(-2, -1)) / counts.clamp_min(1)).mean()
    dt = torch.as_tensor(delta_t, dtype=mask.dtype, device=mask.device)
    l_reg = lambda_reg * dt.square()
    return {
        "total": l_mask + lambda_depth * l_depth + l_reg,
        "mask": l_mask,
        "depth": l_depth,
        "reg": l_reg,
        "n_omega": counts.sum(),
    }


def iou(rend_mask: torch.Tensor, obs_mask: torch.Tensor, epsilon: float = 0.5) -> torch.Tensor:
    r = rend_mask > epsilon
    o = obs_mask > 0.5
    inter = (r & o).flatten(-2).sum(-1).to(rend_mask.dtype)
    union = (r | o).flatten(-2).sum(-1).to(rend_mask.dtype)
    return inter / union.clamp_min(1.0)
