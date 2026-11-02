from __future__ import annotations
import torch
from . import lie


class JointLog:
    def __init__(self, stamps: torch.Tensor, q: torch.Tensor):
        if stamps.ndim != 1:
            raise ValueError("stamps must be 1-D, got shape %s" % (tuple(stamps.shape),))
        if q.ndim != 2 or q.shape[0] != stamps.shape[0]:
            raise ValueError(
                "q must be (M, D) matching stamps (M=%d), got %s"
                % (stamps.shape[0], tuple(q.shape))
            )
        if stamps.shape[0] < 2:
            raise ValueError("need at least 2 log samples to interpolate")
        if not torch.isfinite(stamps).all() or not torch.isfinite(q).all():
            raise ValueError("Joint timestamps and positions must be finite")
        if stamps.device != q.device:
            raise ValueError("Joint timestamps and positions must share a device")
        d = stamps[1:] - stamps[:-1]
        if not bool((d > 0).all()):
            raise ValueError("stamps must be strictly increasing")
        self.stamps = stamps.to(torch.float64)
        self.q = q
        self.dof = q.shape[1]

    def __len__(self) -> int:
        return self.stamps.shape[0]

    @property
    def duration(self) -> float:
        return float(self.stamps[-1] - self.stamps[0])

    def velocity(self) -> torch.Tensor:
        q = self.q.to(torch.float64)
        t = self.stamps
        v = torch.zeros_like(q)
        v[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2]).unsqueeze(-1)
        v[0] = (q[1] - q[0]) / (t[1] - t[0])
        v[-1] = (q[-1] - q[-2]) / (t[-1] - t[-2])
        return v.to(self.q.dtype)

    def speed_at(self, query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.stamps, query.to(torch.float64).contiguous(), right=True) - 1
        idx = idx.clamp(0, len(self) - 2)
        slope = (self.q[idx + 1] - self.q[idx]) / (self.stamps[idx + 1] - self.stamps[idx]).to(
            self.q.dtype
        ).unsqueeze(-1)
        return slope.norm(dim=-1)


def q_sync(log: JointLog, cam_stamps: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    tau = cam_stamps.to(torch.float64) - delta_t.to(torch.float64)
    with torch.no_grad():
        k = torch.searchsorted(log.stamps, tau.detach().contiguous(), right=True) - 1
        k = k.clamp(0, len(log) - 2)
    t0 = log.stamps[k]
    t1 = log.stamps[k + 1]
    q0 = log.q[k]
    q1 = log.q[k + 1]
    w = ((tau - t0) / (t1 - t0)).to(log.q.dtype).unsqueeze(-1)
    return q0 + w * (q1 - q0)


def temporal_jacobian(
    log: JointLog, cam_stamps: torch.Tensor, delta_t: torch.Tensor
) -> torch.Tensor:
    tau = cam_stamps.to(torch.float64) - delta_t.to(torch.float64)
    k = (torch.searchsorted(log.stamps, tau.detach().contiguous(), right=True) - 1).clamp(
        0, len(log) - 2
    )
    dt = (log.stamps[k + 1] - log.stamps[k]).to(log.q.dtype).unsqueeze(-1)
    return -(log.q[k + 1] - log.q[k]) / dt


def check_coverage(
    log: JointLog, cam_stamps: torch.Tensor, delta_t_range: tuple[float, float]
) -> dict:
    (lo, hi) = delta_t_range
    tau_lo = cam_stamps.to(torch.float64) - hi
    tau_hi = cam_stamps.to(torch.float64) - lo
    below = int((tau_lo < log.stamps[0]).sum())
    above = int((tau_hi > log.stamps[-1]).sum())
    return {
        "n_frames": int(cam_stamps.numel()),
        "extrapolate_before": below,
        "extrapolate_after": above,
        "log_span_s": (float(log.stamps[0]), float(log.stamps[-1])),
    }


class LinkPoseModel:
    def __init__(self, fk, mount: str = "eth"):
        mount = mount.lower()
        if mount not in ("eth", "eih"):
            raise ValueError("mount must be 'eth' or 'eih', got %r" % (mount,))
        self.fk = fk
        self.mount = mount

    def __call__(self, T_extrinsic: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        link_T = self.fk(q)
        if self.mount == "eth":
            return T_extrinsic @ link_T
        flange = self.fk.flange(q)
        return T_extrinsic @ lie.inverse(flange).unsqueeze(1) @ link_T


def extrinsic_from_params(phi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return lie.se3_exp(phi, rho)
