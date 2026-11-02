from __future__ import annotations

import torch


def skew(v):
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            torch.stack((zero, -v[..., 2], v[..., 1]), dim=-1),
            torch.stack((v[..., 2], zero, -v[..., 0]), dim=-1),
            torch.stack((-v[..., 1], v[..., 0], zero), dim=-1),
        ),
        dim=-2,
    )


def _coefficients(phi):
    theta2 = phi.square().sum(-1)[..., None, None]
    theta = theta2.clamp_min(1e-8).sqrt()
    small = theta2 < 1e-4
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / theta)
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        2 * torch.sin(theta / 2).square() / theta.square(),
    )
    c = torch.where(
        small,
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (theta - torch.sin(theta)) / theta.pow(3),
    )
    return theta2, theta, small, a, b, c


def so3_exp(phi):
    _, _, _, a, b, _ = _coefficients(phi)
    K = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device)
    return eye + a * K + b * (K @ K)


def so3_log(R):
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    vee = torch.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        dim=-1,
    )
    sin = vee.norm(dim=-1) / 2
    angle = torch.atan2(sin, cos)
    scale = torch.where(angle < 1e-4, 0.5 + angle.square() / 12, angle / (2 * sin.clamp_min(1e-12)))
    ordinary = scale[..., None] * vee
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    axis_outer = (R + R.transpose(-1, -2) - 2 * cos[..., None, None] * eye) / (
        2 * (1 - cos).clamp_min(1e-12)[..., None, None]
    )
    diagonal = axis_outer.diagonal(dim1=-2, dim2=-1).clamp_min(0)
    index = diagonal.argmax(-1)
    column = axis_outer.gather(-1, index[..., None, None].expand(*index.shape, 3, 1)).squeeze(-1)
    divisor = diagonal.gather(-1, index[..., None]).clamp_min(1e-12).sqrt()
    axis = column / divisor
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sign = torch.where((axis * vee).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return torch.where((cos < -0.9999)[..., None], axis * sign * angle[..., None], ordinary)


def left_jacobian(phi):
    _, _, _, _, b, c = _coefficients(phi)
    K = skew(phi)
    return torch.eye(3, dtype=phi.dtype, device=phi.device) + b * K + c * (K @ K)


def left_jacobian_inv(phi):
    theta2, theta, small, _, _, _ = _coefficients(phi)
    d = torch.where(
        small,
        1 / 12 + theta2 / 720 + theta2.square() / 30240,
        (1 - theta / (2 * torch.tan(theta / 2))) / theta.square(),
    )
    K = skew(phi)
    return torch.eye(3, dtype=phi.dtype, device=phi.device) - 0.5 * K + d * (K @ K)


def se3_exp(phi, rho):
    return compose(so3_exp(phi), (left_jacobian(phi) @ rho.unsqueeze(-1)).squeeze(-1))


def se3_log(T):
    phi = so3_log(T[..., :3, :3])
    rho = (left_jacobian_inv(phi) @ T[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    return phi, rho


def compose(R, t):
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def inverse(T):
    Rt = T[..., :3, :3].transpose(-1, -2)
    return compose(Rt, -(Rt @ T[..., :3, 3].unsqueeze(-1)).squeeze(-1))


def transform_points(T, p):
    return (T[..., :3, :3] @ p.transpose(-1, -2)).transpose(-1, -2) + T[..., :3, 3].unsqueeze(-2)


def geodesic_angle(Ra, Rb):
    R = Ra.transpose(-1, -2) @ Rb
    vee = torch.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        dim=-1,
    )
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.atan2(vee.norm(dim=-1) / 2, cos)
