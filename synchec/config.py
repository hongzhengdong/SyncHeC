from __future__ import annotations
import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LossConfig:
    lambda_depth: float = 0.1
    lambda_reg: float = 0.5
    lambda_init: float = 1.0
    epsilon: float = 0.5


@dataclass
class OptimConfig:
    eta_rot: float = 0.001
    eta_trans: float = 0.0005
    eta_t: float = 0.0002
    k_max: int = 15
    passes_per_block: int = 15
    batch_size: int = 4
    early_stopping: bool = True
    tol_trans_mm: float = 0.05
    tol_rot_deg: float = 0.01
    tol_dt_ms: float = 0.5
    tau_vel: float = 0.05


@dataclass
class DataConfig:
    n_keep: int = 60
    camera_fps: float = 30.0
    joint_rate_hz: float = 250.0
    width: int = 848
    height: int = 480
    depth_scale: float = 0.001
    mount: str = "eth"


@dataclass
class PlannerConfig:
    n_knots: int = 25
    duration_s: float = 6.5
    alpha: float = 5e-4
    beta: float = 2.5e-3
    n_candidate_cameras: int = 8
    opt_steps: int = 100
    lr: float = 0.01


@dataclass
class RenderConfig:
    backend: str = "soft"
    sigma_px: float = 1.0
    gamma: float = 0.0001
    face_chunk: int = 2048
    cull_sigma: float | None = 16.0
    tile: int = 24
    depth_gate_sigma: float = 3.0


@dataclass
class StageOneConfig:
    n_sphere_samples: int = 200
    delta_r_m: float = 0.1
    base_cylinder_radius_m: float = 0.06
    camera_height_m: float = 0.0
    refine_steps: int = 200


@dataclass
class Config:
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    stage_one: StageOneConfig = field(default_factory=StageOneConfig)
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    def derived(self) -> dict[str, float]:
        return {
            "tol_trans_m": self.optim.tol_trans_mm * 0.001,
            "tol_rot_rad": self.optim.tol_rot_deg * 3.141592653589793 / 180.0,
            "tol_dt_s": self.optim.tol_dt_ms * 0.001,
        }

    def updates_per_block(self, n_frames: int) -> int:
        if n_frames < 0:
            raise ValueError("n_frames must be nonnegative")
        return self.optim.passes_per_block * math.ceil(n_frames / self.optim.batch_size)

    def validate(self) -> None:
        for name in ("k_max", "passes_per_block", "batch_size"):
            value = getattr(self.optim, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"optim.{name} must be a positive integer")
        for group_name in ("loss", "optim", "data", "planner", "render", "stage_one"):
            group = getattr(self, group_name)
            for f in dataclasses.fields(group):
                value = getattr(group, f.name)
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"{group_name}.{f.name} must be finite")
        if min(self.optim.eta_rot, self.optim.eta_trans, self.optim.eta_t) <= 0:
            raise ValueError("Learning rates must be positive")
        if min(self.loss.lambda_depth, self.loss.lambda_reg, self.loss.lambda_init) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if not 0 < self.loss.epsilon < 1:
            raise ValueError("loss.epsilon must lie in (0, 1)")
        if self.optim.tau_vel < 0:
            raise ValueError("optim.tau_vel must be nonnegative")
        if min(self.optim.tol_trans_mm, self.optim.tol_rot_deg, self.optim.tol_dt_ms) < 0:
            raise ValueError("Stopping tolerances must be nonnegative")
        if self.data.n_keep < 1 or min(self.data.width, self.data.height) < 1:
            raise ValueError("Frame count and image dimensions must be positive")
        if self.data.mount not in ("eth", "eih"):
            raise ValueError("data.mount must be eth or eih")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.render.backend not in ("soft", "nvdiffrast"):
            raise ValueError("render.backend must be soft or nvdiffrast")
        if self.render.backend == "nvdiffrast" and (
            not self.device.startswith("cuda") or self.dtype != "float32"
        ):
            raise ValueError("nvdiffrast requires a CUDA device and float32")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        groups = {
            "loss": LossConfig,
            "optim": OptimConfig,
            "data": DataConfig,
            "planner": PlannerConfig,
            "render": RenderConfig,
            "stage_one": StageOneConfig,
        }
        kw: dict[str, Any] = {}
        for key, klass in groups.items():
            sub = dict(d.get(key, {}))
            known = {f.name for f in dataclasses.fields(klass)}
            unknown = set(sub) - known
            if unknown:
                raise ValueError("unknown %s keys: %s" % (key, sorted(unknown)))
            kw[key] = klass(**sub)
        for scalar in ("seed", "device", "dtype"):
            if scalar in d:
                kw[scalar] = d[scalar]
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
