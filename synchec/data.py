from __future__ import annotations
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import yaml
from .kinematics import JointLog

FORMAT_VERSION = 1
META_NAME = "meta.yaml"
JOINTS_NAME = "joints.csv"
FRAMES_NAME = "frames.csv"
GT_NAME = "ground_truth.yaml"
RGB_DIR = "rgb"
DEPTH_DIR = "depth"
MASK_DIR = "mask"
JOINTS_HEADER_0 = "t_r_seconds"
FRAMES_HEADER = ("index", "t_c_seconds", "rgb", "depth", "mask")


class DatasetFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_any(cls, src: "Intrinsics | Mapping[str, Any] | Any") -> "Intrinsics":
        if isinstance(src, Intrinsics):
            return src
        keys = ("fx", "fy", "cx", "cy", "width", "height")
        if isinstance(src, Mapping):
            missing = [k for k in keys if k not in src]
            if missing:
                raise DatasetFormatError(
                    "camera intrinsics missing key(s) %s; need all of %s" % (missing, list(keys))
                )
            vals = {k: src[k] for k in keys}
        else:
            missing = [k for k in keys if not hasattr(src, k)]
            if missing:
                raise DatasetFormatError(
                    "object of type %s is not usable as intrinsics: missing %s"
                    % (type(src).__name__, missing)
                )
            vals = {k: getattr(src, k) for k in keys}
        return cls(
            fx=float(vals["fx"]),
            fy=float(vals["fy"]),
            cx=float(vals["cx"]),
            cy=float(vals["cy"]),
            width=int(vals["width"]),
            height=int(vals["height"]),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def _imread(path: Path, unchanged: bool = True) -> np.ndarray:
    import cv2

    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise DatasetFormatError(
            "could not decode image %r (missing, empty, or not a PNG)" % (str(path),)
        )
    return img


def _imwrite(path: Path, img: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError("cv2 failed to write %r" % (str(path),))


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise DatasetFormatError("required file %r is missing" % (path.name,))
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and (not r[0].lstrip().startswith("#"))]
    if not rows:
        raise DatasetFormatError("%r is empty; expected a header row" % (path.name,))
    return ([c.strip() for c in rows[0]], rows[1:])


def _strictly_increasing(x: np.ndarray, what: str, path: str) -> None:
    if x.size < 2:
        return
    d = np.diff(x)
    bad = np.nonzero(d <= 0)[0]
    if bad.size:
        i = int(bad[0])
        raise DatasetFormatError(
            "%s in %s must be strictly increasing, but row %d (%r) is not greater than row %d (%r); %d violation(s) total"
            % (what, path, i + 1, float(x[i + 1]), i, float(x[i]), bad.size)
        )


def _opt_float(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if v > 0 else None


def _float(cell: str, path: str, row: int, col: str) -> float:
    try:
        return float(cell)
    except ValueError:
        raise DatasetFormatError(
            "%s row %d column %r: %r is not a number" % (path, row, col, cell)
        ) from None


class CalibDataset:
    def __init__(
        self,
        path: str | Path,
        eager: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        self.root = Path(path).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetFormatError("recording directory %r does not exist" % (str(path),))
        self.dtype = dtype
        self.device = torch.device(device)
        self.meta = self._read_meta()
        self.intrinsics = Intrinsics.from_any(self.meta["camera"])
        self.mount = str(self.meta.get("mount", "eth")).lower()
        if self.mount not in ("eth", "eih"):
            raise DatasetFormatError(
                "meta.yaml 'mount' must be 'eth' or 'eih', got %r" % (self.mount,)
            )
        self.fps = _opt_float(self.meta.get("fps", None))
        self.joint_rate_hz = _opt_float(self.meta.get("joint_rate_hz", None))
        scale = self.meta.get("depth_scale", None)
        self.depth_scale = 0.001 if scale is None else float(scale)
        if not self.depth_scale > 0:
            raise DatasetFormatError(
                "meta.yaml 'depth_scale' must be positive, got %r" % (self.depth_scale,)
            )
        trunc = self.meta.get("depth_trunc_m", None)
        self.depth_trunc_m = float(trunc) if trunc is not None else None
        urdf = self.meta.get("urdf", None)
        self.urdf_path = self.root / str(urdf) if urdf else None
        if self.urdf_path is not None and (not self.urdf_path.exists()):
            raise DatasetFormatError(
                "meta.yaml points at urdf %r but %r does not exist"
                % (str(urdf), str(self.urdf_path))
            )
        self.joint_names: list[str] | None = self.meta.get("joint_names", None)
        self.joint_log = self._read_joints()
        (stamps, rgb_p, depth_p, mask_p) = self._read_frames()
        self.cam_stamps = stamps
        self.rgb_paths = rgb_p
        self.depth_paths = depth_p
        self.mask_paths = mask_p
        declared_dof = self.meta.get("dof", None)
        if declared_dof is not None and int(declared_dof) != self.joint_log.dof:
            raise DatasetFormatError(
                "meta.yaml declares dof=%d but joints.csv has %d joint columns"
                % (int(declared_dof), self.joint_log.dof)
            )
        if self.joint_names is not None and len(self.joint_names) != self.joint_log.dof:
            raise DatasetFormatError(
                "meta.yaml lists %d joint_names but joints.csv has %d joint columns"
                % (len(self.joint_names), self.joint_log.dof)
            )
        self._masks: torch.Tensor | None = None
        self._depths: torch.Tensor | None = None
        self._depth_valid: torch.Tensor | None = None
        self._camera = None
        self._robot = None
        if eager:
            self.load_images()

    @classmethod
    def from_arrays(
        cls,
        *,
        intrinsics: Intrinsics,
        joint_log: JointLog,
        cam_stamps: torch.Tensor,
        masks: torch.Tensor | None,
        depths: torch.Tensor | None,
        depth_valid: torch.Tensor | None,
        mount: str = "eth",
        root: Path | None = None,
        meta: dict[str, Any] | None = None,
        rgb_paths: Sequence[Path | None] | None = None,
        depth_paths: Sequence[Path] | None = None,
        mask_paths: Sequence[Path] | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "CalibDataset":
        obj = cls.__new__(cls)
        obj.root = root
        obj.dtype = dtype
        obj.device = torch.device(device)
        obj.meta = dict(meta or {})
        obj.intrinsics = intrinsics
        obj.mount = mount.lower()
        obj.fps = obj.meta.get("fps", None)
        obj.joint_rate_hz = obj.meta.get("joint_rate_hz", None)
        obj.depth_scale = float(obj.meta.get("depth_scale", 0.001))
        obj.depth_trunc_m = obj.meta.get("depth_trunc_m", None)
        obj.urdf_path = None
        obj.joint_names = obj.meta.get("joint_names", None)
        obj.joint_log = joint_log
        obj.cam_stamps = cam_stamps.to(torch.float64)
        n = int(obj.cam_stamps.numel())
        obj.rgb_paths = list(rgb_paths) if rgb_paths is not None else [None] * n
        obj.depth_paths = list(depth_paths) if depth_paths is not None else []
        obj.mask_paths = list(mask_paths) if mask_paths is not None else []
        obj._masks = masks
        obj._depths = depths
        obj._depth_valid = depth_valid
        obj._camera = None
        obj._robot = None
        return obj

    def _read_meta(self) -> dict[str, Any]:
        p = self.root / META_NAME
        if not p.exists():
            raise DatasetFormatError(
                "%r is not a recording: %s is missing" % (str(self.root), META_NAME)
            )
        with open(p, encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
        if not isinstance(meta, dict):
            raise DatasetFormatError("%s must contain a YAML mapping" % (META_NAME,))
        ver = int(meta.get("format_version", FORMAT_VERSION))
        if ver != FORMAT_VERSION:
            raise DatasetFormatError(
                "%s declares format_version %d but this loader implements %d"
                % (META_NAME, ver, FORMAT_VERSION)
            )
        if "camera" not in meta:
            raise DatasetFormatError("%s has no 'camera' section" % (META_NAME,))
        return meta

    def _read_joints(self) -> JointLog:
        p = self.root / JOINTS_NAME
        (header, rows) = _read_csv(p)
        if header[0] != JOINTS_HEADER_0:
            raise DatasetFormatError(
                "%s: first column must be %r, got %r" % (JOINTS_NAME, JOINTS_HEADER_0, header[0])
            )
        names = header[1:]
        if len(set(names)) != len(names):
            raise DatasetFormatError("Joint column names must be unique")
        if self.joint_names is not None and list(self.joint_names) != names:
            raise DatasetFormatError("meta.yaml joint_names must match the CSV column order")
        self.joint_names = names
        dof = len(header) - 1
        if dof < 1:
            raise DatasetFormatError(
                "%s: expected at least one joint column after %r" % (JOINTS_NAME, JOINTS_HEADER_0)
            )
        if len(rows) < 2:
            raise DatasetFormatError(
                "%s: need at least 2 samples to interpolate, got %d" % (JOINTS_NAME, len(rows))
            )
        t = np.empty(len(rows), dtype=np.float64)
        q = np.empty((len(rows), dof), dtype=np.float64)
        for i, row in enumerate(rows):
            if len(row) != dof + 1:
                raise DatasetFormatError(
                    "%s row %d has %d fields but the header declares %d"
                    % (JOINTS_NAME, i, len(row), dof + 1)
                )
            t[i] = _float(row[0], JOINTS_NAME, i, JOINTS_HEADER_0)
            for j in range(dof):
                q[i, j] = _float(row[j + 1], JOINTS_NAME, i, header[j + 1])
        _strictly_increasing(t, "t_r_seconds", JOINTS_NAME)
        return JointLog(
            torch.from_numpy(t).to(self.device),
            torch.from_numpy(q).to(dtype=self.dtype, device=self.device),
        )

    def _read_frames(self) -> tuple[torch.Tensor, list[Path | None], list[Path], list[Path]]:
        p = self.root / FRAMES_NAME
        (header, rows) = _read_csv(p)
        if tuple(header) != FRAMES_HEADER:
            raise DatasetFormatError(
                "%s: header must be exactly %s, got %s" % (FRAMES_NAME, list(FRAMES_HEADER), header)
            )
        if not rows:
            raise DatasetFormatError("%s has a header but no frames" % (FRAMES_NAME,))
        idx = np.empty(len(rows), dtype=np.int64)
        t = np.empty(len(rows), dtype=np.float64)
        rgb: list[Path | None] = []
        depth: list[Path] = []
        mask: list[Path] = []
        for i, row in enumerate(rows):
            if len(row) != len(FRAMES_HEADER):
                raise DatasetFormatError(
                    "%s row %d has %d fields but the header declares %d"
                    % (FRAMES_NAME, i, len(row), len(FRAMES_HEADER))
                )
            try:
                idx[i] = int(row[0])
            except ValueError:
                raise DatasetFormatError(
                    "%s row %d: index %r is not an integer" % (FRAMES_NAME, i, row[0])
                ) from None
            t[i] = _float(row[1], FRAMES_NAME, i, "t_c_seconds")
            (rgb_cell, depth_cell, mask_cell) = (c.strip() for c in row[2:5])
            rgb.append(self.root / rgb_cell if rgb_cell else None)
            if not depth_cell or not mask_cell:
                raise DatasetFormatError(
                    "%s row %d: the depth and mask cells are required (rgb may be empty)"
                    % (FRAMES_NAME, i)
                )
            depth.append(self.root / depth_cell)
            mask.append(self.root / mask_cell)
        _strictly_increasing(idx.astype(np.float64), "index", FRAMES_NAME)
        _strictly_increasing(t, "t_c_seconds", FRAMES_NAME)
        missing = [
            str(q.relative_to(self.root)) for q in list(depth) + list(mask) if not q.exists()
        ]
        missing += [
            str(q.relative_to(self.root)) for q in rgb if q is not None and (not q.exists())
        ]
        if missing:
            raise DatasetFormatError(
                "%s references %d file(s) that do not exist: %s%s"
                % (FRAMES_NAME, len(missing), missing[:8], " ..." if len(missing) > 8 else "")
            )
        return (torch.from_numpy(t).to(self.device), rgb, depth, mask)

    @property
    def loaded(self) -> bool:
        return self._masks is not None

    def load_images(self) -> None:
        if self.loaded:
            return
        (h, w) = self.intrinsics.shape
        n = self.n_frames
        masks = np.empty((n, h, w), dtype=np.float32)
        depths = np.empty((n, h, w), dtype=np.float32)
        valid = np.empty((n, h, w), dtype=bool)
        for i in range(n):
            (m, d, v) = self._read_frame_arrays(i)
            (masks[i], depths[i], valid[i]) = (m, d, v)
        self._masks = torch.from_numpy(masks).to(dtype=self.dtype, device=self.device)
        self._depths = torch.from_numpy(depths).to(dtype=self.dtype, device=self.device)
        self._depth_valid = torch.from_numpy(valid).to(self.device)

    def _read_frame_arrays(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        (h, w) = self.intrinsics.shape
        raw_m = _imread(self.mask_paths[i])
        if raw_m.ndim == 3:
            raw_m = raw_m[..., 0]
        self._check_shape(raw_m, "mask", self.mask_paths[i], h, w)
        if np.issubdtype(raw_m.dtype, np.integer):
            mask = raw_m.astype(np.float32) / np.float32(np.iinfo(raw_m.dtype).max)
        else:
            mask = raw_m.astype(np.float32)
        raw_d = _imread(self.depth_paths[i])
        if raw_d.ndim == 3:
            raw_d = raw_d[..., 0]
        self._check_shape(raw_d, "depth", self.depth_paths[i], h, w)
        if raw_d.dtype != np.uint16:
            warnings.warn(
                "depth image %r has dtype %s; the documented format is 16-bit unsigned millimetres"
                % (self.depth_paths[i].name, raw_d.dtype),
                stacklevel=3,
            )
        valid = raw_d != 0
        depth64 = raw_d.astype(np.float64) * float(self.depth_scale)
        if self.depth_trunc_m is not None:
            valid &= depth64 <= float(self.depth_trunc_m)
        valid &= np.isfinite(depth64)
        depth = np.where(valid, depth64, 0.0).astype(np.float32)
        return (mask, depth, valid)

    def _check_shape(self, img: np.ndarray, what: str, path: Path, h: int, w: int) -> None:
        if img.shape[:2] != (h, w):
            raise DatasetFormatError(
                "%s image %r is %dx%d (HxW) but meta.yaml's intrinsics declare %dx%d; the intrinsics and the images must describe the same sensor or every projection is wrong"
                % (what, path.name, img.shape[0], img.shape[1], h, w)
            )

    def frame(self, i: int) -> dict[str, torch.Tensor]:
        if self.loaded:
            return {
                "mask": self._masks[i],
                "depth": self._depths[i],
                "depth_valid": self._depth_valid[i],
                "t_c": self.cam_stamps[i],
            }
        (m, d, v) = self._read_frame_arrays(i)
        return {
            "mask": torch.from_numpy(m).to(dtype=self.dtype, device=self.device),
            "depth": torch.from_numpy(d).to(dtype=self.dtype, device=self.device),
            "depth_valid": torch.from_numpy(v).to(self.device),
            "t_c": self.cam_stamps[i],
        }

    @property
    def masks(self) -> torch.Tensor:
        self.load_images()
        assert self._masks is not None
        return self._masks

    @property
    def depths(self) -> torch.Tensor:
        self.load_images()
        assert self._depths is not None
        return self._depths

    @property
    def depth_valid(self) -> torch.Tensor:
        self.load_images()
        assert self._depth_valid is not None
        return self._depth_valid

    @property
    def camera(self):
        if self._camera is None:
            from .render import PinholeCamera

            k = self.intrinsics
            self._camera = PinholeCamera(k.fx, k.fy, k.cx, k.cy, k.width, k.height)
        return self._camera

    @property
    def robot(self):
        if self._robot is None:
            from .fk import Robot

            if self.urdf_path is not None:
                self._robot = Robot.from_urdf(
                    str(self.urdf_path), dtype=self.dtype, mesh_dir=self.meta.get("mesh_dir")
                )
                flange = self.meta.get("flange_link")
                if flange is not None:
                    if flange not in self._robot._order:
                        raise DatasetFormatError(f"Unknown flange_link: {flange}")
                    self._robot.flange_link = flange
            elif self.meta.get("robot_model") == "box_links":
                self._robot = Robot.from_primitives(n_links=self.dof, dtype=self.dtype)
            else:
                raise DatasetFormatError("meta.yaml must specify a URDF for this recording")
            expected = [j.name for j in self._robot.joints if j.q_index is not None]
            names = self.joint_names
            if names is not None and set(names) == set(expected):
                if names != expected:
                    indices = [names.index(name) for name in expected]
                    self.joint_log = JointLog(self.joint_log.stamps, self.joint_log.q[:, indices])
                    self.joint_names = expected
            elif names is not None and names != [f"q{i}" for i in range(self.dof)]:
                raise DatasetFormatError(
                    "Joint names must match the URDF or use q0, q1, ... in URDF order"
                )
            if self._robot.dof != self.dof:
                raise DatasetFormatError(
                    "the robot model has %d DoF but the joint log has %d columns; forward kinematics would pair q with the wrong joints"
                    % (self._robot.dof, self.dof)
                )
        return self._robot

    @property
    def n_frames(self) -> int:
        return int(self.cam_stamps.numel())

    @property
    def dof(self) -> int:
        return self.joint_log.dof

    def __len__(self) -> int:
        return self.n_frames

    def __repr__(self) -> str:
        k = self.intrinsics
        return "CalibDataset(n_frames=%d, dof=%d, %dx%d, mount=%r, loaded=%s)" % (
            self.n_frames,
            self.dof,
            k.width,
            k.height,
            self.mount,
            self.loaded,
        )

    def invalid_depth_fraction(self) -> float:
        v = self.depth_valid
        return float(1.0 - v.to(torch.float64).mean())

    def subset(self, indices: torch.Tensor | Sequence[int]) -> "CalibDataset":
        idx = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.n_frames):
            raise IndexError(
                "frame indices must lie in [0, %d); got [%d, %d]"
                % (self.n_frames, int(idx.min()), int(idx.max()))
            )
        take = idx.tolist()
        sub = CalibDataset.from_arrays(
            intrinsics=self.intrinsics,
            joint_log=self.joint_log,
            cam_stamps=self.cam_stamps[idx],
            masks=self._masks[idx] if self._masks is not None else None,
            depths=self._depths[idx] if self._depths is not None else None,
            depth_valid=self._depth_valid[idx] if self._depth_valid is not None else None,
            mount=self.mount,
            root=self.root,
            meta=self.meta,
            rgb_paths=[self.rgb_paths[i] for i in take],
            depth_paths=[self.depth_paths[i] for i in take] if self.depth_paths else None,
            mask_paths=[self.mask_paths[i] for i in take] if self.mask_paths else None,
            dtype=self.dtype,
            device=self.device,
        )
        sub.urdf_path = self.urdf_path
        sub._robot = self._robot
        sub._camera = self._camera
        return sub


def high_flow_scores(dataset: CalibDataset, delta_t: float = 0.0) -> torch.Tensor:
    import cv2

    frames = []
    use_rgb = (
        all(p is not None for p in dataset.rgb_paths) and len(dataset.rgb_paths) == dataset.n_frames
    )
    for i in range(dataset.n_frames):
        if use_rgb:
            image = _imread(dataset.rgb_paths[i], unchanged=False)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(
                (dataset.frame(i)["mask"].detach().cpu().numpy() * 255).round().astype(np.uint8)
            )
    scores = np.zeros(dataset.n_frames, dtype=np.float64)
    counts = np.zeros(dataset.n_frames, dtype=np.float64)
    for i in range(1, dataset.n_frames):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i - 1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        support = (
            ((dataset.frame(i - 1)["mask"] > 0.5) | (dataset.frame(i)["mask"] > 0.5))
            .detach()
            .cpu()
            .numpy()
        )
        interval = float(dataset.cam_stamps[i] - dataset.cam_stamps[i - 1])
        score = np.linalg.norm(flow, axis=-1)[support].mean() / interval if support.any() else 0.0
        scores[i - 1 : i + 1] += score
        counts[i - 1 : i + 1] += 1
    return torch.as_tensor(
        scores / np.maximum(counts, 1), dtype=dataset.dtype, device=dataset.device
    )


def select_high_flow_frames(dataset, robot=None, n_keep=60, tau_vel=0.05, delta_t=0.0):
    if n_keep < 1:
        raise ValueError("n_keep must be positive")
    if robot is not None and robot.dof != dataset.dof:
        raise ValueError("Robot and joint log dimensions differ")
    scores = high_flow_scores(dataset)
    count = min(n_keep, dataset.n_frames)
    return torch.sort(torch.argsort(scores, descending=True, stable=True)[:count]).values


def write_dataset(
    path: str | Path,
    *,
    joint_stamps: Any,
    joint_q: Any,
    cam_stamps: Any,
    masks: Any,
    depths: Any,
    intrinsics: Intrinsics | Mapping[str, Any] | Any,
    mount: str = "eth",
    fps: float | None = None,
    joint_rate_hz: float | None = None,
    depth_scale: float = 0.001,
    depth_trunc_m: float | None = None,
    rgb: Any = None,
    urdf: str | None = None,
    joint_names: Sequence[str] | None = None,
    ground_truth: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / META_NAME).exists() and (not overwrite):
        raise FileExistsError(
            "%r already contains a %s; pass overwrite=True to replace it" % (str(root), META_NAME)
        )
    root.mkdir(parents=True, exist_ok=True)
    k = Intrinsics.from_any(intrinsics)
    t_r = np.asarray(_to_numpy(joint_stamps), dtype=np.float64).reshape(-1)
    q = np.asarray(_to_numpy(joint_q), dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != t_r.size:
        raise ValueError(
            "joint_q must be (M, D) with M = len(joint_stamps) = %d, got %s"
            % (t_r.size, (q.shape,))
        )
    _strictly_increasing(t_r, "joint_stamps", "write_dataset")
    t_c = np.asarray(_to_numpy(cam_stamps), dtype=np.float64).reshape(-1)
    _strictly_increasing(t_c, "cam_stamps", "write_dataset")
    m = np.asarray(_to_numpy(masks), dtype=np.float64)
    d = np.asarray(_to_numpy(depths), dtype=np.float64)
    for name, arr in (("masks", m), ("depths", d)):
        if arr.ndim != 3 or arr.shape[0] != t_c.size:
            raise ValueError(
                "%s must be (N, H, W) with N = len(cam_stamps) = %d, got %s"
                % (name, t_c.size, (arr.shape,))
            )
        if arr.shape[1:] != (k.height, k.width):
            raise ValueError(
                "%s is %dx%d (HxW) but the intrinsics declare %dx%d"
                % (name, arr.shape[1], arr.shape[2], k.height, k.width)
            )
    rgb_arr = None
    if rgb is not None:
        rgb_arr = np.asarray(_to_numpy(rgb))
        if rgb_arr.shape[:3] != (t_c.size, k.height, k.width):
            raise ValueError(
                "rgb must be (N, H, W, 3) matching the masks, got %s" % (rgb_arr.shape,)
            )
    for sub in (MASK_DIR, DEPTH_DIR) + ((RGB_DIR,) if rgb_arr is not None else ()):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, float, str, str, str]] = []
    for i in range(t_c.size):
        stem = "%06d.png" % (i,)
        mask_rel = "%s/%s" % (MASK_DIR, stem)
        depth_rel = "%s/%s" % (DEPTH_DIR, stem)
        rgb_rel = "%s/%s" % (RGB_DIR, stem) if rgb_arr is not None else ""
        mi = np.clip(m[i], 0.0, 1.0)
        _imwrite(root / mask_rel, np.rint(mi * 255.0).astype(np.uint8))
        di = d[i]
        good = np.isfinite(di) & (di > 0.0)
        raw = np.zeros(di.shape, dtype=np.uint16)
        ticks = np.rint(np.where(good, di, 0.0) / depth_scale)
        over = good & (ticks > np.iinfo(np.uint16).max)
        if over.any():
            raise ValueError(
                "%d depth value(s) exceed the 16-bit range at depth_scale=%g (max representable %.3f m); raise depth_scale"
                % (int(over.sum()), depth_scale, np.iinfo(np.uint16).max * depth_scale)
            )
        raw[good] = ticks[good].astype(np.uint16)
        _imwrite(root / depth_rel, raw)
        if rgb_arr is not None:
            _imwrite(root / rgb_rel, np.asarray(rgb_arr[i], dtype=np.uint8))
        rows.append((i, float(t_c[i]), rgb_rel, depth_rel, mask_rel))
    with open(root / FRAMES_NAME, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FRAMES_HEADER)
        for i, t, rgb_rel, depth_rel, mask_rel in rows:
            w.writerow([i, repr(t), rgb_rel, depth_rel, mask_rel])
    with open(root / JOINTS_NAME, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        names = (
            list(joint_names) if joint_names is not None else ["q%d" % j for j in range(q.shape[1])]
        )
        w.writerow([JOINTS_HEADER_0] + names)
        for i in range(t_r.size):
            w.writerow([repr(float(t_r[i]))] + [repr(float(v)) for v in q[i]])
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "camera": k.to_dict(),
        "mount": mount.lower(),
        "fps": float(fps) if fps is not None else None,
        "joint_rate_hz": float(joint_rate_hz) if joint_rate_hz is not None else None,
        "depth_scale": float(depth_scale),
        "depth_trunc_m": float(depth_trunc_m) if depth_trunc_m is not None else None,
        "depth_units": "millimetre (16-bit PNG); 0 means NO MEASUREMENT",
        "urdf": urdf,
        "dof": int(q.shape[1]),
        "joint_names": names,
    }
    with open(root / META_NAME, "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=True, default_flow_style=False)
    if ground_truth is not None:
        write_ground_truth(root, ground_truth)
    return root


def write_ground_truth(root: str | Path, gt: Mapping[str, Any]) -> Path:
    p = Path(root) / GT_NAME
    payload = dict(gt)
    payload.setdefault(
        "note",
        "Evaluation only. This file is never read by CalibDataset or by the optimizer; it exists so that a synthetic run can be scored.",
    )
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)
    return p


def read_ground_truth(root: str | Path) -> dict[str, Any]:
    p = Path(root) / GT_NAME
    if not p.exists():
        raise FileNotFoundError(
            "%r has no %s; only synthetic recordings carry ground truth" % (str(root), GT_NAME)
        )
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
