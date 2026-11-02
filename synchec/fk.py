from __future__ import annotations
import os
import xml.etree.ElementTree as ET
import torch
from .lie import compose, so3_exp


def _rpy_to_matrix(rpy: tuple[float, float, float], dtype: torch.dtype) -> torch.Tensor:
    (r, p, y) = (torch.tensor(float(v), dtype=dtype) for v in rpy)
    (cr, sr) = (torch.cos(r), torch.sin(r))
    (cp, sp) = (torch.cos(p), torch.sin(p))
    (cy, sy) = (torch.cos(y), torch.sin(y))
    Rx = torch.stack(
        [
            torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)]),
            torch.stack([torch.zeros_like(cr), cr, -sr]),
            torch.stack([torch.zeros_like(cr), sr, cr]),
        ]
    )
    Ry = torch.stack(
        [
            torch.stack([cp, torch.zeros_like(cp), sp]),
            torch.stack([torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)]),
            torch.stack([-sp, torch.zeros_like(cp), cp]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cy, -sy, torch.zeros_like(cy)]),
            torch.stack([sy, cy, torch.zeros_like(cy)]),
            torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)]),
        ]
    )
    return Rz @ Ry @ Rx


def _parse_origin(node: ET.Element | None, dtype: torch.dtype) -> torch.Tensor:
    if node is None:
        return torch.eye(4, dtype=dtype)
    o = node.find("origin")
    if o is None:
        return torch.eye(4, dtype=dtype)
    xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
    return compose(_rpy_to_matrix(tuple(rpy), dtype), torch.tensor(xyz, dtype=dtype))


def _load_obj(path: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    first = tok.split("/")[0]
                    if not first:
                        continue
                    i = int(first)
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts or not faces:
        raise ValueError("no geometry parsed from %s" % os.path.basename(path))
    return (torch.tensor(verts, dtype=dtype), torch.tensor(faces, dtype=torch.long))


def box_mesh(
    half: tuple[float, float, float] | float,
    dtype: torch.dtype = torch.float32,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(half, (int, float)):
        half = (float(half), float(half), float(half))
    (hx, hy, hz) = half
    (cx, cy, cz) = center
    v = torch.tensor(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=dtype,
    ) + torch.tensor([cx, cy, cz], dtype=dtype)
    f = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=torch.long,
    )
    return (v, f)


def cylinder_mesh(
    radius: float, length: float, segments: int = 12, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    ang = torch.arange(segments, dtype=dtype) * (2.0 * torch.pi / segments)
    x = radius * torch.cos(ang)
    y = radius * torch.sin(ang)
    hz = length * 0.5
    bottom = torch.stack([x, y, torch.full_like(x, -hz)], dim=-1)
    top = torch.stack([x, y, torch.full_like(x, hz)], dim=-1)
    centers = torch.tensor([[0.0, 0.0, -hz], [0.0, 0.0, hz]], dtype=dtype)
    v = torch.cat([bottom, top, centers], dim=0)
    (ib, it) = (2 * segments, 2 * segments + 1)
    faces = []
    for k in range(segments):
        n = (k + 1) % segments
        faces.append([k, n, segments + n])
        faces.append([k, segments + n, segments + k])
        faces.append([ib, n, k])
        faces.append([it, segments + k, segments + n])
    return (v, torch.tensor(faces, dtype=torch.long))


def sphere_mesh(
    radius: float, rings: int = 6, sectors: int = 10, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    verts = [[0.0, 0.0, radius]]
    for i in range(1, rings):
        phi = torch.pi * i / rings
        for j in range(sectors):
            th = 2.0 * torch.pi * j / sectors
            verts.append(
                [
                    radius * float(torch.sin(torch.tensor(phi)) * torch.cos(torch.tensor(th))),
                    radius * float(torch.sin(torch.tensor(phi)) * torch.sin(torch.tensor(th))),
                    radius * float(torch.cos(torch.tensor(phi))),
                ]
            )
    verts.append([0.0, 0.0, -radius])
    last = len(verts) - 1
    faces = []
    for j in range(sectors):
        faces.append([0, 1 + j, 1 + (j + 1) % sectors])
    for i in range(rings - 2):
        (base, nxt) = (1 + i * sectors, 1 + (i + 1) * sectors)
        for j in range(sectors):
            jn = (j + 1) % sectors
            faces.append([base + j, nxt + j, nxt + jn])
            faces.append([base + j, nxt + jn, base + jn])
    base = 1 + (rings - 2) * sectors
    for j in range(sectors):
        faces.append([last, base + (j + 1) % sectors, base + j])
    return (torch.tensor(verts, dtype=dtype), torch.tensor(faces, dtype=torch.long))


class _Joint:
    __slots__ = (
        "name",
        "type",
        "parent",
        "child",
        "origin",
        "axis",
        "limit_lower",
        "limit_upper",
        "q_index",
    )

    def __init__(
        self,
        name: str,
        jtype: str,
        parent: str,
        child: str,
        origin: torch.Tensor,
        axis: torch.Tensor,
        limit_lower: float,
        limit_upper: float,
        q_index: int | None,
    ):
        self.name = name
        self.type = jtype
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis
        self.limit_lower = limit_lower
        self.limit_upper = limit_upper
        self.q_index = q_index


class Robot:
    def __init__(
        self,
        joints: list[_Joint],
        root: str,
        link_meshes: dict[str, tuple[torch.Tensor, torch.Tensor]],
        flange_link: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.joints = joints
        self.root = root
        self.dtype = dtype
        self.dof = sum((1 for j in joints if j.q_index is not None))
        self._order = self._topological_order()
        self.link_names = [n for n in self._order if n in link_meshes]
        self.link_meshes = {k: link_meshes[k] for k in self.link_names}
        self.flange_link = flange_link or (self._order[-1] if self._order else root)
        self._joint_by_child = {j.child: j for j in joints}
        self._limits = self._collect_limits()
        self._cat_cache: tuple[torch.Tensor, torch.Tensor, list[int]] | None = None

    @classmethod
    def from_urdf(
        cls, path: str, dtype: torch.dtype = torch.float32, mesh_dir: str | None = None
    ) -> "Robot":
        tree = ET.parse(path)
        root_el = tree.getroot()
        base_dir = mesh_dir or os.path.dirname(os.path.abspath(path))
        joints: list[_Joint] = []
        q_index = 0
        children: set[str] = set()
        parents: set[str] = set()
        for je in root_el.findall("joint"):
            jtype = je.get("type", "fixed")
            if jtype not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"Unsupported URDF joint type: {jtype}")
            if je.find("mimic") is not None:
                raise ValueError("Mimic joints must be resolved in the calibration URDF")
            parent = je.find("parent").get("link")
            child = je.find("child").get("link")
            parents.add(parent)
            children.add(child)
            origin = _parse_origin(je, dtype)
            ax = je.find("axis")
            axis = torch.tensor(
                [
                    float(v)
                    for v in ((ax.get("xyz") if ax is not None else None) or "1 0 0").split()
                ],
                dtype=dtype,
            )
            n = float(axis.norm())
            if n > 0:
                axis = axis / n
            lim = je.find("limit")
            lo = float(lim.get("lower", "-3.14159")) if lim is not None else -3.14159
            hi = float(lim.get("upper", "3.14159")) if lim is not None else 3.14159
            movable = jtype in ("revolute", "continuous", "prismatic")
            joints.append(
                _Joint(
                    je.get("name", "joint%d" % len(joints)),
                    jtype,
                    parent,
                    child,
                    origin,
                    axis,
                    lo,
                    hi,
                    q_index if movable else None,
                )
            )
            if movable:
                q_index += 1
        roots = sorted(parents - children)
        roots = sorted({link.get("name") for link in root_el.findall("link")} - children)
        if len(roots) != 1:
            raise ValueError("URDF must define one connected kinematic tree")
        base = roots[0]
        meshes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for le in root_el.findall("link"):
            name = le.get("name")
            got = cls._link_geometry(le, base_dir, dtype)
            if got is None:
                continue
            meshes[name] = got
        return cls(joints, base, meshes, dtype=dtype)

    @staticmethod
    def _link_geometry(le: ET.Element, base_dir: str, dtype: torch.dtype):
        from pathlib import Path
        import trimesh

        parts = []
        for visual in le.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                raise ValueError(f"Link {le.get('name')} has a visual without geometry")
            origin = _parse_origin(visual, dtype)
            mesh = geometry.find("mesh")
            box = geometry.find("box")
            cylinder = geometry.find("cylinder")
            sphere = geometry.find("sphere")
            if mesh is not None:
                filename = mesh.get("filename")
                if not filename:
                    raise ValueError(f"Link {le.get('name')} has an empty mesh filename")
                if filename.startswith("package://"):
                    relative = Path(filename[len("package://") :])
                    candidates = []
                    for parent in (Path(base_dir), *Path(base_dir).parents):
                        candidates.append(parent / relative)
                        if relative.parts and parent.name == relative.parts[0]:
                            candidates.append(parent.joinpath(*relative.parts[1:]))
                    path = next((p for p in candidates if p.is_file()), None)
                else:
                    path = Path(filename.removeprefix("file://"))
                    if not path.is_absolute():
                        path = Path(base_dir) / path
                    path = path if path.is_file() else None
                if path is None:
                    raise FileNotFoundError(f"Cannot resolve URDF mesh: {filename}")
                loaded = trimesh.load_mesh(path, process=False)
                if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
                    raise ValueError(f"Mesh has no triangles: {filename}")
                verts = torch.tensor(loaded.vertices.copy(), dtype=dtype)
                faces = torch.tensor(loaded.faces.copy(), dtype=torch.long)
                scale = torch.tensor(
                    [float(v) for v in mesh.get("scale", "1 1 1").split()], dtype=dtype
                )
                if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale == 0).any():
                    raise ValueError(f"Invalid mesh scale: {filename}")
                verts = verts * scale
                if float(scale.prod()) < 0:
                    faces = faces[:, [0, 2, 1]]
            elif box is not None:
                size = [float(v) for v in box.attrib["size"].split()]
                if len(size) != 3 or min(size) <= 0:
                    raise ValueError("Box dimensions must be positive")
                verts, faces = box_mesh(tuple(v / 2 for v in size), dtype)
            elif cylinder is not None:
                radius, length = float(cylinder.attrib["radius"]), float(cylinder.attrib["length"])
                if min(radius, length) <= 0:
                    raise ValueError("Cylinder dimensions must be positive")
                verts, faces = cylinder_mesh(radius, length, dtype=dtype)
            elif sphere is not None:
                radius = float(sphere.attrib["radius"])
                if radius <= 0:
                    raise ValueError("Sphere radius must be positive")
                verts, faces = sphere_mesh(radius, dtype=dtype)
            else:
                raise ValueError(f"Unsupported visual geometry on link {le.get('name')}")
            parts.append((verts @ origin[:3, :3].T + origin[:3, 3], faces))
        if not parts:
            return None
        vertices, triangles, offset = [], [], 0
        for verts, faces in parts:
            vertices.append(verts)
            triangles.append(faces + offset)
            offset += len(verts)
        return (torch.cat(vertices), torch.cat(triangles))

    @classmethod
    def from_primitives(
        cls,
        n_links: int = 3,
        link_length: float = 0.25,
        link_half_width: float = 0.035,
        dtype: torch.dtype = torch.float32,
    ) -> "Robot":
        joints: list[_Joint] = []
        meshes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        (base_v, base_f) = cylinder_mesh(0.06, 0.08, dtype=dtype)
        meshes["link0"] = (base_v, base_f)
        for i in range(n_links):
            axis = torch.tensor([0.0, 0.0, 1.0] if i % 2 == 0 else [0.0, 1.0, 0.0], dtype=dtype)
            origin = torch.eye(4, dtype=dtype)
            origin[2, 3] = 0.04 if i == 0 else 0.0
            if i > 0:
                origin[0, 3] = link_length
            joints.append(
                _Joint(
                    "j%d" % i,
                    "revolute",
                    "link%d" % i,
                    "link%d" % (i + 1),
                    origin,
                    axis,
                    -2.6,
                    2.6,
                    i,
                )
            )
            (v, f) = box_mesh(
                (link_length * 0.5, link_half_width, link_half_width),
                dtype,
                center=(link_length * 0.5, 0.0, 0.0),
            )
            meshes["link%d" % (i + 1)] = (v, f)
        return cls(joints, "link0", meshes, flange_link="link%d" % n_links, dtype=dtype)

    def _topological_order(self) -> list[str]:
        kids: dict[str, list[str]] = {}
        for j in self.joints:
            kids.setdefault(j.parent, []).append(j.child)
        (order, stack) = ([], [self.root])
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            order.append(node)
            stack.extend(reversed(sorted(kids.get(node, []))))
        return order

    def _collect_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.full((self.dof,), -3.14159, dtype=self.dtype)
        hi = torch.full((self.dof,), 3.14159, dtype=self.dtype)
        for j in self.joints:
            if j.q_index is not None:
                lo[j.q_index] = j.limit_lower
                hi[j.q_index] = j.limit_upper
        return (lo, hi)

    @property
    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._limits

    def link_index(self, name: str) -> int:
        return self.link_names.index(name)

    def _joint_transform(self, joint: _Joint, q: torch.Tensor) -> torch.Tensor:
        n = q.shape[0]
        if joint.type == "prismatic":
            R = torch.eye(3, dtype=q.dtype, device=q.device).expand(n, 3, 3)
            t = joint.axis.to(q.dtype).to(q.device) * q.unsqueeze(-1)
            return compose(R, t)
        if joint.type in ("revolute", "continuous"):
            phi = joint.axis.to(q.dtype).to(q.device).unsqueeze(0) * q.unsqueeze(-1)
            return compose(so3_exp(phi), torch.zeros(n, 3, dtype=q.dtype, device=q.device))
        return torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4)

    def link_transforms(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] != self.dof:
            raise ValueError("expected q with %d columns, got %d" % (self.dof, q.shape[-1]))
        n = q.shape[0]
        out = {self.root: torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4)}
        for name in self._order:
            if name == self.root:
                continue
            j = self._joint_by_child[name]
            parent = out.get(j.parent)
            if parent is None:
                continue
            origin = j.origin.to(q.dtype).to(q.device).expand(n, 4, 4)
            qi = (
                q[:, j.q_index]
                if j.q_index is not None
                else torch.zeros(n, dtype=q.dtype, device=q.device)
            )
            out[name] = parent @ origin @ self._joint_transform(j, qi)
        return out

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        tf = self.link_transforms(q)
        return torch.stack([tf[name] for name in self.link_names], dim=1)

    def flange(self, q: torch.Tensor) -> torch.Tensor:
        tf = self.link_transforms(q)
        if self.flange_link in tf:
            return tf[self.flange_link]
        for name in reversed(self._order):
            if name in tf:
                return tf[name]
        raise ValueError("no link transforms computed")

    def _concat_meshes(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if self._cat_cache is None:
            (vs, fs, counts, offset) = ([], [], [], 0)
            for name in self.link_names:
                (v, f) = self.link_meshes[name]
                vs.append(v)
                fs.append(f + offset)
                counts.append(v.shape[0])
                offset += v.shape[0]
            self._cat_cache = (torch.cat(vs, 0), torch.cat(fs, 0), counts)
        return self._cat_cache

    def posed_mesh(self, link_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (verts, faces, counts) = self._concat_meshes()
        if link_T.shape[1] != len(counts):
            raise ValueError(
                "link_T has %d links but the robot renders %d" % (link_T.shape[1], len(counts))
            )
        verts = verts.to(link_T.dtype).to(link_T.device)
        idx = torch.repeat_interleave(
            torch.arange(len(counts), device=link_T.device),
            torch.tensor(counts, device=link_T.device),
        )
        T = link_T[:, idx]
        v = (T[..., :3, :3] @ verts.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        return (v + T[..., :3, 3], faces.to(link_T.device))

    def n_faces(self) -> int:
        return int(self._concat_meshes()[1].shape[0])
