# SyncHeC

**Markerless Spatio-Temporal Hand-Eye Calibration via Differentiable Rendering and Continuous-Time Joint Optimization**

SyncHeC estimates hand-eye extrinsics and the image–joint time offset from a continuous robot motion. Continuous-time forward kinematics connects the time offset to dense mask and depth rendering losses. An alternating optimizer updates pose and time in separate mini-batch blocks.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/hongzhengdong/SyncHeC.git
cd SyncHeC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The PyTorch soft renderer runs on CPU. For the CUDA renderer, install CUDA-enabled PyTorch and [Nvdiffrast](https://nvlabs.github.io/nvdiffrast/) with a compatible CUDA toolkit:

```bash
python -m pip install setuptools wheel ninja
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

The renderer is selected explicitly through the configuration.

## Synthetic example

```bash
python run_synthetic.py --out runs/synthetic
```

This generates 24 images at 64 × 48 resolution from a four-link robot with an 87 ms image–joint offset. Calibration starts from a perturbed reference pose and a zero time offset. The unregularized synthetic configuration is in [configs/synthetic.json](configs/synthetic.json).

The command writes the recording and a report containing the initial and final errors, recovered transform, configuration, and optimization history:

```text
runs/synthetic/
  recording/
  report.json
```

Additional experiments use the same entry point:

```bash
python run_synthetic.py --joint-descent --out runs/joint
python run_synthetic.py --no-depth --out runs/mask_only
python run_synthetic.py --constant-velocity --out runs/constant_velocity
```

## Recording calibration

Prepare the recording using the [data format](docs/data_format.md), including robot masks, metric depth, camera intrinsics, and a URDF with its meshes.

```bash
python run_calibration.py \
  --recording data/recording \
  --config configs/calibration.json \
  --base-mask data/recording/base_mask.png \
  --out runs/calibration
```

The base-link mask belongs to the static frame selected by --static-frame, which defaults to frame 0. Stage 1 estimates the camera pose from this frame before joint space–time optimization. Set the base-cylinder radius and camera height in the configuration to the geometry of the recording.

To start from an existing spatial calibration:

```bash
python run_calibration.py \
  --recording data/recording \
  --config configs/calibration.json \
  --init-extrinsic data/initial_extrinsic.json \
  --out runs/calibration
```

The initialization JSON contains a 4 × 4 matrix under the extrinsic key. The same format is written to runs/calibration/calibration.json.

[configs/calibration.json](configs/calibration.json) selects Nvdiffrast on CUDA and uses 60 selected frames, batches of four, and 15 passes per block. With 60 frames above the temporal velocity threshold, each cycle executes 225 temporal and 225 spatial Adam updates. Optimizer moments persist across passes and cycles. The temporal gate is recomputed at the beginning of every cycle.

For CPU execution, add --device cpu --backend soft. Recording inputs determine the actual image dimensions and timestamps.

## Coordinate and time conventions

| Quantity | Convention |
| --- | --- |
| EtH extrinsic | T_cb maps robot-base coordinates into camera coordinates |
| EiH extrinsic | T_ce maps flange coordinates into camera coordinates |
| Synchronized time | t_joint = t_camera − delta_t |
| Translation and depth | Meters |
| Time and offsets | Seconds internally; CLI flags ending in -ms use milliseconds |
| Revolute joints | Radians |
| Prismatic joints | Meters |
| Image coordinates | x right, y down; camera z forward |

The joint log must cover the selected image timestamps and the offset search interval. The optimizer bounds the offset by this coverage.

Masks and depth maps are observation inputs. The function synchec.depth.fuse_metric_depth aligns predicted depth to the sensor by the median ratio on their valid overlap, preserves valid sensor measurements, and fills missing pixels. Mask residuals use the full image. Depth residuals are normalized separately for each frame and then averaged; the offset penalty is applied once per mini-batch objective.

## Multiple cameras

synchec.optimize.calibrate_multicamera accepts a list of CalibrationProblem instances and one initial extrinsic per camera. The default estimates camera-specific offsets. Set shared_offset=True for a shared offset and separate extrinsics.

Shared-offset problems use the same timestamped joint log. Their temporal objective sums the camera data terms and adds one offset penalty. The shared model applies when the cameras' timing references establish offset equivalence.

## Trajectory refinement

The trajectory utility optimizes a 25-control-point B-spline using projected URDF landmarks:

```bash
python run_planner.py \
  --recording data/recording \
  --extrinsic runs/calibration/calibration.json \
  --out runs/trajectory
```

It writes joint samples to trajectory.csv and spline parameters to trajectory.json. An initial spline can be supplied through --initial-trajectory. Joint limits are enforced on the control points.

The planner API accepts a collision_free callback and a differentiable visibility penalty. The CLI loads these through --constraints module:function; the factory receives the robot, camera, and anchor transform and returns a dictionary of callbacks. The output records which constraints were applied. This utility performs landmark-based spline refinement; the paper's acquisition planner scores rendered pixels and uses an RRT* seed.

## Package

| Module | Function |
| --- | --- |
| config.py | Configuration and update accounting |
| data.py | Recording I/O and optical-flow frame selection |
| depth.py | Sensor-aligned metric depth completion |
| fk.py | Differentiable URDF forward kinematics and mesh loading |
| kinematics.py | Continuous-time interpolation and camera mounting models |
| lie.py | SO(3) and SE(3) operations |
| render.py | Soft and Nvdiffrast rendering; mask and depth objectives |
| stage1.py | Spatial anchoring and mask refinement |
| optimize.py | Alternating, joint, and multi-camera calibration |
| planner.py | B-splines and excitation-based trajectory refinement |
| synthetic.py | Synthetic robot recordings |

## Citation

```bibtex
@misc{hong2026synchec,
  author = {Hong, Zhengdong},
  title = {{SyncHeC}: Markerless Spatio-Temporal Hand-Eye Calibration
           via Differentiable Rendering and Continuous-Time Joint Optimization},
  year = {2026},
  url = {https://github.com/hongzhengdong/SyncHeC}
}
```

## License

The code is licensed for academic and non-commercial research use under [LICENSE](LICENSE).
