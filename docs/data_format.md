# Recording format

A recording contains synchronized file references and the original image and joint timestamps.

```text
recording/
  meta.yaml
  joints.csv
  frames.csv
  rgb/
  depth/
  mask/
  base_mask.png
  robot/
    robot.urdf
    meshes/
```

## Metadata

```yaml
format_version: 1
mount: eth
camera:
  fx: 615.0
  fy: 615.0
  cx: 423.5
  cy: 239.5
  width: 848
  height: 480
fps: 30
joint_rate_hz: 250
depth_scale: 0.001
urdf: robot/robot.urdf
flange_link: tool_link
```

Replace camera intrinsics, image dimensions, and robot paths with the values for the recording. The mount is eth or eih. For EiH, flange_link identifies the link whose frame defines T_ce. For EtH, the URDF root defines the base frame.

Mesh filenames may be relative to the URDF, absolute, or use package:// paths. Meshes must be present. OBJ, STL, and DAE are loaded through trimesh. Primitive URDF visuals are also supported. A calibration URDF contains fixed, revolute, continuous, or prismatic joints; resolve mimic joints before export.

## Joint log

The first CSV column is t_r_seconds. Remaining columns contain joint positions.

```csv
t_r_seconds,q0,q1,q2,q3,q4,q5,q6
0.000,0.1,-0.2,0.3,0.0,0.4,-0.1,0.2
0.004,0.101,-0.199,0.301,0.0,0.399,-0.099,0.201
```

Named columns may use the URDF's movable-joint names in any order; the loader reorders them to the kinematic model. Generic q0, q1, and subsequent columns use the movable-joint order in the URDF. If meta.yaml specifies joint_names, it must match the CSV header order.

Timestamps must be finite and strictly increasing. Joint angles are radians and prismatic displacements are meters. Use consistent clock origins for the image and joint streams.

## Image frames

The header is exactly:

```csv
index,t_c_seconds,rgb,depth,mask
0,0.100,rgb/000000.png,depth/000000.png,mask/000000.png
1,0.133333333,rgb/000001.png,depth/000001.png,mask/000001.png
```

Paths are relative to the recording directory. The RGB cell may be empty. All supplied images match the camera dimensions.

| Input | Encoding |
| --- | --- |
| Robot mask | Grayscale PNG, 0 background and 255 robot |
| Depth | Single-channel PNG; raw value × depth_scale gives meters |
| RGB | Color image, optional |
| Base-link mask | Grayscale PNG, 0 background and 255 base link |

Nonpositive and nonfinite depth pixels are excluded from supervision. The base-link mask is used for Stage 1; it must refer to the frame selected by --static-frame.

The frame selector ranks observed optical-flow magnitude over robot pixels. It uses RGB when every frame has RGB, and mask images otherwise. The temporal optimizer separately gates the selected frames using synchronized joint velocity.

## Spatial initialization

An existing extrinsic can be passed as JSON:

```json
{
  "extrinsic": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.5],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

The example illustrates the format. Supply the calibrated T_cb or T_ce for the selected mounting model.

## Outputs

calibration.json contains the extrinsic, its transform convention, recovered offset in seconds and milliseconds, selected frame indices, configuration, renderer, optimizer, and per-cycle history. Temporal and spatial update counts are recorded separately.

The synthetic generator additionally writes ground_truth.yaml for evaluation and declares robot_model: box_links in meta.yaml. Real recordings identify their geometry through urdf.
