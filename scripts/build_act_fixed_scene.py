#!/usr/bin/env python3
"""Build/review a simulation-only scene; no ROS, CAN, recording or training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "sim/scenes/act_fixed_20261015"


def vec(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def camera(parent, name, position, target, fovy=48):
    z = np.asarray(position) - np.asarray(target)
    z = z / np.linalg.norm(z)
    up = np.array([0., 0., 1.]) if abs(z[2]) < .99 else np.array([-1., 0., 0.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return ET.SubElement(parent, "camera", name=name, pos=vec(position),
                         xyaxes=vec(np.r_[x, y]), fovy=str(fovy))


def build(directory):
    cfg = json.loads((directory / "measurements.json").read_text())
    t, c, b = cfg["table"], cfg["cube"], cfg["black_box"]
    depth, width = t["size_forward_lateral_m"]
    near = t["column_front_x_m"] + t["near_edge_gap_from_column_m"]
    cx, cy, z = near + depth / 2, t["center_y_m"], t["top_z_m"]
    cube = [near + c["gap_near_long_edge_m"] + c["edge_m"] / 2,
            cy - width / 2 + c["gap_robot_right_short_edge_m"] + c["edge_m"] / 2,
            z + c["edge_m"] / 2]
    box = [cx, cy - width / 2 + b["gap_robot_right_short_edge_m"] + b["outer_size_m"][1] / 2, z]
    root = ET.Element("mujoco", model="openarm_fixed_scene_20261015_review")
    root.append(ET.Comment("SIMULATION GEOMETRY PREVIEW. See measurements.json for assumptions. Not a hardware motion configuration."))
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.001", cone="elliptic", impratio="10")
    ET.SubElement(root, "statistic", center=vec([.23, 0, .36]), extent="1.15")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960", azimuth="145", elevation="-25")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.35 0.35 0.35", specular="0 0 0")
    asset = ET.SubElement(root, "asset")
    robot = ROOT / "official/openarm_mujoco/v2/pedestal/openarm_pedestal.xml"
    ET.SubElement(asset, "model", name="robot", file=os.path.relpath(robot, directory))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.4 -0.4 1.6", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="base_datum_surface", type="plane", size="1.5 1.5 .01",
                  pos="0 0 -0.001", rgba=".19 .28 .25 1")
    ET.SubElement(world, "attach", model="robot", prefix="")
    table = ET.SubElement(world, "body", name="table", pos=vec([cx, cy, z-t["thickness_m"]/2]))
    # Rounded slab made of convex primitives; same shape for rendering/contact.
    hx, hy, h, r = depth/2, width/2, t["thickness_m"]/2, t["corner_radius_m"]
    for name, size in [("table_top", [hx-r,hy,h]), ("table_top_cross", [hx,hy-r,h])]:
        ET.SubElement(table, "geom", name=name, type="box", size=vec(size), rgba=".88 .87 .82 1",
                      friction="1 .005 .0001")
    for i, (sx, sy) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):
        ET.SubElement(table,"geom",name=f"table_corner_{i}",type="cylinder",pos=vec([sx*(hx-r),sy*(hy-r),0]),
                      size=vec([r,h]),rgba=".88 .87 .82 1",friction="1 .005 .0001")
    obj = ET.SubElement(world, "body", name="orange_cube", pos=vec(cube), euler=vec([0,0,c["yaw_rad"]]))
    ET.SubElement(obj, "freejoint", name="cube_free")
    ET.SubElement(obj, "geom", name="blue_cube_geom", type="box", size=vec([c["edge_m"]/2]*3),
                  rgba=".02 .18 .9 1", mass=str(c["mass_kg"]), friction=vec(c["friction"]))
    # Preserve the existing open five-piece container, rather than a solid mesh.
    legacy = ET.parse(directory / b["source"]).getroot()
    frame = copy.deepcopy(legacy.find("./worldbody/body[@name='black_frame']"))
    frame.set("pos", vec(box))
    for geom, name in zip(frame.findall("geom"), ["floor","x_plus","x_minus","y_plus","y_minus"]):
        geom.set("name", "black_box_"+name)
    world.append(frame)
    camera(world, "preview_top", [cx,0,1.65], [cx,0,z], 35)
    camera(world, "preview_overview", [1.25,-1.10,1.12], [.20,0,.34], 43)
    camera(world, "preview_side", [.28,-1.65,.65], [.28,0,.36], 36)
    calibration=directory/'source_snapshot/global_camera_20261015.calib'
    if calibration.exists():
        from scipy.spatial.transform import Rotation
        cal=yaml.safe_load(calibration.read_text())
        if cal['parameters']['robot_base_frame']!='world' or cal['parameters']['tracking_base_frame']!='global_camera_link':
            raise ValueError('Unexpected global calibration frames')
        tr=cal['transform']['translation'];qr=cal['transform']['rotation']
        R=Rotation.from_quat([qr[k] for k in ['x','y','z','w']]).as_matrix()
        # ROS link: x forward/y left/z up. MuJoCo camera: x right/y up/-z forward.
        # Optical translation and actual intrinsics remain explicitly pending.
        world.append(ET.Comment('Global calibrated link pose; RGB optical offset/intrinsics still pending. fovy=42 is a preview assumption.'))
        ET.SubElement(world,'camera',name='camera_ceiling',pos=vec([tr[k] for k in ['x','y','z']]),
                      xyaxes=vec(np.r_[-R[:,1],R[:,2]]),fovy='42')
    else:
        camera(world, "camera_ceiling", [cx,0,1.0], [cx,0,z], 48)
    ET.SubElement(root, "custom")
    ET.SubElement(root.find("custom"), "text", name="scene_status", data=cfg['status'])
    path = directory / "scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    keys = ET.SubElement(root, "keyframe")
    home = yaml.safe_load((directory/"source_snapshot/physical_home.yaml").read_text())
    for name in ["mechanical_zero_preview", "physical_home_reference"]:
        data = mujoco.MjData(model)
        for side, grip in [("left", .35), ("right", -.35)]:
            for index in (1,2):
                joint = model.joint(f"openarm_{side}_finger_joint{index}")
                data.qpos[joint.qposadr[0]] = grip
        if name == "physical_home_reference":
            for i, q in enumerate(home["right_home_joint_positions_rad"], 1):
                data.qpos[model.joint(f"openarm_right_joint{i}").qposadr[0]] = q
        for i in range(model.nu):
            data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i,0]]]
        ET.SubElement(keys,"key",name=name,qpos=vec(data.qpos),ctrl=vec(data.ctrl))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    derived = {"table_near_x_m":near,"table_far_x_m":near+depth,"table_center_surface_xyz_m":[cx,cy,z],
               "cube_center_xyz_m":cube,"box_bottom_center_xyz_m":box,
               "placement_interpretation":"box short-edge distance measured from robot-right edge, same as cube",
               "model_origin_note":"z=0 is robot base datum, not room floor",
               "scene_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory/"derived_geometry.json").write_text(json.dumps(derived,indent=2)+"\n")
    return path, cfg, derived


def validate(path, cfg, derived):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key("mechanical_zero_preview").id)
    mujoco.mj_forward(m,d)
    cube = m.body("orange_cube").id
    assert np.allclose(d.xpos[cube], derived["cube_center_xyz_m"])
    assert m.nu == 16 and m.nq == 25
    assert len([i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("black_box_")]) == 5
    t,c,b=cfg["table"],cfg["cube"],cfg["black_box"]
    table_ids=np.flatnonzero(m.geom_bodyid==m.body("table").id)
    lower=[];upper=[]
    for i in table_ids:
        half=m.geom_size[i].copy()
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half=np.array([half[0],half[0],half[1]])
        lower.append(d.geom_xpos[i]-half);upper.append(d.geom_xpos[i]+half)
    lo=np.min(lower,axis=0);hi=np.max(upper,axis=0)
    assert np.allclose(hi-lo,[*t["size_forward_lateral_m"],t["thickness_m"]])
    assert np.isclose(hi[2],t["top_z_m"])
    assert np.isclose(lo[0]-t["column_front_x_m"],t["near_edge_gap_from_column_m"])
    cg=m.geom("blue_cube_geom").id
    cube_min=d.geom_xpos[cg]-m.geom_size[cg]
    assert np.allclose(2*m.geom_size[cg],[c["edge_m"]]*3)
    assert np.isclose(cube_min[0]-lo[0],c["gap_near_long_edge_m"])
    assert np.isclose(cube_min[1]-lo[1],c["gap_robot_right_short_edge_m"])
    box_ids=np.flatnonzero(m.geom_bodyid==m.body("black_frame").id)
    blo=np.min(d.geom_xpos[box_ids]-m.geom_size[box_ids],axis=0)
    bhi=np.max(d.geom_xpos[box_ids]+m.geom_size[box_ids],axis=0)
    assert np.allclose(bhi-blo,b["outer_size_m"]), "configured box dimensions differ from source XML"
    assert np.isclose(blo[1]-lo[1],b["gap_robot_right_short_edge_m"])
    assert np.isclose(blo[2],hi[2])
    # Contact check uses this scene only; it is not a robot execution clearance.
    initial = [{"geom1":m.geom(c.geom1).name,"geom2":m.geom(c.geom2).name,"distance_m":float(c.dist)}
               for c in d.contact if c.dist < -1e-5]
    start = d.xpos[cube].copy()
    for _ in range(2000):
        mujoco.mj_step(m,d)
    assert np.isfinite(d.qpos).all()
    displacement = np.linalg.norm(d.xpos[cube]-start)
    assert displacement < .002, f"cube settling drift {displacement}"
    report={"mujoco_version":mujoco.__version__,"xml_load":"passed","nq":m.nq,"nu":m.nu,
            "initial_penetrations_over_10um":initial,"cube_2s_settling_displacement_m":float(displacement),
            "geometry_checks":"compiled table size/height/gap, cube size/edge gaps, box outer size/edge gap passed",
            "expert_rollout_tested":False,"camera_calibrated":False,
            "hardware_connected":False,"approved_for_collection":False}
    (path.parent/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def render(path, cfg, derived):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key("mechanical_zero_preview").id)
    mujoco.mj_forward(m,d)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    out=path.parent/"previews";out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout="constrained")
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        for ax, cam, title in zip(axes.flat,["preview_overview","preview_top","preview_side"],
                                  ["OpenArm + measured tabletop layout","Top view (robot at top; robot right at left)","Side view: tabletop 27 cm above robot base"]):
            renderer.update_scene(d,camera=cam,scene_option=option)
            rgb=renderer.render().copy()
            plt.imsave(out/f"{cam}.png",rgb)
            ax.imshow(rgb);ax.axis("off");ax.set_title(title)
    ax=axes[1,1]
    near=derived["table_near_x_m"];far=derived["table_far_x_m"]
    cube=derived["cube_center_xyz_m"];box=derived["box_bottom_center_xyz_m"]
    ax.add_patch(Rectangle((-.3,near),.6,.4,facecolor="#f0eee5",edgecolor="black"))
    ax.add_patch(Rectangle((cube[1]-.02,cube[0]-.02),.04,.04,facecolor="#064ddd"))
    ax.add_patch(Rectangle((box[1]-.07,box[0]-.07),.14,.14,facecolor="#333333"))
    ax.add_patch(Rectangle((box[1]-.06,box[0]-.06),.12,.12,facecolor="#777777"))
    def dim(a,b,label,offset=(0,0)):
        ax.annotate("",xy=a,xytext=b,arrowprops={"arrowstyle":"<->","color":"#805500"})
        ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,ha="center",va="center",fontsize=9,
                bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    dim((-.3,far+.035),(.3,far+.035),"60 cm")
    dim((.335,near),(.335,far),"40 cm")
    dim((cube[1],near),(cube[1],cube[0]-.02),"12.7 cm",(.055,0))
    dim((-.3,cube[0]),(cube[1]-.02,cube[0]),"8.4 cm",(0,-.025))
    dim((-.3,box[0]+.10),(box[1]-.07,box[0]+.10),"24 cm to outer wall")
    ax.plot([0,0],[.03,far],"--",color="#999999",lw=1)
    ax.plot([-.03,.03],[.03,.03],color="black",lw=3)
    dim((.11,.03),(.11,near),"15.5 cm",(.07,0))
    ax.text(-.07,.035,"column front",ha="right",fontsize=9)
    ax.set_xlim(-.36,.40);ax.set_ylim(far+.07,-.01);ax.set_aspect("equal")
    ax.axis("off");ax.set_title("Dimensions / confirm the 24 cm reference edge")
    fig.suptitle("Fixed scene review — geometry draft, cameras/contact not calibrated",fontsize=16)
    fig.savefig(out/"scene_review.png",dpi=140)
    plt.close(fig)
    return out/"scene_review.png"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=DEFAULT)
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--view",action="store_true",help="Open static MuJoCo inspection window; never steps physics")
    args=parser.parse_args()
    path,cfg,derived=build(args.directory.resolve())
    print(path)
    print(json.dumps(validate(path,cfg,derived),indent=2))
    if args.render:print(render(path,cfg,derived))
    if args.view:
        import time
        import mujoco.viewer
        m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m,d,m.key("mechanical_zero_preview").id)
        mujoco.mj_forward(m,d)
        with mujoco.viewer.launch_passive(m,d) as viewer:
            viewer.opt.geomgroup[3]=0
            viewer.cam.lookat[:]=[.23,0,.36];viewer.cam.distance=1.45
            viewer.cam.azimuth=145;viewer.cam.elevation=-25
            while viewer.is_running():
                viewer.sync();time.sleep(.03)


if __name__ == "__main__":
    main()
