# This script uses BlenderProc to render multi-view templates of a given CAD model,
# generating RGB images, masks, and NOCS (Normalized Object Coordinate Space) data.
# Main workflow: load CAD, normalize, set camera poses, render, and save results.

import blenderproc as bproc

import os
import argparse
import cv2
import numpy as np
import trimesh

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
args = parser.parse_args()

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

# Initialize BlenderProc
bproc.init()

# Calculate normalization scale for the CAD model
# The model will be scaled so that its maximum radius is 0.5
# Returns the scaling factor
def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)

# Load camera poses (NOCS format)
cam_poses = np.load(cnos_cam_fpath)

# Calculate the scale factor for the CAD model
if args.normalize:
    scale = get_norm_info(args.cad_path)
else:
    scale = 1

# Iterate over each camera pose and render a template for each
for idx, cam_pose in enumerate(cam_poses):
    
    bproc.clean_up()  # Clean up the scene to avoid accumulation

    # Load the CAD model
    obj = bproc.loader.load_obj(args.cad_path)[0]
    obj.set_scale([scale, scale, scale])  # Apply scaling
    obj.set_cp("category_id", 1)  # Set category ID

    # (Optionally) assigning material colors to untextured objects
    if args.colorize:
        color = [args.base_color, args.base_color, args.base_color, 0.]
        material = bproc.material.create('obj')
        material.set_principled_shader_value('Base Color', color)
        obj.set_material(0, material)

    # Convert cnos camera pose from NOCS to Blender format
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
    bproc.camera.add_camera_pose(cam_pose)
    
    # Set up a point light source
    light_scale = 2.5
    light_energy = 1000
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    light1.set_energy(light_energy)

    # Set the number of rendering samples
    bproc.renderer.set_max_amount_of_samples(50)
    # Render whole pipeline (RGB and NOCS data)
    data = bproc.renderer.render()
    data.update(bproc.renderer.render_nocs())
    
    # Ensure the save directory exists
    save_fpath = os.path.join(args.output_dir, "templates")
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    # Save the RGB image (convert BGR to RGB)
    color_bgr_0 = data["colors"][0]
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
    cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

    # Save the mask (last channel of NOCS)
    mask_0 = data["nocs"][0][..., -1]
    cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
    
    # Save the NOCS coordinates (normalized to [-1,1])
    xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
    np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))