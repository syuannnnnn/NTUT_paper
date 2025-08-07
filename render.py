import argparse
import math
import os
import random
import sys
import time
import numpy as np
import bpy
import zipfile
import tempfile
from mathutils import Vector
import mathutils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the STL file",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save rendered images"
)
parser.add_argument(
    "--engine",
    type=str,
    default="CYCLES",
    choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument(
    "--num_images",
    type=int,
    default=12,
    help="Number of images to render (fixed to 12)"
)
parser.add_argument(
    "--resolution",
    type=int,
    default=1024,
    help="Resolution of rendered images"
)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = True

def sample_spherical(radius_min=1.9, radius_max=2.6, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def set_camera_location(camera, option: str):
    assert option in ['fixed', 'random', 'front']
    
    if option == 'fixed':
        x, y, z = 0, -2.25, 0
    elif option == 'random':
        x, y, z = sample_spherical(radius_min=1.9, radius_max=2.6, maxz=1.6, minz=-0.75)
    elif option == 'front':
        x, y, z = 0, -np.random.uniform(1.9, 2.6, 1)[0], 0

    camera.location = x, y, z
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def add_lighting(option: str) -> None:
    assert option in ['fixed', 'random']
    
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    
    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == 'fixed':
        light.energy = 30000
        bpy.data.objects["Area"].location = (0, 1, 0.5)
    elif option == 'random':
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location = (
            random.uniform(-2., 2.),
            random.uniform(-2., 2.),
            random.uniform(1.0, 3.0)
        )

    bpy.data.objects["Area"].scale = (200, 200, 200)

def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    if object_path.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj

def normalize_scene(box_scale: float):
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 24
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def save_images(object_file: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene(box_scale=2)
    add_lighting(option='random')
    camera, cam_constraint = setup_camera()
    
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
    # img_dir = os.path.join(args.output_dir, 'images')
    
    img_dir = args.output_dir
    os.makedirs(img_dir, exist_ok=True)
    
    for i in range(12):  # Fixed to 12 images
        camera_option = 'random' if i > 0 else 'front'
        camera = set_camera_location(camera, option=camera_option)
        
        render_path = os.path.join(img_dir, f"{object_uid}_{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

def extract_stl_from_zip(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find the first .stl file in the zip
        stl_files = [f for f in zip_ref.namelist() if f.lower().endswith('.stl')]
        if not stl_files:
            raise ValueError("No STL file found in the ZIP archive")
        if len(stl_files) > 1:
            print(f"Multiple STL files found, using the first one: {stl_files[0]}")
        
        # Extract to a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_ref.extract(stl_files[0], temp_dir)
        return os.path.join(temp_dir, stl_files[0])

if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.lower().endswith('.zip'):
            stl_path = extract_stl_from_zip(args.object_path)
        else:
            raise ValueError("Input file must be a ZIP file")
        
        save_images(stl_path)
        end_i = time.time()
        print(f"Finished rendering {stl_path} in {end_i - start_i} seconds")
        
        # Clean up temporary files
        temp_dir = os.path.dirname(stl_path)
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to render {args.object_path}")
        print(e)