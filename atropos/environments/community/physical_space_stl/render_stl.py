import math
import os
import sys

import bpy

# Get args after --
argv = sys.argv
argv = argv[argv.index("--") + 1 :]  # args after --

input_stl = argv[0]
output_dir = argv[1]

# Clear existing objects
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Import STL
bpy.ops.import_mesh.stl(filepath=input_stl)
obj = bpy.context.selected_objects[0]

# Center the object at origin
bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")
obj.location = (0, 0, 0)

# Add Sun light
sun_light_data = bpy.data.lights.new(name="SunLight", type="SUN")
sun_light_object = bpy.data.objects.new(name="SunLight", object_data=sun_light_data)
sun_light_object.location = (10, 10, 10)
bpy.context.collection.objects.link(sun_light_object)

# Create camera
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Set render resolution
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

# Rendering parameters
angles = [0, 120, 240]  # degrees around Z axis
radius = 10
elevation = 5

for i, angle in enumerate(angles):
    rad = math.radians(angle)
    cam_x = radius * math.cos(rad)
    cam_y = radius * math.sin(rad)
    cam_z = elevation
    cam_obj.location = (cam_x, cam_y, cam_z)

    # Point camera to object center (0,0,0)
    direction = -cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Render
    bpy.context.scene.render.filepath = os.path.join(output_dir, f"render_{i}.png")
    bpy.ops.render.render(write_still=True)
