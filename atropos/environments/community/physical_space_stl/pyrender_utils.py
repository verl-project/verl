import os

import numpy as np
import pyrender
import trimesh

# Headless rendering with GPU acceleration (egl), for non-GPU environments omesa will be faster
os.environ["PYOPENGL_PLATFORM"] = "egl"


def create_look_at_matrix(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """
    Create a look-at transformation matrix for a camera.

    eye - position of the camera
    target - position the camera is looking at
    up - up direction for the camera

    returns the 4x4 transformation matrix
    """
    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)

    # Forward vector (from eye to target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Side vector (right vector)
    side = np.cross(forward, up)
    side = side / np.linalg.norm(side)

    # Recompute up vector to ensure orthogonality
    up = np.cross(side, forward)
    up = up / np.linalg.norm(up)

    # Create the rotation matrix | NOTE: PyRender uses OpenGL convention
    # where camera looks down negative z-axis
    R = np.eye(4)
    R[0, :3] = side
    R[1, :3] = up
    R[2, :3] = -forward

    T = np.eye(4)  # translation matrix
    T[:3, 3] = eye

    # The camera pose matrix is the inverse of the view matrix
    # but we need to return the pose directly for pyrender
    return T @ R


class PyRenderOffline:
    def __init__(self, width=224, height=224):  # Standard CLIP image size
        self.width = width
        self.height = height
        try:
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=self.width, viewport_height=self.height, point_size=1.0
            )
        except Exception as e:
            print(
                f"Failed to initialize OffscreenRenderer (is a display server/EGL/OSMesa available?): {e}"
            )
            print(
                "Try: pip install pyglet; or for headless: export PYOPENGL_PLATFORM=osmesa (or egl)"
            )
            self.renderer = None  # Fallback or raise error
            raise

        # Create camera poses using explicit transformation matrices

        # Front view (looking along the -Z axis)
        front_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])

        # Top view (looking along the -Y axis)
        top_pose = np.array([[1, 0, 0, 0], [0, 0, 1, 5], [0, -1, 0, 0], [0, 0, 0, 1]])

        # Diagonal view (from upper corner)
        side_pose = np.array(
            [
                [0.866, -0.25, 0.433, 3],  # Camera right vector
                [0.0, 0.866, 0.5, 3],  # Camera up vector
                [-0.5, -0.433, 0.75, 5],  # Camera forward vector (pointing at origin)
                [0, 0, 0, 1],
            ]
        )

        # Store camera poses
        self.camera_poses = [front_pose, top_pose, side_pose]

        # Debug print for the camera poses cause it doesn't look right
        # and I'm not a game developer lol
        print("Camera poses:")
        for i, pose in enumerate(self.camera_poses):
            print(f"Camera {i}:\n{pose}")

        # slightly wider field of view to ensure objects are visible
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.5, aspectRatio=1.0)

        # Bright point light
        self.light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    def render_mesh_to_images(self, mesh_obj: trimesh.Trimesh):
        if not self.renderer:
            print("Renderer not initialized, cannot render.")
            return [
                np.zeros((self.height, self.width, 3), dtype=np.uint8) for _ in range(3)
            ]

        print(
            f"Rendering mesh with {len(mesh_obj.vertices)} vertices and {len(mesh_obj.faces)} faces"
        )

        images = []

        # Make a copy to avoid modifying original mesh
        render_mesh = mesh_obj.copy()

        # Center and scale the mesh for visibility
        render_mesh.apply_translation(-render_mesh.centroid)
        scale_factor = 0.8 / np.max(
            render_mesh.extents
        )  # Scale to unit size but slightly smaller
        render_mesh.apply_scale(scale_factor)

        # Create a ground plane
        ground_plane = trimesh.creation.box([4.0, 4.0, 0.01])
        ground_plane.apply_translation([0, 0, -0.5])

        # Color scheme for wireframe blueprint
        blueprint_bg = [0.98, 0.98, 1.0, 1.0]  # Light blue background
        wireframe_color = [0.0, 0.4, 0.8, 1.0]  # Medium bright blue for wireframes
        grid_color = [0.0, 0.2, 0.4, 1.0]  # Darker blue for grid

        # Wireframe material for the edges - completely opaque
        wireframe_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=wireframe_color,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            wireframe=True,
        )

        # Grid material for the ground plane
        grid_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=grid_color,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            wireframe=True,
        )

        for i, pose in enumerate(self.camera_poses):
            # Create a fresh scene with blueprint background
            scene = pyrender.Scene(
                ambient_light=[0.8, 0.8, 0.95], bg_color=blueprint_bg
            )

            # Add ground plane as grid
            plane_mesh = pyrender.Mesh.from_trimesh(
                ground_plane, material=grid_material
            )
            scene.add(plane_mesh)

            # For wireframe rendering, we'll create line segments directly
            edges = render_mesh.edges_unique
            edge_vertices = []
            edge_indices = []

            # Extract all edges for wireframe rendering
            for _, edge in enumerate(edges):
                v0_idx = len(edge_vertices)
                edge_vertices.append(render_mesh.vertices[edge[0]])
                edge_vertices.append(render_mesh.vertices[edge[1]])
                edge_indices.append([v0_idx, v0_idx + 1])

            # Create lines primitive for edges
            if len(edge_vertices) > 0:
                edge_verts = np.array(edge_vertices, dtype=np.float32)
                edge_indices = np.array(edge_indices, dtype=np.uint32)

                # Create a primitive for the lines
                primitive = pyrender.Primitive(
                    positions=edge_verts,
                    indices=edge_indices,
                    mode=pyrender.constants.GLTF.LINES,
                    material=wireframe_material,
                )

                # Create a mesh with just the line primitive
                edge_mesh = pyrender.Mesh(primitives=[primitive])
                scene.add(edge_mesh)

            # Add camera
            scene.add(self.camera, pose=pose)

            # Add light from camera direction (key light)
            scene.add(self.light, pose=pose)

            # Add second light from above for better visibility
            top_light_pose = np.eye(4)
            top_light_pose[1, 3] = 3.0
            scene.add(self.light, pose=top_light_pose)

            try:
                color, _ = self.renderer.render(scene)

                # Post-process to enhance blueprint effect
                # Convert to float for processing
                img_float = color.astype(np.float32) / 255.0

                # Add a subtle grid pattern to the background
                grid_size = 40
                grid_intensity = 0.02

                # Draw faint horizontal grid lines
                for y in range(0, color.shape[0], grid_size):
                    img_float[y : y + 1, :, :] = np.minimum(
                        img_float[y : y + 1, :, :] + grid_intensity, 1.0
                    )

                # Draw faint vertical grid lines
                for x in range(0, color.shape[1], grid_size):
                    img_float[:, x : x + 1, :] = np.minimum(
                        img_float[:, x : x + 1, :] + grid_intensity, 1.0
                    )

                # Convert back to uint8
                processed_img = (img_float * 255).astype(np.uint8)

                images.append(processed_img)
                print(f"Rendered wireframe view {i}")

            except Exception as e:
                print(f"Error during rendering view {i}: {e}")
                images.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))

        return images

    def __del__(self):
        if self.renderer:
            self.renderer.delete()
