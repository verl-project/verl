import matplotlib.pyplot as plt
import trimesh
from pyrender_utils import PyRenderOffline


def test_render_example():
    """
    Example script to test the PyRenderOffline class with a real mesh.
    Only runs if a GPU/rendering environment is available. See README.md for more details.
    """
    try:
        # Create a high-quality sphere for wireframe rendering
        # Using more subdivisions for smoother appearance
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

        # Create a perfect sphere instead of a noisy one
        # This will look better as a wireframe/blueprint for our test ;)
        print(
            f"Created sphere with {len(sphere.vertices)} vertices and {len(sphere.faces)} faces"
        )
        print(
            f"Sphere has {len(sphere.edges_unique)} unique edges for wireframe rendering"
        )

        renderer = PyRenderOffline(
            width=512, height=512
        )  # larger dimensions than the CLIP size for better detail

        images = renderer.render_mesh_to_images(sphere)

        # Display the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        view_names = ["Front", "Top", "Diagonal"]

        for i, (img, name) in enumerate(zip(images, view_names)):
            axes[i].imshow(img)
            axes[i].set_title(f"{name} View")
            axes[i].axis("off")

        plt.savefig("test_rendered_sphere_views.png")
        plt.close()

        print("Successfully rendered sphere from 3 viewpoints")
        print("Images saved to rendered_sphere_views.png")
        return True
    except Exception as e:
        print(f"Failed to run renderer example: {e}")
        return False


if __name__ == "__main__":
    test_render_example()
