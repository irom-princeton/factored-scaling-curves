import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from trimesh.creation import box, cone, cylinder
from trimesh.visual.material import PBRMaterial

ASSET_PATH = os.path.expanduser(
    "~/.maniskill/data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
)


def get_obj_asset(obj_name):
    # List to store the paths of matching .glb files
    matching_files = []
    # Convert the obj_name to lowercase for case-insensitive comparison
    obj_name_lower = obj_name.lower()
    # Loop through all files in the ASSET_PATH directory
    for file in os.listdir(ASSET_PATH):
        # Check if the file contains the obj_name (case-insensitive) and has a .glb extension
        if obj_name_lower in file.lower() and file.endswith(".glb"):
            # Append the full path of the matching file to the list
            matching_files.append(os.path.join(ASSET_PATH, file))
    return matching_files


def generate_table_obj(
    cfg_path="guided_dc/cfg/simulation/pick_and_place.yaml",
    visualize=False,
    image_path="guided_dc/assets/table/Slide1.png",
    # image_path="extended_image.png",
):
    #########################
    #                       #
    #    Generate Meshes    #
    #                       #
    #########################

    config = OmegaConf.load(cfg_path)
    table_config = config.env.scene_builder.table
    table_top_thickness = table_config.thickness
    table_top_width = table_config.width
    table_top_length = table_config.length
    leg_length = table_config.leg_length
    leg_thickness = table_config.leg_thickness

    # Create the table top
    top_mesh = trimesh.creation.box(
        extents=(table_top_length, table_top_width, table_top_thickness)
    )
    # Create the table legs (4 legs)
    legs = []
    leg_positions = [
        (
            -table_top_length / 2 + leg_thickness / 2,
            -table_top_width / 2 + leg_thickness / 2,
        ),  # Front-left
        (
            -table_top_length / 2 + leg_thickness / 2,
            table_top_width / 2 - leg_thickness / 2,
        ),  # Back-left
        (
            table_top_length / 2 - leg_thickness / 2,
            -table_top_width / 2 + leg_thickness / 2,
        ),  # Front-right
        (
            table_top_length / 2 - leg_thickness / 2,
            table_top_width / 2 - leg_thickness / 2,
        ),  # Back-right
    ]
    for pos in leg_positions:
        leg = trimesh.creation.box(extents=(leg_thickness, leg_thickness, leg_length))
        leg.apply_translation(
            [pos[0], pos[1], -table_top_thickness / 2 - leg_length / 2]
        )
        legs.append(leg)

    table = trimesh.util.concatenate([top_mesh, *legs])

    #########################
    #                       #
    #      Attach images    #
    #                       #
    #########################

    im = Image.open(image_path)

    def crop_to_ratio(image, target_ratio):
        """
        Crop an image's height to match the specified aspect ratio.
        The width remains untouched.

        Args:
        - image (str or PIL.Image.Image): The input image or its file path.
        - target_ratio (float): The desired width-to-height aspect ratio (e.g., 2/1 for 2:1).

        Returns:
        - PIL.Image.Image: The cropped image.
        """

        # Load the image if a path is given
        if isinstance(image, str):
            image = Image.open(image)

        # Get current image size
        width, height = image.size

        # Calculate the desired height based on the target aspect ratio and current width
        desired_height = int(width / target_ratio)

        if desired_height < height:
            # Crop the height
            top = (height - desired_height) // 2
            bottom = top + desired_height
        else:
            # No cropping needed if the image is already shorter
            top = 0
            bottom = height

        # Crop the image (only height is adjusted)
        cropped_image = image.crop((0, top, width, bottom))

        return cropped_image

    cropped_image = crop_to_ratio(im, table_top_length / table_top_width)

    if visualize:
        plt.imshow(np.array(im))
        plt.axis("off")  # Hide axis
        plt.show()
        plt.imshow(np.array(cropped_image))
        plt.axis("off")  # Hide axis
        plt.show()

    # Define UV coordinates for the top face of the tabletop
    # Assuming that the top face vertices are known

    top_z = table_top_thickness / 2  # Top surface lies at this z value

    # Step 2: Filter faces with vertices whose average z-coordinate matches the top_z
    top_faces = []
    for face in top_mesh.faces:
        face_vertices = top_mesh.vertices[face]
        avg_z = np.mean(
            face_vertices[:, 2]
        )  # Calculate the average z value for the face
        if np.isclose(avg_z, top_z):  # Check if the face lies on the top surface
            top_faces.append(face)

    # Step 3: Identify unique vertices used by the top faces
    top_face_idx = np.unique(np.array(top_faces).flatten())

    # Step 4: Create a blank UV map
    uv_map = np.zeros((len(table.vertices), 2))

    # Step 5: Assign the UV coordinates to the top face vertices
    top_uv = np.array(
        [
            [0, 0],  # bottom left
            [0, 1],  # bottom right
            [1, 0],  # top right
            [1, 1],  # top left
        ]
    )

    # Map the UV coordinates to the top face vertices
    # Assuming top_uv corresponds to the 4 vertices of the top face
    uv_map[top_face_idx[:4]] = top_uv

    # # Example usage
    # top_uv = np.array(
    #     [
    #         [0, 0],  # bottom left
    #         [0, 1],  # bottom right
    #         [1, 0],  # top right
    #         [1, 1],  # top left
    #     ]
    # )

    # # Randomize the UV coordinates
    # # randomized_uv = randomize_uv(top_uv, image_width / image_height, shift_range=0.1)

    # # Assign the texture only to the top face of the tabletop
    # # Identify the top face (in this case, the face with the highest z value)
    # top_face_idx = np.where(
    #     np.isclose(top_mesh.vertices[:, 2], table_top_thickness / 2)
    # )[0]

    # breakpoint()
    # # Create a blank UV map for the entire table
    # uv_map = np.zeros((len(table.vertices), 2))

    # # Assign the UV coordinates to the top face vertices
    # uv_map[top_face_idx] = top_uv

    # # # Create a PBR material for reflectiveness
    pbr_material = PBRMaterial(
        name="smooth_paper_wrap",
        baseColorTexture=cropped_image,  # Texture image for the table surface,
        # metallicFactor=0.2,  # Non-metallic (paper is a dielectric material)
        # roughnessFactor=0.5,  # Low roughness for a smooth paper-like finish
        # baseColorFactor=[0.1] * 4,  # White base color
    )

    # Assign the texture and UV map to the mesh
    table.visual = trimesh.visual.TextureVisuals(
        uv=uv_map, image=cropped_image, material=pbr_material
    )
    # Apply the translation to the mesh
    table.apply_translation([0, 0, -table_top_thickness / 2])

    #########################
    #                       #
    # Export and visualize  #
    #                       #
    #########################

    # Export the table with texture

    table.export("guided_dc/assets/table/table_with_textured_top.obj")
    if visualize:
        table.show()


def generate_plate():
    # Plate dimensions
    bottom_radius = 5  # 10 cm diameter / 2
    top_radius = 7.5  # 15 cm diameter / 2
    depth = 5  # cm (height of the plate)

    # Outer cone: Larger frustum for the plate
    outer_frustum = cone(radius=top_radius, height=depth, sections=100)

    # Inner cone: Smaller frustum to make the plate hollow
    inner_frustum = cone(radius=bottom_radius, height=depth - 0.5, sections=100)
    inner_frustum.apply_translation((0, 0, 0.25))  # Slightly offset for wall thickness

    # Subtract inner cone from outer cone to create a hollow frustum
    plate = outer_frustum.difference(inner_frustum)

    # Align the origin of the plate at the bottom center
    plate.apply_translation((0, 0, depth / 2))

    # Handle dimensions
    handle_length = 7  # cm
    handle_width = 3  # cm
    handle_thickness = 0.5  # cm
    handle_curve_radius = handle_width / 2  # Radius of the rounded tip

    # Create the straight part of the handle
    straight_handle = box(
        extents=(
            handle_thickness,
            handle_length - handle_curve_radius,
            handle_thickness,
        )
    )

    # Create the rounded tip of the handle
    rounded_tip = cylinder(
        radius=handle_curve_radius, height=handle_thickness, sections=50
    )
    rounded_tip.apply_translation(
        (0, (handle_length - handle_curve_radius) / 2, 0)
    )  # Position at the tip

    # Combine the straight part and the rounded tip
    handle = straight_handle.union(rounded_tip)

    # Position the handle at the middle depth of the plate
    handle.apply_translation((0, top_radius + handle_thickness / 2, -depth / 2))

    # Attach the handle to the plate
    final_model = plate.union(handle)

    # Add a metallic appearance
    color = [192, 192, 192, 255]  # Silver color (RGBA)
    visuals = trimesh.visual.ColorVisuals(mesh=final_model, vertex_colors=color)
    final_model.visual = visuals

    # Export to GLB
    final_model.export("guided_dc/assets/table/plate_with_handle.glb")

    print("Plate model saved as 'plate_with_handle.glb'")


if __name__ == "__main__":
    generate_table_obj(visualize=True)
    # generate_plate()
