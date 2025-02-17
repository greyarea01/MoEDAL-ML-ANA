import os
import numpy as np
import matplotlib.pyplot as plt
from modules.dataset_tools import coordinates

def crop_image_with_boundary_and_padding(img, x, y, crop_size):
    """
    Crop an image around a specified point, adjusting for boundaries, with padding if needed.

    Parameters:
        img (np.ndarray): A 2D image (height x width) to crop from.
        x (int): The x-coordinate of the center point.
        y (int): The y-coordinate of the center point.
        crop_size (int): The size of the cropped image (crop_size x crop_size).

    Returns:
        np.ndarray: The cropped and padded image.
    """
    half_size = crop_size // 2

    # Calculate boundaries with adjustments for edges
    x_start = max(0, x - half_size)
    x_end = min(img.shape[1], x + half_size)  # Width is the second dimension
    y_start = max(0, y - half_size)
    y_end = min(img.shape[0], y + half_size)  # Height is the first dimension

    # Crop the image
    cropped = img[y_start:y_end, x_start:x_end]

    # Pad if necessary
    pad_x_start = max(0, half_size - x)
    pad_x_end = max(0, (x + half_size) - img.shape[1])
    pad_y_start = max(0, half_size - y)
    pad_y_end = max(0, (y + half_size) - img.shape[0])

    # Add padding to the cropped region if it exceeds the image boundaries
    padded = np.pad(
        cropped,
        ((pad_y_start, pad_y_end), (pad_x_start, pad_x_end)),
        mode='constant',
        constant_values=0
    )

    return padded


def create_image_panel(global_coord=None, save_folder="gallery", title="Image Panel", description=None,
                       extra_fcn_coords=None, extra_dcc_coords=None,
                       clean_front_path="numpy_data_eval/preprocessed_clean_eval.npy",
                       clean_back_path="numpy_data_eval/preprocessed_clean_reverse_eval.npy",
                       exposed_front_path="numpy_data_eval/preprocessed_dirty_eval.npy",
                       exposed_back_path="numpy_data_eval/preprocessed_dirty_reverse_eval.npy",
                       crop_size=28):
    """
    Create a panel of 4 sub-images given a global coordinate.

    Parameters:
        global_coord (tuple or None): Global coordinates (x, y) for the primary point of interest. If None, a coordinate
                                      from `extra_fcn_coords` or `extra_dcc_coords` will be used.
        save_folder (str): Folder to save the output image panel.
        title (str): Title for the image panel.
        description (str, optional): Additional description to include below the panel. Defaults to None.
        extra_fcn_coords (list, optional): List of extra global coordinates to plot on clean_front with a blue cross. Defaults to None.
        extra_dcc_coords (list, optional): List of extra global coordinates to plot on clean_front with a red cross. Defaults to None.
        clean_front_path (str): Path to the NumPy array for clean front image data. Default is "numpy_data_eval/preprocessed_clean_eval.npy".
        clean_back_path (str): Path to the NumPy array for clean back image data. Default is "numpy_data_eval/preprocessed_clean_reverse_eval.npy".
        exposed_front_path (str): Path to the NumPy array for exposed front image data. Default is "numpy_data_eval/preprocessed_dirty_eval.npy".
        exposed_back_path (str): Path to the NumPy array for exposed back image data. Default is "numpy_data_eval/preprocessed_dirty_reverse_eval.npy".
        crop_size (int): The size of the cropped image (crop_size x crop_size).

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(save_folder, exist_ok=True)

    # Check if global_coord is None and choose a fallback coordinate
    is_primary_provided = global_coord is not None
    if not is_primary_provided:
        if extra_fcn_coords and len(extra_fcn_coords) > 0:
            global_coord = extra_fcn_coords[0]
        elif extra_dcc_coords and len(extra_dcc_coords) > 0:
            global_coord = extra_dcc_coords[0]
        else:
            raise ValueError("No primary or extra coordinates provided for cropping.")

    # Load all necessary image data
    image_data = {
        'clean_front': np.load(clean_front_path),
        'clean_back': np.load(clean_back_path),
        'exposed_front': np.load(exposed_front_path)[:, :, :, 8],  # Extracting the 8th channel
        'exposed_back': np.load(exposed_back_path)[:, :, :, 8],   # Extracting the 8th channel
    }

    # Initialize the coordinate transformation system
    CS = coordinates.Coord_system()
    CS['c1'].flipped = False
    CS['c1'].reverse = False
    CS['cr'].flipped = True
    CS['cr'].reverse = True
    CS['dr'].flipped = True
    CS['dr'].reverse = True

    # Convert global coordinates to local coordinates for each dataset
    clean_front_ixy = CS['c1'].get_ixy(global_coord)
    clean_back_ixy = CS.convert(clean_front_ixy, 'ccr0')
    exposed_front_ixy = CS.convert(clean_front_ixy, 'cd')
    exposed_back_ixy = CS.convert(clean_front_ixy, 'cdr')

    # Adjust indices to align with dataset specifications
    clean_front_ixy = tuple([clean_front_ixy[0] - 251] + list(clean_front_ixy[1:]))
    clean_back_ixy = tuple([clean_back_ixy[0] - 251] + list(clean_back_ixy[1:]))
    exposed_front_ixy = tuple([exposed_front_ixy[0] - 251] + list(exposed_front_ixy[1:]))
    exposed_back_ixy = tuple([exposed_back_ixy[0] - 251] + list(exposed_back_ixy[1:]))

    # Prepare mapping for image data and local coordinates
    local_coords = {
        'clean_front': clean_front_ixy,
        'clean_back': clean_back_ixy,
        'exposed_front': exposed_front_ixy,
        'exposed_back': exposed_back_ixy,
    }

    # Crop images based on local coordinates
    cropped_images = {}
    for key, img_data in image_data.items():
        i, x, y = local_coords[key]
        cropped_images[key] = crop_image_with_boundary_and_padding(img_data[i], x, y, crop_size)

    # Create a figure with 4 subplots for the image panel
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), gridspec_kw={'bottom': 0.3})
    scan_names = ['exposed_front', 'exposed_back', 'clean_front', 'clean_back']

    # Display each cropped image in its corresponding subplot
    for i, scan_name in enumerate(scan_names):
        axs[i].imshow(cropped_images[scan_name], cmap='gray')
        axs[i].set_title(scan_name.replace('_', ' ').capitalize())
        axs[i].axis('off')

    # Highlight the primary coordinate in the clean front image (if provided)
    clean_front_ax = axs[2]
    if is_primary_provided:
        clean_front_ax.plot(crop_size // 2, crop_size // 2, 'ro', label="Primary Coordinate")

    # Annotate extra FCN and DCC coordinates on clean front
    if extra_fcn_coords or extra_dcc_coords:
        if extra_fcn_coords:
            for coord in extra_fcn_coords:
                extra_ixy = CS['c1'].get_ixy(coord)
                ex, ey = extra_ixy[1] - clean_front_ixy[1] + crop_size // 2, extra_ixy[2] - clean_front_ixy[2] + crop_size // 2
                clean_front_ax.plot(ex, ey, 'bx', label="FCN Coordinate")
        if extra_dcc_coords:
            for coord in extra_dcc_coords:
                extra_ixy = CS['c1'].get_ixy(coord)
                ex, ey = extra_ixy[1] - clean_front_ixy[1] + crop_size // 2, extra_ixy[2] - clean_front_ixy[2] + crop_size // 2
                clean_front_ax.plot(ex, ey, 'rx', label="DCC Coordinate")

        # Add a legend for extra coordinates
        clean_front_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    # Set the title
    fig.suptitle(title, fontsize=16)

    # Add description below the plots
    if description:
        plt.figtext(0.5, 0.02, description, wrap=True, horizontalalignment='center', fontsize=12)

    # Save the panel as an image
    save_path = os.path.join(save_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Panel saved at {save_path}")




# Example usage
create_image_panel(
    global_coord=[6171, -206],
    save_folder="gallery/scratch",
    title="[6171, -206]",
    description="Example image panel with legend and adjustable crop size.",
    crop_size=64
)
