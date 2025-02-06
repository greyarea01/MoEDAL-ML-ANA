import os
import math
import numpy as np
import matplotlib.pyplot as plt
from modules.dataset_tools import coordinates
import gc
from tqdm import tqdm  # For progress tracking
from skimage.measure import block_reduce  # For optional downscaling


def convert_global_to_local(global_coordinate, dataset, coord_system):
	"""
	Converts a global coordinate to the corresponding local coordinate
	for the specified dataset using the provided coordinate system.

	Parameters:
		global_coordinate (tuple): (x, y) global coordinate.
		dataset (str): One of "clean_front", "clean_back", "exposed_front", "exposed_back".
		coord_system: An instance of the coordinate system (CS).

	Returns:
		tuple: A tuple (tile_index, local_row_offset, local_col_offset).
	"""
	# First, obtain the clean front mapping.
	clean_front_ixy = coord_system['c1'].get_ixy(global_coordinate)
	# Use the appropriate conversion based on the dataset.
	if dataset == "clean_front":
		return clean_front_ixy
	elif dataset == "clean_back":
		return coord_system.convert(clean_front_ixy, 'ccr0')
	elif dataset == "exposed_front":
		return coord_system.convert(clean_front_ixy, 'cd')
	elif dataset == "exposed_back":
		return coord_system.convert(clean_front_ixy, 'cdr')
	else:
		raise ValueError(f"Invalid dataset: {dataset}.")


def stitch_images_optimized(global_x_range, global_y_range, dataset_key, output_dir,
                            block_size=300, scale_factor=1,
                            clean_front_path="numpy_data_eval/preprocessed_clean_eval.npy",
                            clean_back_path="numpy_data_eval/preprocessed_clean_reverse_eval.npy",
                            exposed_front_path="numpy_data_eval/preprocessed_dirty_eval.npy",
                            exposed_back_path="numpy_data_eval/preprocessed_dirty_reverse_eval.npy"):
	"""
	Optimized stitching function that creates a composite image for a specified range
	of global coordinates by mapping each global coordinate to its corresponding local
	coordinate and extracting the appropriate pixel from the dataset's image data.

	The processing is performed in blocks to reduce memory overhead, and an optional
	downscaling factor can be applied to the final image.

	Parameters:
		global_x_range (tuple): (x_min, x_max) global x-coordinate range.
		global_y_range (tuple): (y_min, y_max) global y-coordinate range.
		dataset_key (str): One of "clean_front", "clean_back", "exposed_front", "exposed_back".
		output_dir (str): Directory where the stitched image will be saved.
		block_size (int): Size (in pixels) of the processing block (default: 300).
		scale_factor (int): Factor to scale down the output image (default: 1, no scaling).
		clean_front_path (str): File path for the clean front dataset.
		clean_back_path (str): File path for the clean back dataset.
		exposed_front_path (str): File path for the exposed front dataset.
		exposed_back_path (str): File path for the exposed back dataset.

	Returns:
		None
	"""
	# Map the dataset key to the corresponding file path.
	dataset_paths = {
		'clean_front': clean_front_path,
		'clean_back': clean_back_path,
		'exposed_front': exposed_front_path,
		'exposed_back': exposed_back_path,
	}
	if dataset_key not in dataset_paths:
		raise ValueError(f"Invalid dataset_key: {dataset_key}. Choose from {list(dataset_paths.keys())}.")

	# Load the image data.
	image_data = np.load(dataset_paths[dataset_key])
	# For exposed datasets, use only the 8th channel.
	if dataset_key in ['exposed_front', 'exposed_back']:
		image_data = image_data[:, :, :, 8]

	# Initialize the coordinate transformation system.
	CS = coordinates.Coord_system()
	CS['c1'].flipped = False
	CS['c1'].reverse = False
	CS['cr'].flipped = True
	CS['cr'].reverse = True
	CS['dr'].flipped = True
	CS['dr'].reverse = True

	# Determine the dimensions of the global canvas.
	x_min, x_max = global_x_range
	y_min, y_max = global_y_range
	canvas_width = x_max - x_min + 1
	canvas_height = y_max - y_min + 1

	# Create the output canvas (stitched image) and a boolean mask to track filled pixels.
	stitched_image = np.zeros((canvas_height, canvas_width), dtype=np.float32)
	filled_mask = np.zeros((canvas_height, canvas_width), dtype=bool)

	# Calculate the number of processing blocks.
	nblocks_x = math.ceil(canvas_width / block_size)
	nblocks_y = math.ceil(canvas_height / block_size)
	total_blocks = nblocks_x * nblocks_y

	# Initialize the progress bar.
	pbar = tqdm(total=total_blocks, desc="Processing blocks", unit="block")

	# Process the canvas block by block.
	for bx in range(x_min, x_max + 1, block_size):
		for by in range(y_min, y_max + 1, block_size):
			# Define block boundaries.
			block_x_end = min(x_max, bx + block_size - 1)
			block_y_end = min(y_max, by + block_size - 1)

			# Dictionary to group pixel mappings by tile index.
			# Key: tile index, Value: list of tuples (global_x, global_y, local_row, local_col)
			block_mappings = {}

			# Iterate over all global coordinates within the current block.
			for gx in range(bx, block_x_end + 1):
				for gy in range(by, block_y_end + 1):
					cx = gx - x_min  # x-coordinate on the stitched canvas
					cy = gy - y_min  # y-coordinate on the stitched canvas
					# Skip if this pixel has already been filled.
					if filled_mask[cy, cx]:
						continue

					# Convert the global coordinate to local coordinates.
					global_coord = (gx, gy)
					# Use "clean_front" conversion as the base.
					local_ixy = convert_global_to_local(global_coord, "clean_front", CS)
					# Adjust the tile index by subtracting 251 as per dataset specification.
					local_ixy = (local_ixy[0] - 251, local_ixy[1], local_ixy[2])
					tile_index, local_row, local_col = local_ixy

					# Skip if the computed tile index or local offsets are negative.
					if tile_index < 0 or tile_index >= image_data.shape[0] or local_row < 0 or local_col < 0:
						continue

					# Group the mapping by tile index.
					if tile_index not in block_mappings:
						block_mappings[tile_index] = []
					block_mappings[tile_index].append((gx, gy, local_row, local_col))

			# Process each group of mappings within the block.
			for tile_index, mappings in block_mappings.items():
				for mapping in mappings:
					gx, gy, local_row, local_col = mapping
					cx = gx - x_min
					cy = gy - y_min
					# Check if local offsets are within the bounds of the tile image.
					if local_col < image_data.shape[1] and local_row < image_data.shape[2]:
						# Retrieve the pixel from the tile image.
						stitched_image[cy, cx] = image_data[tile_index][local_col, local_row]
						filled_mask[cy, cx] = True

			gc.collect()
			pbar.update(1)
	pbar.close()

	# Optional: Downscale the stitched image if a scale factor greater than 1 is specified.
	if scale_factor > 1:
		stitched_image = block_reduce(stitched_image, block_size=(scale_factor, scale_factor), func=np.mean)
		scale_suffix = f"_scaled{scale_factor}"
	else:
		scale_suffix = ""

	# Generate the output filename and title.
	dataset_name = dataset_key.replace('_', ' ').title()
	filename = f"{dataset_key}_{x_min}_{x_max}_{y_min}_{y_max}{scale_suffix}.png"
	title = f"Stitched Image ({dataset_name}): x=({x_min}-{x_max}), y=({y_min}-{y_max}), scale_factor={scale_factor}"

	# Ensure the output directory exists.
	os.makedirs(output_dir, exist_ok=True)
	save_path = os.path.join(output_dir, filename)

	# Save the stitched image without extra margins.
	plt.imsave(save_path, stitched_image, cmap='gray')
	print(f"Optimized stitched image saved to {save_path}")


# Example usage:
stitch_images_optimized(
	global_x_range=(8800, 9000),  # Global x-coordinate range.
	global_y_range=(-300, 500),  # Global y-coordinate range.
	dataset_key="clean_front",  # Dataset to use.
	output_dir="stitched_images",  # Directory to save output.
	scale_factor=1  # 1:1 mapping (no scaling).
)

"""
full img range
x range 5800 - 12500
y range -300 - 11500
"""