import os
from modules.dataset_tools import mhio, coordinates
from modules.compare.panels import create_image_panel
import numpy as np

"""
This script processes and visualizes clean pits, false positives (FAD, DAF, FMD, DMF), and clusters of overlapping points. 
It generates image panels for analysis, highlighting key coordinates and saving them in categorized folders.

## Key Features:
1. Panels for clean pits showing nearby FCN and DCC predictions.
2. Panels for clusters of overlapping FAD and DAF points.
3. Panels for unique FCN false positives (FMD) and unique DCC false positives (DMF).

## Folder Structure:
- `gallery/clean_pits`: Panels for clean pits.
- `gallery/false_positive/intersection`: Panels for overlapping clusters (FAD and DAF).
- `gallery/false_positive/FMD`: Panels for unique FCN false positives.
- `gallery/false_positive/DMF`: Panels for unique DCC false positives.

## ID Systems:
- `FADxxxx`: FCN points overlapping with DCC points (clusters).
- `DAFxxxx`: DCC points overlapping with FCN points (clusters).
- `FMDxxxx`: Unique FCN false positives.
- `DMFxxxx`: Unique DCC false positives.
"""

# Define paths for input data and output directories
clean_pits_path = "predicted_list/ids/Aligned_clean_with_ids.txt"
predicted_fcn_path = "predicted_list/FCN/aligned/final_pred_list.txt"
predicted_dcc_path = "predicted_list/DCC/aligned/final_pred_list.txt"
fad_ids_path = "predicted_list/ids/FP_FCN_common_DCC_with_ids.txt"
daf_ids_path = "predicted_list/ids/FP_DCC_common_FCN_with_ids.txt"
fmd_ids_path = "predicted_list/ids/FP_unique_FCN_with_ids.txt"
dmf_ids_path = "predicted_list/ids/FP_unique_DCC_with_ids.txt"

output_folder_clean = "gallery/56pixel/clean_pits"
output_folder_intersection = "gallery/56pixel/false_positive/intersection"
output_folder_fmd = "gallery/56pixel/false_positive/FMD"
output_folder_dmf = "gallery/56pixel/false_positive/DMF"

# Tolerance for identifying nearby points
tolerance = 15

# Ensure all output folders exist
os.makedirs(output_folder_clean, exist_ok=True)
os.makedirs(output_folder_intersection, exist_ok=True)
os.makedirs(output_folder_fmd, exist_ok=True)
os.makedirs(output_folder_dmf, exist_ok=True)

# Step 1: Load clean pits with IDs
clean_pits = mhio.load_json(clean_pits_path)

# Step 2: Load predicted FCN and DCC pits
predicted_fcn = mhio.load_json(predicted_fcn_path)
predicted_dcc = mhio.load_json(predicted_dcc_path)

# --------------------------
# Panels for Clean Pits
# --------------------------

crop_size = 28*2
for clean_pit in clean_pits:
    primary_id = clean_pit["id"]
    primary_coord = clean_pit["coordinate"]


    # Identify nearby FCN and DCC predictions within the tolerance
    fcn_overlap_checker = coordinates.Overlap(
        np.array(predicted_fcn), np.array([primary_coord]), tolerance
    )
    dcc_overlap_checker = coordinates.Overlap(
        np.array(predicted_dcc), np.array([primary_coord]), tolerance
    )

    extra_fcn_coords = [
        predicted_fcn[idx]
        for idx, dist in enumerate(fcn_overlap_checker.dist[:, 0])
        if dist < tolerance
    ]
    extra_dcc_coords = [
        predicted_dcc[idx]
        for idx, dist in enumerate(dcc_overlap_checker.dist[:, 0])
        if dist < tolerance
    ]

    # Create a descriptive panel for the clean pit
    description = "Extra coordinates:\n"
    if extra_fcn_coords:
        description += f"FCN ({len(extra_fcn_coords)} points): {extra_fcn_coords}\n"
    if extra_dcc_coords:
        description += f"DCC ({len(extra_dcc_coords)} points): {extra_dcc_coords}\n"

    # Save the panel
    create_image_panel(
        global_coord=primary_coord,
        save_folder=output_folder_clean,
        title=f"{primary_id} {primary_coord}",
        description=description,
        extra_fcn_coords=extra_fcn_coords,
        extra_dcc_coords=extra_dcc_coords,
        crop_size=crop_size,
    )

print(f"Panels saved to {output_folder_clean}")

# --------------------------
# Panels for FAD and DAF Clusters
# --------------------------
# Load FAD and DAF ID lists
fad_ids = mhio.load_json(fad_ids_path)
daf_ids = mhio.load_json(daf_ids_path)

# Group clusters by their last 4 digits (cluster ID)
clusters = {}

for item in fad_ids + daf_ids:
    cluster_id = item["id"][-4:]
    if cluster_id not in clusters:
        clusters[cluster_id] = {"fcn_coords": [], "dcc_coords": []}
    if "FAD" in item["id"]:
        clusters[cluster_id]["fcn_coords"].append(item["coordinate"])
    elif "DAF" in item["id"]:
        clusters[cluster_id]["dcc_coords"].append(item["coordinate"])

# Generate a panel for each cluster
for cluster_id, coords in clusters.items():
    fcn_coords = coords["fcn_coords"]
    dcc_coords = coords["dcc_coords"]

    # Create a description for the cluster
    description = "Cluster Details:\n"
    if fcn_coords:
        description += f"FAD Points ({len(fcn_coords)}): {fcn_coords}\n"
    if dcc_coords:
        description += f"DAF Points ({len(dcc_coords)}): {dcc_coords}\n"

    # Save the cluster panel
    create_image_panel(
        global_coord=None,
        save_folder=output_folder_intersection,
        title=f"false_positive_cluster_{cluster_id}",
        description=description,
        extra_fcn_coords=fcn_coords,
        extra_dcc_coords=dcc_coords,
        crop_size=crop_size,
    )

print(f"Cluster panels saved to {output_folder_intersection}")

# --------------------------
# Panels for Unique FMD Points
# --------------------------
# Load FMD ID list
fmd_ids = mhio.load_json(fmd_ids_path)

for fmd_item in fmd_ids:
    fmd_id = fmd_item["id"]
    fmd_coord = fmd_item["coordinate"]

    # Save the FMD panel
    create_image_panel(
        global_coord=None,
        save_folder=output_folder_fmd,
        title=f"{fmd_id} {fmd_coord}",
        extra_fcn_coords=[fmd_coord],  # Only FMD coordinate
        extra_dcc_coords=None,
        crop_size=crop_size,
    )

print(f"Panels for FMD points saved to {output_folder_fmd}")

# --------------------------
# Panels for Unique DMF Points
# --------------------------
# Load DMF ID list
dmf_ids = mhio.load_json(dmf_ids_path)

for dmf_item in dmf_ids:
    dmf_id = dmf_item["id"]
    dmf_coord = dmf_item["coordinate"]

    # Save the DMF panel
    create_image_panel(
        global_coord=None,
        save_folder=output_folder_dmf,
        title=f"{dmf_id} {dmf_coord}",
        extra_fcn_coords=None,
        extra_dcc_coords=[dmf_coord],  # Only DMF coordinate
        crop_size=crop_size,
    )

print(f"Panels for DMF points saved to {output_folder_dmf}")
