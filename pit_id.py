from modules.dataset_tools import mhio, coordinates
import numpy as np
from collections import defaultdict

"""
This script processes coordinates from FCN and DCC false positive lists to categorize 
them into shared clusters and unique points. IDs are assigned to organize and analyze 
overlapping and unique points.

ID SYSTEMS:
1. `Cxxxx`: Clean pit coordinates, unique to each point.
2. `FADxxxx`: Shared false positives in FCN and DCC lists within a tolerance of 15 pixels. Last 4 digits is shared between a cluster for FAD and DAF.
3. `DAFxxxx`: Shared false positives in DCC and FCN lists within a tolerance of 15 pixels.
4. `FMDxxxx`: Unique false positives in FCN list (not in DCC).
5. `DMFxxxx`: Unique false positives in DCC list (not in FCN).
"""

# Step 1: Load the list of clean pit coordinates
# Assign unique IDs prefixed with 'C' (e.g., "C0001") for clean pits
common_list = mhio.load_json("Aligned_clean.txt")
common_list_with_ids = [{"id": f"C{str(idx).zfill(4)}", "coordinate": coord} for idx, coord in enumerate(common_list)]

# Step 2: Load the false positives list from FCN and DCC methods
# Each list contains the coordinates of false positives detected by the respective method
false_pos_list_FCN = mhio.load_json("predicted_list/FCN/aligned/final_false_pos.txt")
false_pos_list_DCC = mhio.load_json("predicted_list/DCC/aligned/final_false_pos.txt")

# Step 3: Identify overlaps between FCN and DCC false positives within a tolerance
tolerance = 15  # Maximum distance for two points to be considered overlapping
overlap_checker = coordinates.Overlap(false_pos_list_FCN, false_pos_list_DCC, tolerance)
overlap_pairs = np.where(overlap_checker.dist < tolerance)

# Extract indices of overlapping points in FCN and DCC lists
common_ids_FCN = overlap_pairs[0]
common_ids_DCC = overlap_pairs[1]

# Step 4: Build connectivity graph
# Use a dictionary to track connected components of overlapping points
graph = defaultdict(set)

# Add edges for overlapping pairs between FCN and DCC lists
for fcn_idx, dcc_idx in zip(common_ids_FCN, common_ids_DCC):
    graph[fcn_idx].add(len(false_pos_list_FCN) + dcc_idx)  # Map DCC indices uniquely
    graph[len(false_pos_list_FCN) + dcc_idx].add(fcn_idx)  # Add reverse connection

# Step 5: Find connected components (clusters of overlapping points)
visited = set()
clusters = []

def dfs(node, cluster):
    """Depth-First Search to find all connected nodes in the graph."""
    visited.add(node)
    cluster.append(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, cluster)

# Perform DFS for all unvisited nodes to find clusters
for node in graph:
    if node not in visited:
        cluster = []
        dfs(node, cluster)
        clusters.append(cluster)

# Step 6: Assign shared IDs to clusters
cluster_ids = {}
for idx, cluster in enumerate(clusters):
    cluster_id = f"{str(idx).zfill(4)}"  # Shared 4-digit cluster ID
    for node in cluster:
        cluster_ids[node] = cluster_id

# Step 7: Generate shared IDs for `FAD` and `DAF`
# FAD: FCN points that are part of overlapping clusters
common_fpf = []
for i in range(len(false_pos_list_FCN)):
    if i in cluster_ids:
        common_fpf.append({"id": f"FAD{cluster_ids[i]}", "coordinate": false_pos_list_FCN[i]})

# DAF: DCC points that are part of overlapping clusters
common_fpd = []
for i in range(len(false_pos_list_DCC)):
    node_idx = len(false_pos_list_FCN) + i  # Map DCC index uniquely
    if node_idx in cluster_ids:
        common_fpd.append({"id": f"DAF{cluster_ids[node_idx]}", "coordinate": false_pos_list_DCC[i]})

# Step 8: Generate IDs for unique points in FCN and DCC lists
# FMD: Points in FCN but not in DCC
unique_fcn_ids = set(range(len(false_pos_list_FCN))) - set(common_ids_FCN)
unique_fcn_fps = [{"id": f"FMD{str(idx).zfill(4)}", "coordinate": false_pos_list_FCN[i]}
                  for idx, i in enumerate(unique_fcn_ids)]

# DMF: Points in DCC but not in FCN
unique_dcc_ids = set(range(len(false_pos_list_DCC))) - set(common_ids_DCC)
unique_dcc_fps = [{"id": f"DMF{str(idx).zfill(4)}", "coordinate": false_pos_list_DCC[i]}
                  for idx, i in enumerate(unique_dcc_ids)]

# Step 9: Save results to JSON files
mhio.save_json(common_fpf, "predicted_list/ids/FP_FCN_common_DCC_with_ids.txt")
mhio.save_json(common_fpd, "predicted_list/ids/FP_DCC_common_FCN_with_ids.txt")
mhio.save_json(unique_fcn_fps, "predicted_list/ids/FP_unique_FCN_with_ids.txt")
mhio.save_json(unique_dcc_fps, "predicted_list/ids/FP_unique_DCC_with_ids.txt")
mhio.save_json(common_list_with_ids, "predicted_list/ids/Aligned_clean_with_ids.txt")

# Summary of results
print(f"Number of FAD (intersection) FPs in FCN: {len(common_fpf)}")
print(f"Number of FAD (intersection) FPs in DCC: {len(common_fpd)}")
print(f"Number of FMD (FCN minus DCC) FPs: {len(unique_fcn_fps)}")
print(f"Number of DMF (DCC minus FCN) FPs: {len(unique_dcc_fps)}")

print(common_list_with_ids)
