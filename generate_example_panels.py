import os
import numpy as np
from modules.dataset_tools import mhio, coordinates
from modules.compare.Example_plots import create_example_panel

# -------------------------------------------------------
# Configuration: paths & parameters
# -------------------------------------------------------
clean_pits_path       = "predicted_list/ids/Aligned_clean_with_ids.txt"
predicted_fcn_path    = "predicted_list/FCN/aligned/final_pred_list.txt"
predicted_dcc_path    = "predicted_list/DCC/aligned/final_pred_list.txt"
predicted_fcn_front_path    = "predicted_etch_pits/FCN/final_pr.txt"
predicted_dcc_front_path    = "predicted_etch_pits/DCC/final_pr.txt"
predicted_fcn_reverse_path = "predicted_etch_pits/FCN/final_reverse_pr.txt"
predicted_dcc_reverse_path = "predicted_etch_pits/DCC/final_reverse_pr.txt"
fad_ids_path          = "predicted_list/ids/FP_FCN_common_DCC_with_ids.txt"
daf_ids_path          = "predicted_list/ids/FP_DCC_common_FCN_with_ids.txt"
fmd_ids_path          = "predicted_list/ids/FP_unique_FCN_with_ids.txt"
dmf_ids_path          = "predicted_list/ids/FP_unique_DCC_with_ids.txt"

output_folder_clean         = "gallery/examples"
output_folder_intersection  = "gallery/examples"
output_folder_fmd           = "gallery/examples"
output_folder_dmf           = "gallery/examples"

tolerance = 15  # pixels

# make sure output dirs exist (create_example_panel will also mkdir)
for d in (output_folder_clean, output_folder_intersection, output_folder_fmd, output_folder_dmf):
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------
# Load all JSON lists
# -------------------------------------------------------
clean_pits       = mhio.load_json(clean_pits_path)
predicted_fcn    = mhio.load_json(predicted_fcn_path)
predicted_dcc    = mhio.load_json(predicted_dcc_path)
predicted_fcn_front   = mhio.load_json(predicted_fcn_front_path)
predicted_dcc_front   = mhio.load_json(predicted_dcc_front_path)
predicted_fcn_reverse = mhio.load_json(predicted_fcn_reverse_path)
predicted_dcc_reverse = mhio.load_json(predicted_dcc_reverse_path)
fad_ids          = mhio.load_json(fad_ids_path)
daf_ids          = mhio.load_json(daf_ids_path)
fmd_ids          = mhio.load_json(fmd_ids_path)
dmf_ids          = mhio.load_json(dmf_ids_path)

# -------------------------------------------------------
# Build clusters for FAD/DAF
# -------------------------------------------------------
clusters = {}
for item in (fad_ids + daf_ids):
    cluster_id = item["id"][-4:]
    if cluster_id not in clusters:
        clusters[cluster_id] = {"fcn_coords": [], "dcc_coords": []}
    if item["id"].startswith("FAD"):
        clusters[cluster_id]["fcn_coords"].append(item["coordinate"])
    else:  # DAF
        clusters[cluster_id]["dcc_coords"].append(item["coordinate"])

# -------------------------------------------------------
# Your list of cases (fill this in)
# -------------------------------------------------------
id_list = ["C0001", "C0025", "C0708", "C0743", "C0874", "C0997", "C1222", "FMD0008", "FMD0012", "FAD0003"]

processed_clusters = set()

# -------------------------------------------------------
# Iterate and generate panels
# -------------------------------------------------------
for case_id in id_list:
    prefix = case_id[:3]

    # ---- CLEAN PIT (Cxxxx) ----
    if case_id.startswith("C"):
        entry = next((c for c in clean_pits if c["id"] == case_id), None)
        if entry is None:
            print(f"[WARN] Clean pit ID {case_id} not found.")
            continue

        primary = entry["coordinate"]
        # find nearby FCN / DCC
        fcn_chk = coordinates.Overlap(np.array(predicted_fcn),
                                            np.array([primary]),
                                            tolerance)
        dcc_chk = coordinates.Overlap(np.array(predicted_dcc),
                                            np.array([primary]),
                                            tolerance)
        fcn_chk_front = coordinates.Overlap(np.array(predicted_fcn_front),
                                      np.array([primary]),
                                      tolerance)
        dcc_chk_front = coordinates.Overlap(np.array(predicted_dcc_front),
                                      np.array([primary]),
                                      tolerance)
        fcn_chk_back = coordinates.Overlap(np.array(predicted_fcn_reverse),
                                            np.array([primary]),
                                            tolerance)
        dcc_chk_back = coordinates.Overlap(np.array(predicted_dcc_reverse),
                                            np.array([primary]),
                                            tolerance)

        extra_fcn = [
            predicted_fcn[i] for i, dist in enumerate(fcn_chk.dist[:,0])
            if dist < tolerance
        ]
        extra_dcc = [
            predicted_dcc[i] for i, dist in enumerate(dcc_chk.dist[:,0])
            if dist < tolerance
        ]
        extra_fcn_front = [
            predicted_fcn_front[i] for i, dist in enumerate(fcn_chk_front.dist[:, 0])
            if dist < tolerance
        ]
        extra_dcc_front = [
            predicted_dcc_front[i] for i, dist in enumerate(dcc_chk_front.dist[:, 0])
            if dist < tolerance
        ]
        extra_fcn_back = [
            predicted_fcn_reverse[i] for i, dist in enumerate(fcn_chk_back.dist[:, 0])
            if dist < tolerance
        ]
        extra_dcc_back = [
            predicted_dcc_reverse[i] for i, dist in enumerate(dcc_chk_back.dist[:, 0])
            if dist < tolerance
        ]

        desc = ""
        if extra_fcn:
            desc += f"FCN ({len(extra_fcn)} pts): {extra_fcn}\n"
        if extra_dcc:
            desc += f"DCC ({len(extra_dcc)} pts): {extra_dcc}\n"

        create_example_panel(
            global_coord=tuple(primary),
            extra_fcn_coords=None,
            extra_dcc_coords=None,
            extra_fcn_coords_front=extra_fcn_front,
            extra_dcc_coords_front=extra_dcc_front,
            extra_fcn_coords_back=extra_fcn_back,
            extra_dcc_coords_back=extra_dcc_back,
            save_folder=output_folder_clean,
            save_name=case_id,
            title=case_id,
            description=None
        )

    # ---- INTERSECTION CLUSTER (FAD / DAF) ----
    elif prefix in ("FAD", "DAF"):
        cluster_id = case_id[-4:]
        if cluster_id in processed_clusters:
            continue
        processed_clusters.add(cluster_id)

        coords = clusters.get(cluster_id, {"fcn_coords": [], "dcc_coords": []})
        desc = (f"Cluster {cluster_id}: "
                f"{len(coords['fcn_coords'])} FCN pts, "
                f"{len(coords['dcc_coords'])} DCC pts")

        create_example_panel(
            global_coord=None,
            extra_fcn_coords=coords["fcn_coords"],
            extra_dcc_coords=coords["dcc_coords"],
            save_folder=output_folder_intersection,
            save_name=f"cluster_{cluster_id}",
            title=f"Cluster {cluster_id}",
            description=None
        )

    # ---- UNIQUE FCN FALSE POSITIVE (FMD) ----
    elif case_id.startswith("FMD"):
        entry = next((f for f in fmd_ids if f["id"] == case_id), None)
        if entry is None:
            print(f"[WARN] FMD ID {case_id} not found.")
            continue

        create_example_panel(
            global_coord=None,
            extra_fcn_coords=[entry["coordinate"]],
            extra_dcc_coords=None,
            save_folder=output_folder_fmd,
            save_name=case_id,
            title=case_id,
            description=None
        )

    # ---- UNIQUE DCC FALSE POSITIVE (DMF) ----
    elif case_id.startswith("DMF"):
        entry = next((f for f in dmf_ids if f["id"] == case_id), None)
        if entry is None:
            print(f"[WARN] DMF ID {case_id} not found.")
            continue

        create_example_panel(
            global_coord=None,
            extra_fcn_coords=None,
            extra_dcc_coords=[entry["coordinate"]],
            save_folder=output_folder_dmf,
            save_name=case_id,
            title=case_id,
            description=None
        )

    else:
        print(f"[WARN] Unrecognized ID prefix for {case_id!r}")

print("Done generating all panels.")
