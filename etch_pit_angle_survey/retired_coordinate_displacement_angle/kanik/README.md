# Kanik etch-pit angle survey

This directory contains an additive survey of the displacement between Kanik's
clean C1 and C2 etch-pit coordinates. It does not modify the existing analysis
or coordinate files.

Reported distances use `5.6 um/pixel` and are written in micrometers. Coordinate
columns remain in the repository's original global or local coordinate units.

The C1/C2 angle is measured from the foil normal using a 200 um separation
between the C1 and C2 surfaces:

```text
angle = atan(C1-to-C2 lateral distance / 200 um)
```

At the historical 12-pixel matching boundary, the lateral distance is 67.2 um
and the angle is approximately 18.57 degrees. Because the matching comparison
is strict `< 12` pixels, accepted matches have angles strictly below this
boundary.

## Inputs

- `predicted_list/ids/Aligned_clean_with_ids.txt`: clean truth coordinates and
  their existing `Cxxxx` IDs.
- `kanik/og_data/CleanTestCentroid.txt`: Kanik C1 local centroids.
- `kanik/og_data/CleanReverseTestCentroid.txt`: Kanik Cr local centroids.
- `kanik/og_data/Clean2TestCentroid.txt`: Kanik C2 local coordinates in
  `(image, x, y)` form.
- `kanik/cleanTest.txt`, `kanik/cleanReverse.txt`, and `kanik/clean2.txt`:
  checked-in global coordinates used to validate all three transformations.
- `kanik/clean_aligned.txt`: checked-in Kanik aligned C1 list used to validate
  the reconstructed three-surface selection.

Kanik's C1 and already-flipped Cr centroids are transformed with
`Coord_system()["c1"].get_gxy(...)`. C2 is transformed with
`Coord_system()["c2"].get_gxy(...)`, placing it in the same global frame.
Each transformation reproduces its checked-in global list exactly.

## Kanik aligned-list reconstruction

The original producer script for `kanik/clean_aligned.txt` is not present in
the repository. Its selection is nevertheless reproduced exactly using the
repository's `coordinates.Overlap` behavior and the checked-in inputs:

1. Ignore raw centroid rows whose image number is zero.
2. Require a C1 point to have at least one Cr point at strict Euclidean
   distance `< 67.2 um` (the historical 12-pixel threshold).
3. Require that C1 point to have at least one transformed C2 point at strict
   Euclidean distance `< 67.2 um`.
4. Require `Coord_system()["c1"].get_ixy(C1)[0] >= 251`.
5. Retain the surviving coordinates in their original C1 order.

This reconstructs all 1,298 rows of `kanik/clean_aligned.txt` in the same order
with zero coordinate difference. Cr is used only as an eligibility condition;
Cr coordinates and C1-to-Cr distances are not written to the survey CSVs.

## Matching

The default threshold is a strict Euclidean distance `< 67.2 um`, equivalent
to the repository's original `< 12` pixel `coordinates.Overlap` behavior.

For every clean truth ID, the script:

1. Collects all reconstructed aligned Kanik C1 candidates within 67.2 um of
   the clean truth.
2. Collects all transformed Kanik C2 candidates within 67.2 um of each C1
   candidate.
3. Collects all Kanik Cr candidates within the historical 67.2 um threshold
   of each C1 candidate.
4. Stores every feasible C1/Cr/C2 combination. This preserves all Cr
   alternatives rather than silently choosing the nearest one.
5. Selects the triplet with the smallest truth-to-C1 distance, then the
   smallest C1-to-C2 distance. For the same C1/C2 pair, the smallest C1-to-Cr
   distance ranks first. Remaining ties are deterministic.

This rule treats the clean truth as a C1 measurement. Multiple candidates are
not discarded: they appear in `all_candidate_pairs.csv`. The selected-results
file also flags ambiguous candidates and C1 or C2 selections shared by more
than one clean ID.

## Run

From anywhere:

```bash
python /home/aman/Moedal/MoEDAL-ML-ANA/etch_pit_angle_survey/retired_coordinate_displacement_angle/kanik/survey_clean_c1_c2_distances.py
```

The command prints
`clean_id,c1_to_c2_distance_um,c1_c2_angle_from_foil_normal_deg,status` and
writes:

- `data/clean_c1_c2_distances.csv`: one selected result per clean ID.
- `data/all_candidate_pairs.csv`: every feasible C1/Cr/C2 triplet, including
  alternatives and all requested Cr coordinates and distances.
- `data/summary.json`: input counts, ambiguity counts, transformation
  validation, and distance statistics.

In `all_candidate_pairs.csv`, `final_match_status` is
`chosen_for_final_output` for the rank-1 triplet used by the selected-results
file. Other valid triplets are labelled `alternative_not_chosen`; they passed
the distance thresholds but ranked below the chosen triplet.

Use `--quiet` to generate the data files without printing all clean IDs. Use
`--threshold-um N` to change the downstream truth-to-C1 and C1-to-C2 survey
threshold in micrometers; reconstruction of the historical Kanik aligned list
remains fixed at its original 67.2 um threshold.
