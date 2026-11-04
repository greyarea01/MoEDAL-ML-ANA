# Lewis etch-pit angle survey

This additive directory reconstructs Lewis's clean C1/Cr/C2 alignment and
measures the C1-to-C2 displacement and angle for every ID in the final clean
truth list. Existing repository files are not modified.

## Sources

- `modules/dataset_tools/metadata/all_matched_local_locations.txt`: Lewis's
  original local C1, Cr, and C2 centroids and historical aligned memberships.
- `modules/dataset_tools/metadata/all_aligned_Global_locations.txt`: checked-in
  integer global coordinates used to validate each transformation and aligned
  list.
- `predicted_list/ids/Aligned_clean_with_ids.txt`: final clean coordinates and
  existing `Cxxxx` IDs.

## Exact Lewis reconstruction

The original metadata-producer script is not present, but its operational
behavior is reproduced exactly:

1. Transform C1 with `Coord_system()["c1"].get_gxy(...)`.
2. Transform C2 with `Coord_system()["c2"].get_gxy(...)`.
3. Mark Cr as flipped and reverse, then transform it with
   `Coord_system()["cr"].get_gxy(...)`.
4. On the full-precision transformed coordinates, retain C1 points with at
   least one Cr partner at strict Euclidean distance `< 12` pixels.
5. From that retained C1 list, keep points with at least one C2 partner at
   strict Euclidean distance `< 12` pixels.
6. Truncate global coordinates toward zero with Python `int()` only when
   constructing the historical integer metadata.

Matching before integer truncation is important. This process reproduces the
2,970-point C1/Cr list and 2,682-point C1/Cr/C2 list exactly, in source order.
All original C1, Cr, and C2 global lists are also reproduced exactly.

The script additionally validates the downstream truth construction:

- 1,272 Lewis aligned evaluation pits after the inverse C1 image cut.
- 1,263 coordinates shared with Kanik at the historical threshold.
- 1,242 coordinates after the existing exposed-image edge cut.
- Exact coordinate agreement with the ID-labelled clean truth list.

## Distances and angles

Distances use `5.6 um/pixel`. The C1/C2 angle is measured from the foil normal
using a 200 um separation between the C1 and C2 surfaces:

```text
angle = atan(C1-to-C2 lateral distance / 200 um)
```

The historical 12-pixel boundary is 67.2 um and corresponds to
18.572349 degrees. Since matching is strict `< 12` pixels, accepted C1/C2
matches have angles below that boundary.

Every valid C1/Cr/C2 combination is retained in `all_candidate_pairs.csv`.
Because the final truth coordinates originate from Lewis C1, a C1 point whose
historical truncated coordinate exactly equals the truth coordinate ranks
before merely nearby C1 candidates. Remaining ranking uses truth-to-C1
distance, C1-to-C2 distance, and truth-to-C2 distance. For the same C1/C2 pair,
the nearer Cr candidate ranks first. Alternative triplets remain in the CSV
for auditability.

## Run

```bash
python /home/aman/Moedal/MoEDAL-ML-ANA/etch_pit_angle_survey/retired_coordinate_displacement_angle/lewis/survey_lewis_c1_c2_angles.py
```

The command prints the selected distance, angle, and status for every clean ID
and writes:

- `data/clean_c1_c2_distances.csv`: one selected Lewis triplet per clean ID.
- `data/all_candidate_pairs.csv`: every feasible Lewis C1/Cr/C2 triplet.
- `data/summary.json`: reconstruction checks, counts, and distance/angle
  statistics.

Use `--quiet` to suppress per-ID printing. Use `--threshold-um N` to change the
downstream truth-to-C1 and C1-to-C2 survey threshold; historical Lewis aligned
membership remains fixed at its original 67.2 um threshold.
