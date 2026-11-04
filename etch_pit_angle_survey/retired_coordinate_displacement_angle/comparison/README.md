# Detection-angle comparison

This directory compares Lewis and Kanik angle candidates for each clean ID
with the aligned FCN and DCC detection results.

A clean pit is `detected` when the nearest aligned prediction from that model
has strict Euclidean distance `< 15` pixels, matching
`modules/compare/test_FCN_aligned.py` and
`modules/compare/test_DCC_aligned.py`. A `missed` clean pit is a false negative.
False-positive predictions are not included because they do not correspond to
a clean truth ID and therefore have no clean C1/C2 angle.

For every clean ID, the plot generator ranks all unique Lewis/Kanik candidate
pairs by `abs(Lewis angle - Kanik angle)` and retains at most the two pairs
with the smallest errors. IDs with only one pair contribute one observation;
IDs with multiple pairs contribute exactly two. Ties retain the deterministic
candidate order from `lewis_kanik_angle_comparison.csv`.

Detection status remains an ID-level property and is copied to each retained
candidate pair. Consequently, the plots and binned detection rates are
candidate-weighted rather than one-observation-per-ID summaries.

Run from anywhere:

```bash
python /home/aman/Moedal/MoEDAL-ML-ANA/etch_pit_angle_survey/retired_coordinate_displacement_angle/comparison/plot_detection_angles.py
```

The generator uses NumPy for the matching calculations and the installed
Gnuplot `pngcairo`/`pdfcairo` backends for rendering.

Outputs:

- `data/clean_id_detection_angles.csv`: selected angles, nearest-prediction
  distances, individual model status, and joint status for all 1,242 clean IDs;
  this remains an ID-level audit file and is not the plotting input.
- `data/lowest_error_plot_candidates.csv`: the exact one or two lowest-error
  candidate pairs per clean ID used by every plot.
- `data/detection_angle_summary.json`: ID counts, retained-candidate counts,
  and descriptive angle statistics for the plotting candidates.
- `plots/detected_vs_missed_angle_distributions.{png,pdf}`: normalized angle
  histograms for detected and missed candidates for each model and angle source.
- `plots/joint_detection_angle_distributions.{png,pdf}`: candidate-angle
  distributions for detected by both, FCN only, DCC only, missed by both, and
  the aggregate missed-by-FCN and missed-by-DCC categories.
- `plots/detection_rate_vs_angle.{png,pdf}`: detection percentage in two-degree
  angle bins, with the number of retained candidates in each bin shown in gray.

The dashed line at approximately 18.57 degrees is the angle corresponding to
the strict 12-pixel C1/C2 geometry-matching boundary used by the angle surveys.

The first four detection outcomes are mutually exclusive. The two additional
missed-model rows are aggregate views: `missed by FCN` contains DCC-only plus
missed-by-both IDs, while `missed by DCC` contains FCN-only plus
missed-by-both IDs.
