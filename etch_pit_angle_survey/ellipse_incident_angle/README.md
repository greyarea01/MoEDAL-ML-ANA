# Image-based ellipse incident-angle pipeline

This is the active etch-pit incident-angle analysis. It fits an ellipse to the
dark outer boundary of each clean pit independently on C1 and C2. The angle is
computed as:

```text
incident_angle_degrees = degrees(acos(minor_axis_pixels / major_axis_pixels))
```

The stored C1 and C2 coordinates identify the corresponding pit and locate its
image crop. C1-to-C2 coordinate displacement is not used to calculate the
angle.

## Inputs

- `inputs/clean_surface_locations.csv`: 1,242 clean IDs, truth coordinates,
  and the matching local C1 and C2 image locations. It intentionally contains
  no transformed C1-to-C2 displacement or coordinate-derived angle.
- `numpy_data_eval/preprocessed_clean_eval.npy`: C1 images 251 through 500.
- `numpy_data_eval/clean_2_eval.npy`: C2 images 251 through 500.
- `predicted_list/FCN/aligned/final_pred_list.txt` and the corresponding DCC
  file: used only for the detection-outcome report.

The surface-location manifest was extracted from the validated Lewis clean-ID
association before the old coordinate-angle analysis was archived. Coordinate
registration is therefore still used to identify the same physical pit on the
two surfaces, but it does not enter the ellipse angle.

## Processing

For each pit and surface, the pipeline:

1. Extracts a 49 by 49 pixel crop centred on the stored local coordinate.
2. Marks every crop pixel outside the real source image as invalid.
3. Applies a Gaussian blur using a normalized convolution over valid source
   pixels only.
4. Computes an inverse Otsu threshold from valid pixels only.
5. Applies a 3 by 3 morphological close to each pre-closing connected
   component independently and clears every invalid pixel again. Contours from
   different components remain separate even if their closed masks overlap;
   this prevents closing from attaching a disconnected scratch to a pit.
6. Finds the external contour of each independently closed component.
7. Rejects contour candidates with fewer than five points, area below 15
   square pixels, minor diameter below 3 pixels, or major diameter above 47
   pixels.
8. Scores the remaining contours using centre offset, solidity, and ellipse
   residual, then selects the lowest-score candidate.
9. Fits an ellipse and applies all quality controls below.

### Edge handling

The pipeline never reflects, mirrors, replicates, or otherwise invents pixels
outside the source image. Invalid pixels are excluded from blur normalization,
Otsu threshold estimation, segmentation, and contour creation. If a selected
contour reaches a real source-image boundary, the pit fails QC F because its
shape is truncated.

This fixes the earlier diagnostic-script bug where `BORDER_REFLECT_101`
created a mirror copy of an edge pit and allowed the false copy to join the
real contour.

## Quality controls

- **A — candidate available:** at least five contour points, area at least 15
  square pixels, minor diameter at least 3 pixels, and major diameter at most
  47 pixels.
- **B — centre offset:** ellipse centre is at most 10 pixels from the stored
  pit location.
- **C — solidity:** contour area divided by convex-hull area is at least 0.92.
- **D — ellipse residual:** the 90th percentile of the normalized radial
  mismatch is at most 0.15.
- **E — analysis-crop boundary:** the selected contour does not touch the 49 by
  49 crop boundary.
- **F — source-image boundary:** the selected contour does not touch a real
  source-image boundary.

All failed-cut combinations in the outputs are exhaustive and mutually
exclusive. Candidate angles from failed fits are retained for diagnostics but
are never included in accepted angle or C1-to-C2 error distributions.

## Run

From the repository root:

```bash
python etch_pit_angle_survey/ellipse_incident_angle/run_pipeline.py
```

Use `--no-reports` to recreate only the CSV and JSON outputs. `--data-dir` and
`--report-dir` can redirect generated files.

### Evaluate a fit configuration

The configuration-selection loss is implemented separately so that running
the reporting pipeline does not automatically perform the additional
coordinate-perturbation fits. From the repository root, run:

```bash
python etch_pit_angle_survey/ellipse_incident_angle/evaluate_loss.py
```

The evaluator runs the nominal fit and eight one-pixel coordinate
perturbations on each nominally accepted surface. A surface is a robust pass
only when its nominal fit passes and a strict majority of its perturbations
pass. With the default eight perturbations, at least five must pass. For a
robust surface, each failed perturbation contributes the configured stability
failure penalty. A surface that fails the majority gate becomes a robust QC
failure instead and does not enter the angle-based loss terms.

The six nonnegative raw loss terms are:

```text
coverage_penalty = 1 - robust_paired_pass_count / clean_pit_count
central_absolute_error = mean(abs(C2_angle - C1_angle))
tail_error = weighted mean of the configured absolute-error percentiles
signed_bias = abs(mean(C2_angle - C1_angle))
mean_stability_error = mean coordinate-perturbation cost
low_angle_penalty = max(0, low_angle_fraction - allowed_low_angle_fraction)
```

The configured default weights are explicitly marked provisional until
`weights_are_provisional` is changed or `--weights-final` is supplied. Use
`--k1-coverage` through `--k6-low-angle` for one-run overrides. All ellipse-fit
settings, perturbation offsets, the strict-majority threshold, failure
penalty, tail percentiles and their weights, low-angle settings, and minimum
coverage guardrail are command-line configurable. Run `evaluate_loss.py
--help` for the complete interface. The dataclasses in `ellipse_fit.py` and
`loss_function.py` are the single source of truth for command-line defaults.

### Search Gaussian blur phase space

Run the adaptive kernel-and-sigma search from the repository root with:

```bash
python etch_pit_angle_survey/ellipse_incident_angle/search_gaussian_phase_space.py
```

The search starts from the Gaussian settings in `FitConfig`, evaluates the
two-dimensional neighborhood, moves to its lowest-loss point, and repeats
until no adjacent point improves the loss or the iteration limit is reached.
The requested default increments are one pixel for the kernel and 0.2 pixels
for sigma. OpenCV permits only positive odd Gaussian kernels, so an even
one-pixel proposal is recorded as invalid and the next odd kernel in that
direction is evaluated. Previously evaluated points are cached.

Search steps, limits, starting point, stopping tolerance, and output paths are
command-line configurable. Run `search_gaussian_phase_space.py --help` for the
complete interface.

## Outputs

Data:

- `data/ellipse_incident_angles.csv`: one row per clean ID with local
  locations, complete C1 and C2 fit measurements, every QC result, paired
  angle errors, and FCN/DCC detection categories.
- `data/summary.json`: configuration, counts, statistics, and edge-handling
  provenance.
- `data/qc_failure_combinations.csv`: exhaustive failure buckets for C1 and
  C2.
- `data/detection_categories.csv`: compact FCN/DCC detection-versus-angle
  table.
- `data/solidity_residual_failure_index.csv`: page and panel lookup for the
  failure-case PDF.
- `data/current_config_loss_summary.json`: complete fit and loss
  configuration, raw terms, weighted terms, coverage, tail percentiles, and
  perturbation audit counts for the evaluated configuration.
- `data/current_config_loss_per_pit.csv`: nominal and robust QC status,
  perturbation pass/fail counts, stability, and loss inputs for every clean
  ID.
- `data/gaussian_phase_space_search.csv`: raw and weighted loss terms for
  every valid kernel-and-sigma point evaluated by the adaptive search.
- `data/gaussian_phase_space_search.json`: search configuration, path, stop
  reason, invalid even-kernel proposals, best point, and all evaluations.

Reports:

- `reports/incident_angle_report.pdf`: random examples, QC definitions,
  surface failure buckets, C1 and C2 angle distributions, and absolute and
  signed C1-to-C2 error distributions.
- `reports/solidity_residual_failure_cases.pdf`: raw and overlaid crops for all
  C1 and C2 fits that fail solidity or ellipse residual.
- `reports/detection_outcomes.pdf`: FCN and DCC detected, missed, and
  detected-but-QC-failed angle distributions.
- `reports/gaussian_phase_space_search.pdf`: sampled phase space, weighted
  term comparison, and best-point summary from the adaptive search.

## Verification

Run the focused tests with:

```bash
python -m unittest discover -s etch_pit_angle_survey/ellipse_incident_angle/tests -v
```
