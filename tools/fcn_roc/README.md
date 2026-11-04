# FCN operating-characteristic calculation

Code downloaded from the Pascal full-imbalance pit-weighted scan on 2026-11-04.
`fcn_roc_pipeline.py` extracts coordinates from saved model probability maps,
applies V1.1 cuts, aligns sides, evaluates inclusive 12-pixel matches, and scans
independent front/back thresholds plus alpha/beta at sigma 8. It produces PR
and FROC curves (FP/mm²); it does not train or infer the neural network itself.
The plotting function additionally accepts an optional selected working point.
The original root-level script remains unchanged.

Dependencies: Python, NumPy, SciPy, Matplotlib, Pillow, and the coordinate code
in `source_snapshot/` (preserved Python sources from Pascal).

For a new scan, prepare RUN/inputs with front_probability_masks.npy,
back_probability_masks.npy (background-positive convention),
Aligned_clean_with_ids.txt (active truths), analysis_v2_regions.json, and
source_snapshot/modules/dataset_tools/metadata/Image_location_metadata.txt
alongside the supplied source snapshot. Input data and scan caches were not
downloaded as part of this request. Use an isolated run directory per model.

```bash
python tools/fcn_roc/fcn_roc_pipeline.py --run-root RUN --model-name MODEL --mode full --workers 12
python tools/fcn_roc/plot_selected_working_point.py --results-csv SCAN.csv --output-root FIGURE_RUN
```

The second command only replots existing results and defaults to selected
configuration 7dc92f2228c54ad55267. The scanner's automatically reported
maximum-F1 optimum is a diagnostic, not the selected manuscript working point.

Publication plots were generated on Pascal in
`/home/au152/MoEDAL/fcn_roc_pit_weighted_publication_selected_20261104/`, using
`/home/au152/MoEDAL/fcn_roc_pit_weighted_full_v1p1_cuts_targeted_refinement_20261104/results/combined_results.csv`.
The input tables and caches remain there. Local PDF/PNG copies are under
`pipeline_runs/Analysis_V1.2_pit_weighted_FCN_working_point/plots/roc/`.
The PR ordinate uses cubic scaling; FROC abscissa uses logarithmic scaling.
The red cross denotes TP=1171, FP=35, FN=61 in the ROC evaluator. The separate
local analysis reports FP=30 after its additional image-validity filter; see
MANUSCRIPT_UPDATE_NOTES.md for the manuscript reporting distinction.
