# Isolated pipeline runs

Generated results are no longer stored at the repository root. New results must
be written below one explicitly created run directory. The directory keeps the
legacy internal layout (`predicted_etch_pits/`, `predicted_list/`, `plots/`,
`gallery/`, `my_panels/`, `stitched_images/`, and generated NumPy masks), so a
complete result can be moved or archived as one unit.

For an exact downstream rerun using the committed coordinate predictions:

```bash
python run_pipeline.py \
  --run-dir pipeline_runs/2026-10-29_baseline \
  --prediction-mode seed \
  --prediction-source pipeline_runs/2026-11-01_phase1_bytecheck
```

To begin with U-Net inference, supply the selected post-processing parameters
explicitly; the command intentionally has no hidden defaults for them:

```bash
python run_pipeline.py \
  --run-dir pipeline_runs/2026-10-29_unet \
  --prediction-mode unet \
  --prediction-source pipeline_runs/2026-11-01_phase1_bytecheck \
  --front-threshold FRONT \
  --back-threshold BACK \
  --sigma SIGMA --alpha ALPHA --beta BETA \
  --require-gpu
```

Spatial-resolution plots are part of the default analysis but can be skipped.
The deprecated efficiency/purity curves, full gallery, and selected example
panels are opt-in stages. Run `python run_pipeline.py --help` for the switches.

```python
from modules.pipeline_paths import create_pipeline_run, seed_artifacts

create_pipeline_run("pipeline_runs/2026-10-29_baseline")
seed_artifacts([
    "predicted_etch_pits/CNN/final_pr.txt",
    "predicted_etch_pits/CNN/final_reverse_pr.txt",
    "predicted_etch_pits/FCN/final_pr.txt",
    "predicted_etch_pits/FCN/final_reverse_pr.txt",
    "predicted_etch_pits/DCC/final_pr.txt",
    "predicted_etch_pits/DCC/final_reverse_pr.txt",
])
```

Every run contains `pipeline_run.json`, including creation metadata and hashes
for copied starting predictions and immutable truth/coordinate inputs. Creating a run refuses any existing path;
writing an existing artifact also fails. Choose a new run name for every rerun.
The downstream runner also writes `pipeline_summary.json` with the TP, FP, FN,
prediction, and fragment counts for every method and view.

Static inputs stay outside the run and are never modified:

- `model/`
- `numpy_data_eval/preprocessed_*.npy`
- `modules/dataset_tools/metadata/`
- `kanik/og_data/` and the checked clean-coordinate inputs

The `etch_pit_angle_survey/` tree is outside this refactor.

The verified Phase-1 reference run is local at
`pipeline_runs/2026-11-01_phase1_bytecheck`. Its 118 copied/generated artifacts
were byte-identical to the former root-level files before those redundant files
were removed. Because `pipeline_runs/` is intentionally ignored, preserve or
archive that directory before cleaning the working copy.
