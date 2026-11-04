# Isolated pipeline runs

Generated results are no longer stored at the repository root. New results must
be written below one explicitly created run directory. The directory keeps the
legacy internal layout (`predicted_etch_pits/`, `predicted_list/`, `plots/`,
`gallery/`, `my_panels/`, `stitched_images/`, and generated NumPy masks), so a
complete result can be moved or archived as one unit.

The pipeline has three explicit scientific starting points. CNN is always
seeded from its two preprocessed front/back coordinate files plus the three
supporting result files
`GXY_allchan_008thresh_eval.txt`, `IXY_allchan_008thresh.txt`, and
`IXY_allchan_008thresh_eval.txt` under `predicted_etch_pits/CNN/`. To begin
from all six already preprocessed CNN/FCN/DCC front/back coordinate files:

```bash
python run_pipeline.py \
  --run-dir pipeline_runs/NEW_COORDINATE_RUN \
  --start-from preprocessed_predicted_coordinates \
  --input-source pipeline_runs/paperV1_baseline \
  --clean-id-registry clean_pit_id_registries/paperV1.json
```

Every input source must also contain a complete archived `kanik/` directory.
The whole tree is copied into each new run before analysis, including the clean
truth inputs, raw DCC centroid outputs, and historical result files.

To begin from stored FCN probability maps and Kanik's raw DCC centroid outputs,
the input source must contain `model_outputs/FCN/front_probability_masks.npy`
and `back_probability_masks.npy`. DCC raw outputs are copied from the input
source's archived `kanik/og_data/` directory and converted with the current
coordinate presets and the retained four-pixel image-edge cut:

```bash
python run_pipeline.py \
  --run-dir pipeline_runs/2026-10-29_model_output \
  --start-from model_output \
  --input-source INPUT_ROOT \
  --clean-id-registry clean_pit_id_registries/paperV1.json \
  --front-threshold FRONT --back-threshold BACK \
  --sigma SIGMA --alpha ALPHA --beta BETA
```

To begin with the FCN model, supply the same reconstruction parameters
explicitly. FCN probability maps are generated below `model_outputs/FCN/`.
Until a DCC model runner is added, DCC deliberately falls back to its copied
raw Kanik model outputs and records that fallback in the run manifest:

```bash
python run_pipeline.py \
  --run-dir pipeline_runs/2026-10-29_unet \
  --start-from model \
  --input-source CNN_COORDINATE_SOURCE \
  --clean-id-registry clean_pit_id_registries/paperV1.json \
  --front-threshold FRONT \
  --back-threshold BACK \
  --sigma SIGMA --alpha ALPHA --beta BETA \
  --require-gpu
```

Spatial-resolution plots, the full clean-pit/false-positive gallery, and the
selected manuscript example panels are part of the default pipeline. Each can
be skipped explicitly. The deprecated efficiency/purity curves remain opt-in.
Run `python run_pipeline.py --help` for the switches.

Every analysis explicitly selects an immutable clean-pit ID registry. Optional
ID-preserving truth cuts can be added by repeating, for example,
`--fiducial-cut-file fiducial_cuts/analysis_v2.json`. Registry and cut-file
paths and hashes are recorded in the run manifest. Active and excluded truth
records retain their registry IDs inside the run.

Spatial cuts shared by predictions and clean truth are selected with, for
example, `--spatial-fiducial-file fiducial_cuts/analysis_v2_regions.json`.
They preserve the six raw preprocessed coordinate inputs and write separate
fiducial-filtered inputs plus a complete removal audit inside the run.
Analysis v2 additionally selects
`--fiducial-cut-file fiducial_cuts/analysis_v2_truth_blacklist.json` to omit
`C0123`, `C0300`, and `C0997` without changing the global clean-ID registry.

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
    "predicted_etch_pits/CNN/GXY_allchan_008thresh_eval.txt",
    "predicted_etch_pits/CNN/IXY_allchan_008thresh.txt",
    "predicted_etch_pits/CNN/IXY_allchan_008thresh_eval.txt",
])
```

Every run contains `pipeline_run.json`, including creation metadata and hashes
for copied starting predictions and immutable truth/coordinate inputs. Creating
a run refuses any existing path; writing an existing artifact also fails.
Choose a new run name for every rerun. The downstream runner also writes
`pipeline_summary.json` with the TP, FP, FN, prediction, and fragment counts
for every method and view.

Static inputs stay outside the run and are never modified:

- `models/`
- `numpy_data_eval/preprocessed_*.npy`
- `modules/dataset_tools/metadata/`
- `clean_pit_id_registries/`
- `fiducial_cuts/`

The former root-level `kanik/` tree is archived inside pipeline runs and is
carried forward from the explicitly selected `--input-source`.

The `etch_pit_angle_survey/` tree is outside this refactor.

`pipeline_runs/paperV1_baseline` is explicitly recorded as a
`preprocessed_predicted_coordinates` start. It retains all six CNN/FCN/DCC
front/back coordinate inputs and the three supporting CNN result files.
