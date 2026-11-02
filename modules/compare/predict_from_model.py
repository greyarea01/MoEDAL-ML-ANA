"""Generate front/back U-Net probability masks as isolated run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from modules.pipeline_paths import (
    RUN_MANIFEST,
    active_run_root,
    artifact_path,
    create_pipeline_run,
    sha256_file,
    source_path,
)


def _predict_dataset(model, source: Path, destination: Path, batch_size: int) -> dict:
    array = np.load(source, mmap_mode="r", allow_pickle=False)
    if array.ndim != 4 or array.shape[-1] != 10:
        raise ValueError(f"Expected (N,H,W,10) input at {source}; found {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input at {source}; found {array.dtype}")

    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to replace stale partial output: {partial}")
    output_shape = (*array.shape[:3], 1)
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=output_shape
    )

    minimum = float("inf")
    maximum = float("-inf")
    total = 0.0
    elements = 0
    started = time.time()
    try:
        for start in range(0, array.shape[0], batch_size):
            stop = min(start + batch_size, array.shape[0])
            batch = np.asarray(array[start:stop], dtype=np.float32) / 255.0
            prediction = np.asarray(model.predict_on_batch(batch), dtype=np.float32)
            expected = (stop - start, *output_shape[1:])
            if prediction.shape != expected:
                raise ValueError(
                    f"Unexpected model output {prediction.shape}; expected {expected}"
                )
            output[start:stop] = prediction
            minimum = min(minimum, float(prediction.min()))
            maximum = max(maximum, float(prediction.max()))
            total += float(prediction.sum(dtype=np.float64))
            elements += prediction.size
            print(f"{destination.stem}: {stop}/{array.shape[0]} images", flush=True)
        output.flush()
        del output
        output = None
        os.replace(partial, destination)
    except Exception:
        if output is not None:
            del output
        raise

    return {
        "input_path": str(source),
        "input_shape": list(array.shape),
        "input_dtype": str(array.dtype),
        "input_sha256": sha256_file(source),
        "output_path": str(destination),
        "output_shape": list(output_shape),
        "output_dtype": "float32",
        "output_sha256": sha256_file(destination),
        "probability_min": minimum,
        "probability_max": maximum,
        "probability_mean": total / elements,
        "elapsed_seconds": time.time() - started,
    }


def generate_probability_masks(
    *,
    model_path: str | os.PathLike[str] = "models/FCN/unet_ROC.keras",
    front_input: str | os.PathLike[str] = "numpy_data_eval/preprocessed_dirty_eval.npy",
    back_input: str | os.PathLike[str] = "numpy_data_eval/preprocessed_dirty_reverse_eval.npy",
    batch_size: int = 4,
    require_gpu: bool = False,
) -> dict:
    """Run inference once and save both probability stacks below the active run."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model_file = source_path(model_path)
    front_file = source_path(front_input)
    back_file = source_path(back_input)
    front_output = artifact_path(
        "model_outputs/FCN/front_probability_masks.npy", for_write=True
    )
    back_output = artifact_path(
        "model_outputs/FCN/back_probability_masks.npy", for_write=True
    )
    metadata_output = artifact_path(
        "model_outputs/FCN/generation_metadata.json", for_write=True
    )

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if require_gpu and not gpus:
        raise RuntimeError("GPU inference was requested but TensorFlow sees no GPU")

    model = tf.keras.models.load_model(model_file, compile=False)
    result = {
        "model_path": str(model_file),
        "model_sha256": sha256_file(model_file),
        "tensorflow_version": tf.__version__,
        "visible_gpus": [gpu.name for gpu in gpus],
        "batch_size": batch_size,
        "datasets": {},
    }
    result["datasets"]["front"] = _predict_dataset(
        model, front_file, front_output, batch_size
    )
    result["datasets"]["back"] = _predict_dataset(
        model, back_file, back_output, batch_size
    )
    with metadata_output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")

    manifest_path = active_run_root() / RUN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["probability_mask_generation"] = result
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="models/FCN/unet_ROC.keras")
    parser.add_argument("--front-input", default="numpy_data_eval/preprocessed_dirty_eval.npy")
    parser.add_argument("--back-input", default="numpy_data_eval/preprocessed_dirty_reverse_eval.npy")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    create_pipeline_run(args.run_dir, metadata={"entry_point": __file__})
    result = generate_probability_masks(
        model_path=args.model,
        front_input=args.front_input,
        back_input=args.back_input,
        batch_size=args.batch_size,
        require_gpu=args.require_gpu,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
