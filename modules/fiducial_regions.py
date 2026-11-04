"""Validate and apply spatial fiducial regions to truth and predictions."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Sequence

from modules.dataset_tools import coordinates, mhio
from modules.pipeline_paths import (
    PROJECT_ROOT,
    artifact_path,
    source_path,
)


FIDUCIAL_ROOT = (PROJECT_ROOT / "fiducial_cuts").resolve()
METHODS = ("CNN", "FCN", "DCC")
SIDES = ("front", "back")
SIDE_SUFFIX = {"front": "pr", "back": "reverse_pr"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _coordinate(value: object, *, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context} must be a two-value coordinate list")
    return [
        _number(value[0], context=f"{context}[0]"),
        _number(value[1], context=f"{context}[1]"),
    ]


def load_fiducial_regions(pathname: str | Path) -> tuple[Path, dict, str]:
    """Load one repository-level spatial fiducial configuration."""
    path = source_path(pathname)
    if FIDUCIAL_ROOT not in path.parents:
        raise ValueError(
            f"Spatial fiducial configuration must be below {FIDUCIAL_ROOT}: {path}"
        )
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_fiducial_regions(config, context=str(path))
    return path, config, _sha256(path)


def validate_fiducial_regions(config: object, *, context: str = "fiducial regions") -> dict:
    """Validate the geometry needed by the analysis-v2 fiducial policy."""
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{context} must be an object using schema_version 1")
    if config.get("coordinate_frame") != "c1_global_pixels":
        raise ValueError(f"{context} must use the c1_global_pixels coordinate frame")
    if config.get("boundary_policy") != "inclusive":
        raise ValueError(f"{context} must use the inclusive boundary policy")

    edge = config.get("image_edges")
    if not isinstance(edge, dict):
        raise ValueError(f"{context}.image_edges must be an object")
    margin = _number(edge.get("margin_px"), context="image_edges.margin_px", positive=True)
    width = _number(edge.get("image_width_px"), context="image_edges.image_width_px", positive=True)
    height = _number(edge.get("image_height_px"), context="image_edges.image_height_px", positive=True)
    if margin * 2 >= min(width, height):
        raise ValueError("image edge margin leaves no image interior")
    if margin != 4.0:
        raise ValueError("analysis-v2 image edge margin must be exactly 4 pixels")
    first = edge.get("first_evaluation_image")
    last = edge.get("last_evaluation_image")
    if not isinstance(first, int) or not isinstance(last, int) or first > last:
        raise ValueError("image edge evaluation-image bounds are invalid")
    if edge.get("front_scan") != "d" or edge.get("back_scan") != "dr":
        raise ValueError("image edge scans must be d (front) and dr (back)")

    holes = config.get("alignment_holes")
    if not isinstance(holes, dict) or holes.get("shape") != "circle":
        raise ValueError(f"{context}.alignment_holes must define circles")
    hole_radius = _number(
        holes.get("radius_px"), context="alignment_holes.radius_px", positive=True
    )
    if hole_radius != 184.0:
        raise ValueError("analysis-v2 alignment-hole radius must be exactly 184 pixels")
    centers = holes.get("centers")
    if not isinstance(centers, list) or len(centers) != 4:
        raise ValueError("alignment_holes.centers must contain the four circles")
    seen_holes = set()
    side_counts = {"front": 0, "back": 0}
    for index, center in enumerate(centers):
        if not isinstance(center, dict):
            raise ValueError(f"alignment hole {index} must be an object")
        center_id = center.get("id")
        side = center.get("side")
        if not isinstance(center_id, str) or center_id in seen_holes:
            raise ValueError(f"alignment hole {index} has an invalid or duplicate ID")
        if side not in side_counts:
            raise ValueError(f"alignment hole {center_id} has invalid side {side!r}")
        _coordinate(center.get("coordinate"), context=f"alignment hole {center_id}")
        seen_holes.add(center_id)
        side_counts[side] += 1
    if side_counts != {"front": 2, "back": 2}:
        raise ValueError("alignment-hole configuration needs two circles per side")

    physical = config.get("physical_inconsistencies")
    if not isinstance(physical, dict) or physical.get("shape") != "axis_aligned_square":
        raise ValueError(
            f"{context}.physical_inconsistencies must define axis-aligned squares"
        )
    box_width = _number(
        physical.get("width_px"),
        context="physical_inconsistencies.width_px",
        positive=True,
    )
    box_height = _number(
        physical.get("height_px"),
        context="physical_inconsistencies.height_px",
        positive=True,
    )
    if box_width != 8.0 or box_height != 8.0:
        raise ValueError("physical-inconsistency boxes must be exactly 8 x 8 pixels")
    features = physical.get("features")
    if not isinstance(features, list) or len(features) != 10:
        raise ValueError("physical_inconsistencies.features must contain ten features")
    seen_features = set()
    review_sources = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(f"physical inconsistency {index} must be an object")
        feature_id = feature.get("id")
        source = feature.get("review_source")
        if not isinstance(feature_id, str) or feature_id in seen_features:
            raise ValueError(
                f"physical inconsistency {index} has an invalid or duplicate ID"
            )
        if not isinstance(source, str) or not source:
            raise ValueError(f"physical inconsistency {feature_id} needs a source")
        _coordinate(
            feature.get("front_center"), context=f"{feature_id}.front_center"
        )
        _coordinate(
            feature.get("back_center"), context=f"{feature_id}.back_center"
        )
        seen_features.add(feature_id)
        review_sources.add(source)
    expected_sources = {
        f"false_positive_cluster_{index:04d}"
        for index in (0, 1, 2, 3, 4, 8, 9, 12, 14)
    } | {"DMF0001"}
    if review_sources != expected_sources:
        raise ValueError(
            "physical-inconsistency review sources do not match the requested set"
        )
    return config


class FiducialRegionEvaluator:
    """Classify common-frame points against the configured spatial regions."""

    def __init__(self, config: dict):
        self.config = validate_fiducial_regions(config)
        self._coordinate_system: coordinates.Coord_system | None = None

    @property
    def coordinate_system(self) -> coordinates.Coord_system:
        if self._coordinate_system is None:
            system = coordinates.Coord_system()
            system["dr"].flipped = True
            system["dr"].reverse = True
            self._coordinate_system = system
        return self._coordinate_system

    def infer_local_coordinate(
        self, coordinate: Sequence[Real], *, side: str
    ) -> list[float]:
        """Map a C1-global point back to the nearest d/dr source image."""
        if side not in SIDES:
            raise ValueError(f"Unsupported prediction side: {side}")
        point = _coordinate(list(coordinate), context="global coordinate")
        c1_local = self.coordinate_system["c1"].get_ixy(point)
        preset = "cd" if side == "front" else "cdr"
        local = self.coordinate_system.convert(c1_local, preset)
        return [float(local[0]), float(local[1]), float(local[2])]

    def _image_edge_hit(self, local: Sequence[Real], *, side: str) -> dict | None:
        if len(local) != 3:
            raise ValueError("local coordinate must be [image, x, y]")
        image, x_value, y_value = (
            _number(local[0], context="local image"),
            _number(local[1], context="local x"),
            _number(local[2], context="local y"),
        )
        edge = self.config["image_edges"]
        if not edge["first_evaluation_image"] <= image <= edge["last_evaluation_image"]:
            return None
        margin = float(edge["margin_px"])
        width = float(edge["image_width_px"])
        height = float(edge["image_height_px"])
        hit_edges = []
        if x_value < margin:
            hit_edges.append("left")
        if x_value > width - margin:
            hit_edges.append("right")
        if y_value < margin:
            hit_edges.append("top")
        if y_value > height - margin:
            hit_edges.append("bottom")
        if not hit_edges:
            return None
        return {
            "side": side,
            "local_coordinate": [image, x_value, y_value],
            "edges": hit_edges,
        }

    def classify(
        self,
        coordinate: Sequence[Real],
        *,
        side: str | None = None,
        local_coordinate: Sequence[Real] | None = None,
    ) -> dict:
        """Return every fiducial reason applying to one global coordinate.

        Predictions provide their side and, when retained by preprocessing, their
        exact source-local coordinate. Truth has no single prediction side, so its
        edge membership is evaluated in both d and dr.
        """
        point = _coordinate(list(coordinate), context="global coordinate")
        if side is not None and side not in SIDES:
            raise ValueError(f"Unsupported prediction side: {side}")
        if side is None and local_coordinate is not None:
            raise ValueError("truth classification cannot take one side-local coordinate")

        matches: dict[str, list] = {}
        holes = self.config["alignment_holes"]
        radius = float(holes["radius_px"])
        hole_hits = []
        for center in holes["centers"]:
            distance = math.dist(point, center["coordinate"])
            if distance <= radius:
                hole_hits.append(
                    {
                        "id": center["id"],
                        "side": center["side"],
                        "distance_px": distance,
                    }
                )
        if hole_hits:
            matches["alignment_hole"] = hole_hits

        physical = self.config["physical_inconsistencies"]
        half_width = float(physical["width_px"]) / 2.0
        half_height = float(physical["height_px"]) / 2.0
        physical_hits = []
        for feature in physical["features"]:
            for feature_side in SIDES:
                center = feature[f"{feature_side}_center"]
                if (
                    abs(point[0] - center[0]) <= half_width
                    and abs(point[1] - center[1]) <= half_height
                ):
                    physical_hits.append(
                        {
                            "id": feature["id"],
                            "review_source": feature["review_source"],
                            "center_side": feature_side,
                        }
                    )
        if physical_hits:
            matches["physical_inconsistency"] = physical_hits

        edge_hits = []
        edge_local_provenance = "provided"
        if side is None:
            edge_local_provenance = "inferred"
            for truth_side in SIDES:
                local = self.infer_local_coordinate(point, side=truth_side)
                hit = self._image_edge_hit(local, side=truth_side)
                if hit:
                    edge_hits.append(hit)
        else:
            if local_coordinate is None:
                edge_local_provenance = "inferred"
                local_coordinate = self.infer_local_coordinate(point, side=side)
            hit = self._image_edge_hit(local_coordinate, side=side)
            if hit:
                edge_hits.append(hit)
        if edge_hits:
            matches["image_edge"] = edge_hits

        reasons = [
            name
            for name in ("alignment_hole", "image_edge", "physical_inconsistency")
            if name in matches
        ]
        return {
            "excluded": bool(reasons),
            "reasons": reasons,
            "matches": matches,
            "edge_local_coordinate_provenance": edge_local_provenance,
        }


def _load_optional_local_coordinates(method: str, side: str, count: int) -> list | None:
    candidates = [
        f"predicted_etch_pits/{method}/{side}_local_coordinates.txt",
    ]
    if method == "CNN" and side == "front":
        candidates.append("predicted_etch_pits/CNN/IXY_allchan_008thresh.txt")
    for relative in candidates:
        path = artifact_path(relative)
        if path.is_file():
            local = mhio.load_json(relative)
            if len(local) != count:
                raise ValueError(
                    f"{relative} contains {len(local)} coordinates for {count} "
                    f"{method} {side} predictions"
                )
            return local
    return None


def apply_prediction_fiducials(config_path: str | Path) -> dict:
    """Filter all six front/back prediction lists and retain a complete audit."""
    path, config, digest = load_fiducial_regions(config_path)
    evaluator = FiducialRegionEvaluator(config)
    audit = {
        "schema_version": 1,
        "fiducial_region_file": str(path),
        "fiducial_region_sha256": digest,
        "methods": {},
    }

    for method in METHODS:
        method_audit = {}
        for side in SIDES:
            suffix = SIDE_SUFFIX[side]
            source_relative = f"predicted_etch_pits/{method}/final_{suffix}.txt"
            output_relative = f"predicted_etch_pits/{method}/fiducial_{suffix}.txt"
            removed_relative = (
                f"predicted_etch_pits/{method}/fiducial_removed_{side}.json"
            )
            points = mhio.load_json(source_relative)
            local_points = _load_optional_local_coordinates(method, side, len(points))
            retained = []
            removed = []
            reason_counts = {
                "alignment_hole": 0,
                "image_edge": 0,
                "physical_inconsistency": 0,
            }
            for index, point in enumerate(points):
                local = None if local_points is None else local_points[index]
                classification = evaluator.classify(
                    point, side=side, local_coordinate=local
                )
                if classification["excluded"]:
                    for reason in classification["reasons"]:
                        reason_counts[reason] += 1
                    removed.append(
                        {
                            "id": f"{method}_{side[0].upper()}{index:06d}",
                            "source_index": index,
                            "coordinate": point,
                            **classification,
                        }
                    )
                else:
                    retained.append(point)
            mhio.save_json(retained, output_relative)
            mhio.save_json(removed, removed_relative)
            method_audit[side] = {
                "source": source_relative,
                "output": output_relative,
                "source_count": len(points),
                "retained_count": len(retained),
                "removed_count": len(removed),
                "removed_by_reason": reason_counts,
                "edge_local_coordinate_provenance": (
                    "provided" if local_points is not None else "inferred"
                ),
            }
        audit["methods"][method] = method_audit

    destination = artifact_path(
        "predicted_etch_pits/fiducial_cut_audit.json", for_write=True
    )
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    return audit

