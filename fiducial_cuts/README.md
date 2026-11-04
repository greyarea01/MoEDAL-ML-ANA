# Fiducial cut files

An analysis may pass zero or more cut files with repeated
`--fiducial-cut-file` arguments. Cuts exclude registered clean IDs while
preserving every surviving ID. They never create or renumber IDs.
Cut files are immutable pipeline inputs and must be stored in this directory.

Schema version 1 is:

```json
{
  "schema_version": 1,
  "clean_id_registry_sha256": "SHA-256 of the complete selected registry",
  "description": "Human-readable purpose of this cut",
  "exclude_clean_ids": [
    {"id": "C0123", "reason": "Explicit physical/fiducial reason"}
  ]
}
```

The registry hash prevents a cut file from being applied to the wrong identity
system. The pipeline stores active IDs, excluded IDs, reasons, selected source
paths, and hashes inside each run.

Spatial fiducial files are selected separately with
`--spatial-fiducial-file`. They are applied before front/back alignment to all
prediction lists and to the immutable clean-pit registry. Raw preprocessed
coordinates remain unchanged; filtered coordinate lists and a removal audit are
stored in the run. `analysis_v2_regions.json` defines:

- four alignment-hole disks with 184 px radius;
- the strictly-less-than-4 px border of every evaluation `d`/`dr` image, using
  exact source-local coordinates when they are available; and
- 8 x 8 px axis-aligned boxes around the ten reviewed physical-inconsistency
  features.

Circle and box boundaries are inclusive. The edge convention retains points
exactly 4 px from an image edge, matching the existing DCC preprocessing.

For analysis v2, use both run inputs together:

```text
--fiducial-cut-file fiducial_cuts/analysis_v2_truth_blacklist.json
--spatial-fiducial-file fiducial_cuts/analysis_v2_regions.json
```

The blacklist excludes `C0123`, `C0300`, and `C0997` from that analysis only.
It does not edit the global registry or renumber any surviving clean pit.
