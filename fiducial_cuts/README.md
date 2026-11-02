# Clean-truth fiducial cut files

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
