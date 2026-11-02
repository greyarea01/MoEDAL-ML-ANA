# Clean-pit ID registries

Each JSON file in this directory is an immutable clean-pit identity system.
Registries are pipeline inputs, never generated inside a pipeline run. Select
one explicitly when starting an analysis:

```bash
--clean-id-registry clean_pit_id_registries/paperV1.json
```

A registry is an ordered JSON list of records with a unique permanent `Cxxxx`
ID and a two-value global C1 coordinate. Pipeline truth reconstruction must
match the complete selected registry before run-specific cuts are applied.
Removing a pit from an analysis never renumbers any surviving pit.

`paperV1.json` contains the exact ordered JSON records from the historical
`predicted_list/ids/Aligned_clean_with_ids.txt` and preserves all 1,242 IDs
used by the manuscript, galleries, and angle surveys. The registry file adds a
terminal newline; pipeline ID output retains the historical serialization.
