# Etch-pit incident-angle survey

The active analysis is the image-based ellipse pipeline in
`ellipse_incident_angle/`. It measures an incident angle independently on C1
and C2 from the fitted shape of the same clean etch pit.

The former C1-to-C2 coordinate-displacement analysis is preserved for audit in
`retired_coordinate_displacement_angle/`. It is not used by the active
incident-angle pipeline.

## Active pipeline

Run from the repository root:

```bash
python etch_pit_angle_survey/ellipse_incident_angle/run_pipeline.py
```

The command recreates the result tables, summaries, and three PDF reports. See
`ellipse_incident_angle/README.md` for the exact segmentation, ellipse fit,
quality controls, edge handling, inputs, and outputs.

## Directory layout

```text
etch_pit_angle_survey/
  ellipse_incident_angle/                 active image-based pipeline
  retired_coordinate_displacement_angle/ historical coordinate-based method
```
