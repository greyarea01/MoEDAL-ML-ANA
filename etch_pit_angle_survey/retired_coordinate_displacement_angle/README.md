# Retired coordinate-displacement angle method

This directory preserves the former Lewis and Kanik C1-to-C2 coordinate
surveys and their comparison for reproducibility. These files are historical
outputs and are not inputs to the active incident-angle calculation.

## Why this method was retired

The method transformed C1 and C2 pit locations into one global frame, measured
their lateral separation, assumed a 200 micrometre distance between surfaces,
and calculated:

```text
coordinate_angle_degrees = degrees(atan(lateral_separation_micrometres / 200))
```

That angle is too sensitive to inter-surface registration error for this data:

- The checked datum-2 mismatch is approximately 5.13 pixels.
- The selected Lewis C1-to-C2 separation has a mean of approximately 4.12
  pixels and a maximum of approximately 11.60 pixels.
- The datum mismatch is therefore larger than the mean displacement used as
  the angle signal and is a substantial fraction of even the largest selected
  displacement.
- Candidate pairing required a strict separation below 12 pixels. The same
  quantity was then converted into an angle, which constrained the reported
  Lewis distribution below approximately 18.57 degrees by construction.
- Multiple C1, Cr, and C2 candidates also made some coordinate associations
  ambiguous.

Consequently, the coordinate angle can contain registration residuals of the
same size as the intended physical signal. The active method instead estimates
an incident angle independently from each pit's ellipse on each surface, then
uses the C1-to-C2 difference only as a consistency check.

## Preserved contents

- `lewis/`: reconstruction and all Lewis candidate tables.
- `kanik/`: reconstruction and all Kanik candidate tables.
- `comparison/`: Lewis/Kanik comparisons and the old detection plots.

The archived scripts remain runnable from their new paths, but they must not be
used as the current incident-angle result. Their own README files describe the
historical procedure and outputs.
