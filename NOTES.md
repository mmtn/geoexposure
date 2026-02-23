# Development notes

### Conceptual structure

- Set up `Environment`:
  - Defines temporal bins and spatial grid / sampling scheme.
  - Holds `EnvironmentalData` layers and a list of metrics.

- For each `Trajectory`:
  - Split into windows based on temporal resolution
  - For each time window:
    - `Mobility` &rarr; gives occupancy distribution.
      - For each metric in metrics:
        - `Metric` asks `EnvironmentalData` layers for the values it needs in this window.
        - `Metric` returns exposure values for that window.
