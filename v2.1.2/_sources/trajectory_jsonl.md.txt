# Trajectory JSON Lines

`JsonlTrajectory` is the lazy atomistic view of the OPTIMADE trajectory
JSON Lines holding format. It is loaded through
`httk.core.load("run.traj.jsonl")` and saved with
`httk.core.save(trajectory, "run.traj.jsonl")`; the serializer passes a
generator of frame mappings to the reader/writer layer, so frames are not
collected into a list. Compressed destinations such as `run.traj.jsonl.gz` use
the normal core text compression path.

## The holding format

The reader/writer layer lives in `httk.atomistic.io.optimade_jsonl`. The public
filename convention is `.traj.jsonl`; the loader registers `.jsonl` because core
strips one compression suffix and dispatches on the remaining final suffix. Thus
`.traj.jsonl.gz` works through the normal text datastream/compression path.

The first line is an OPTIMADE 1.2.0 dense partial-data header with an
`x-httk-trajectory` description. Subsequent lines are frame objects containing
`index`, `fractional_site_positions`, and `observables`; variable-cell files
also contain `lattice_vectors`. See the `httk.atomistic.io.optimade_jsonl`
module docstring for the normative schema.

The format is float64 presentation data and intentionally has no exact-token
channel. Binary framing is not part of this format; its framing and
random-access trade-offs remain a separate design decision.

`TrajectoryJsonlFile.path` returns the source filename string used to construct
the lazy reader.
