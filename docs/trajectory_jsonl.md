# Trajectory JSON Lines

`JsonlTrajectory` is the lazy atomistic view of the
`httk-trajectory-jsonl` holding format. It is loaded through
`httk.core.load("run.traj.jsonl")` and saved with
`httk.core.save(trajectory, "run.traj.jsonl")`; the serializer passes a
generator of frame mappings to *httk-io*, so frames are not collected into a
list. Compressed destinations such as `run.traj.jsonl.gz` use the normal core
text compression path.

The format is float64 presentation data and intentionally has no exact-token
channel. Binary framing is not part of this format; its design is deferred.
