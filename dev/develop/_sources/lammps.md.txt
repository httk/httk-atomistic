# Reading LAMMPS dump trajectories

*httk-atomistic* reads native LAMMPS text dumps through
`httk.atomistic.integrations.lammps`. The low-level `LammpsDumpFile` preserves
every numeric token as an exact rational value. `LammpsTrajectory` converts the
stream into exact `UnitcellStructure` frames and canonical ångström,
picosecond, and electronvolt observables.

Construction is explicit because a dump does not reliably carry species or
units. Atom types are never interpreted as atomic numbers:

```python
from httk.atomistic.integrations.lammps import LammpsTrajectory

trajectory = LammpsTrajectory(
    "atoms.dump.gz",
    species={1: "Si", 2: "O"},
    units="metal",
    timestep="0.001",
)

for structure, (step, positions) in trajectory.samples("step", "cartesian_positions"):
    print(step, structure.cell.volume, positions[0])
```

`frames()` and `samples()` reopen the source and keep memory bounded by one
frame. `observable(name)` returns a tuple and therefore materializes that one
observable. `nframes` and indexed access re-stream the file; the path-backed
source may change between traversals.

## Supported native subset

The reader accepts atom/custom text snapshots with positive, unique `id` and
`type` columns. It supports:

- orthogonal boxes with boundary flags;
- restricted triclinic `xy xz yz` boxes, including negative tilt factors;
- general triclinic `abc origin` boxes;
- exactly selected `x y z`, `xs ys zs`, `xu yu zu`, or `xsu ysu zsu`
  coordinates;
- complete `ix iy iz`, `vx vy vz`, and `fx fy fz` triplets; and
- optional `ITEM: UNITS` and `ITEM: TIME` records.

Unknown per-atom columns are retained only while parsing the current row and
otherwise ignored. Incomplete recognized triplets, duplicate IDs, changing
IDs or types, invalid numbers, invalid cells, and truncated snapshots raise
`ValueError`. If more than one coordinate representation is present, pass
`coordinates=` explicitly.

General triclinic headers do not state boundary flags. Pass a three-boolean
`periodicity=` value for those files. Structures use positions relative to the
box origin; the `cartesian_positions`, `unwrapped_positions`, and `cell_origin`
observables retain the laboratory frame.

## Units, time, and segments

`units="metal"` maps length, time, and energy directly to Å, ps, and eV.
`units="real"` converts fs to ps and kcal/mol to eV using the exact SI values
of the electronvolt and Avogadro constant. Reduced `units="lj"` requires
`lj_scales=(length_A, time_ps, energy_eV)`, with three positive explicit
factors. A printed unit style must match the requested style.

The `time` observable uses printed times when present. Otherwise it is exposed
only when `timestep=` supplies source-time units per printed step. Printed time
therefore supports simulations whose time step changes during the run.

A non-increasing printed step starts a new zero-based segment. A source with
multiple segments raises during traversal unless `segment=` selects one. The
selected segment must keep atom identity, types, periodicity, unit metadata,
and field availability constant.

The native subset does not include YAML, binary dump formats, local/grid
records, arbitrary computed-column semantics, or a thermodynamic-log join.
