# Reading LAMMPS dumps and data files

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

## Data files

`httk.core.load` reads LAMMPS data files (the `read_data` format) and
`httk.core.save` writes them. Data file names such as `data.si` or `in.lmp`
are ambiguous, so the reader claims no extension and the format is always
selected explicitly: `load("data.si", format="lammps-data")`. `save(s, "x.lmp")`
selects the writer by the `.lmp` extension. Only `atomic` atom-style files are
read: a `# atomic` comment on the `Atoms` line (5 columns, or 8 with image flags
`id type x y z ix iy iz`), or no style comment and exactly 5 columns. Any other
declared style is refused.

```python
import tempfile
from pathlib import Path

from httk.core import load, save

text = """Si pair

2 atoms
1 atom types

0.0 4.0 xlo xhi
0.0 4.0 ylo yhi
0.0 4.0 zlo zhi

Masses

1 28.0855 # Si

Atoms # atomic

1 1 0.0 0.0 0.0
2 1 2.0 2.0 2.0
"""

with tempfile.TemporaryDirectory() as folder:
    source = Path(folder) / "data.si"
    source.write_text(text)
    structure = load(str(source), format="lammps-data")
    save(structure, str(Path(folder) / "copy.lmp"))
    print(structure.species_at_sites, structure.cell.volume)
```

The box gives the cell exactly (orthogonal, `xy xz yz` restricted triclinic, or
general `avec`/`bvec`/`cvec` with `abc origin`); positions are the wrapped
Cartesian coordinates relative to the box origin, ordered by atom id. Image
flags, `Velocities` and bonded sections are ignored.

Data files carry no units, so lengths are read as Å. `units="metal"` (default)
and `units="real"` are accepted; `lj`, `si`, `cgs`, `electron`, `micro` and
`nano` are refused. Data files also name no elements. Per atom type the species
come from, in order: the `species=` keyword (a mapping of type id to name, or a
sequence in type order), an `Atom Type Labels` section, `Masses` line comments
(`1 26.98 # Al`), then a `types: Si O` list in the first comment line (as the
*httk* writers produce). The element is read from the name, never inferred from
a mass; if no source names a type, pass `species=`. `Masses` values are kept as
the species masses.

`save` writes `Masses` (omitted when some species has no mass; override with
`masses={"Si": 28.0855}`), an `Atom Type Labels` section so the file reads back
without `species=`, and `Atoms # atomic` (the only `atom_style`). A cell that is
already LAMMPS-framed (`a` along x, `b` in the xy plane, `a_x`, `b_y`, `c_z`
positive) is written exactly. Any other cell is rotated into that frame in
floating point and written to 16 significant digits, so saving an unframed cell
is lossy at that level (and the lattice orientation changes). Left-handed cells
are refused. A neutral payload loaded with `raw=True` is saved back verbatim.
