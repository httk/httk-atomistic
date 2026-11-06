# CP2K input and restart files

*httk-atomistic* reads the structure of a CP2K input file and of the `.restart`
file CP2K writes (restart files use the input syntax), and writes a structure
as a `&SUBSYS` block. The format tag is `cp2k-input`. CP2K's file names are
shared with other codes (`.inp` with ORCA, `.restart` with LAMMPS binary
restarts), so no extension selects it: always pass `format="cp2k-input"`.

```python
import tempfile
from pathlib import Path

import httk.core

text = """&FORCE_EVAL
  &SUBSYS
    &CELL
      ABC 3.0 3.0 5.0
      ALPHA_BETA_GAMMA 90 90 120
    &END CELL
    &COORD
      SCALED .TRUE.
      Fe1 0 0 0
      O 0.5 0 0.5
    &END COORD
    &KIND Fe1
      ELEMENT Fe
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "feo.inp"
    path.write_text(text)
    structure = httk.core.load(str(path), format="cp2k-input")
    print(structure.cell.basis)  # exact: the 120 degree row holds sqrt(3)/2

    httk.core.save(structure, str(Path(tmp) / "geom.inp"), format="cp2k-input")
    print((Path(tmp) / "geom.inp").read_text())
```

The final geometry of a CP2K relaxation is the `<project>-1.restart` file:
read it with `httk.core.load("<project>-1.restart", format="cp2k-input")`.
Reading geometry from the main output (`.out`) is not supported; CP2K prints the
per-step geometry to the `-pos-1.xyz` trajectory instead.

## What loads

The reader takes the single `&FORCE_EVAL/&SUBSYS` section, or a file that holds
just one top-level `&SUBSYS` block (as the writer produces). Keywords are
case-insensitive; `!` and `#` comments and `\` line continuations are honoured.
Compressed files (`.gz`, `.bz2`) are decompressed.

- `&CELL`: either the `A`, `B` and `C` vectors, or `ABC` with
  `ALPHA_BETA_GAMMA` (default `90 90 90`). Lattice parameters are built as for
  CIF files, in CP2K's own frame (a along +x, b in the xy plane, c with positive
  z), so Cartesian `&COORD` rows mean what CP2K means. Angles with exact cosines
  (60, 90, 120 degrees, ...) give an exact cell, with square roots kept exactly;
  other angles (say a monoclinic 104.5) give a deterministic rational
  approximation of the cell vectors within about 1e-10 Å. A fully periodic
  cell keeps the exact lengths and angles as written; with a `PERIODIC` other
  than `XYZ` only the cell vectors are kept. `PERIODIC` (`XYZ` by default, also `XY`,
  `XZ`, `YZ`, `X`, `Y`, `Z`, `NONE`) becomes the cell's `periodicity`.
- `&COORD`: one `kind x y z` line per atom, Cartesian Å or fractional with
  `SCALED .TRUE.` (also `SCALED` alone, or `T`). A fifth (molecule-name) column
  is ignored.
- Species: one per kind name, named after it. Kind names are case-insensitive
  as in CP2K: `si` and `SI` are one species, named by the first spelling. The element is the kind's
  `&KIND` `ELEMENT` keyword; without one, the kind name must start with an
  element symbol, optionally followed by digits and a `_`/`-` suffix
  (`Fe1`, `Si_up`); otherwise the file is refused, never guessed.

Numbers keep the digits written in the file: the structure is exact, and the
precision of the decimal tokens is recorded as for POSCAR files; for `ABC` the
cell precision comes from the three lengths, as for CIF. Cartesian coordinates
are reduced through the cell: when its vectors are approximated, or the
fractional coordinates come out irrational (a Cartesian site in a hexagonal
cell), the site precision claims no better than that approximation (about
1e-10). Pass `precision=` (Cartesian, in Å) to override the coordinate precision.

## What is refused

Lengths must be in Å, CP2K's default: `[angstrom]` may be written, other units
(including `[bohr]` and `&COORD` `UNIT bohr`) are refused, and angles only accept
`[deg]`. Further refusals, each a `ValueError` naming the line:

- preprocessor directives (`@INCLUDE`, `@SET`, `@IF`, ...) and `$` variables:
  read the preprocessed input or the `.restart` file instead;
- several `&FORCE_EVAL` or `&SUBSYS` sections;
- external coordinates (`&TOPOLOGY` `COORD_FILE_NAME`, a `COORD_FILE_FORMAT`
  other than `OFF`) and `CELL_FILE_NAME`;
- `MULTIPLE_UNIT_CELL` other than `1 1 1`, `&CELL_REF`, and `SYMMETRY` other
  than `NONE`;
- ghost kinds (`GHOST`) and kinds whose element cannot be determined;
- files that are not CP2K input at all (not UTF-8 text, or no `&` sections),
  such as ORCA inputs or binary LAMMPS restart files.

## What saves

`httk.core.save(structure, path, format="cp2k-input")` writes a `&SUBSYS` block
meant to be included (`@INCLUDE`) in your own `&FORCE_EVAL`: a `&CELL` with the
`A`, `B`, `C` vectors in Å and `PERIODIC`, a `&COORD` with `SCALED .TRUE.` and
one `kind x y z` line per site, and a `&KIND name` / `ELEMENT symbol` section
only for species whose name is not its element symbol. Species names that
differ only in letter case are refused. Basis sets, potentials
and other `&KIND` settings are yours to add. The writer has no options.

Exact decimals are written exactly; irrational values such as `sqrt(3)/2` are
rounded to 16 significant digits. Species masses and anything other than the
geometry are not written. A payload loaded with `raw=True` is written back
byte-for-byte; writer options then raise a `ValueError`.
