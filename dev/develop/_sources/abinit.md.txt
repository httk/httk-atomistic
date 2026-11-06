# ABINIT input and output files

*httk-atomistic* reads the structure of ABINIT input files and the geometries
an ABINIT main output echoes, and writes the structure variables of an ABINIT
input. The readers live in `httk.atomistic.integrations.abinit.io`;
`httk.core.load` and `httk.core.save` dispatch to them by extension:

| File | `load` returns | `save` writes | Format tag | Extension |
| --- | --- | --- | --- | --- |
| ABINIT input | structure | structure variables only | `abinit-input` | `.abi` |
| ABINIT main output | trajectory | — | `abinit-output` | `.abo` |

Generic names such as `si.in` or `run.out` load with an explicit tag:
`load("si.in", format="abinit-input")`, `load("run.out", format="abinit-output")`.
Compressed files (`.abi.gz`, `.abo.bz2`) are decompressed transparently.

## Reading inputs

```python
import tempfile
from fractions import Fraction
from pathlib import Path

from httk.core import SurdVector, load

text = """
# Diamond silicon
acell 3*10.26          # Bohr unless a unit follows
rprim 0 1/2 1/2  1/2 0 1/2  1/2 1/2 0
natom 2 ntypat 1 typat 2*1 znucl 14
xred 0 0 0  1/4 1/4 1/4
ecut 8                 # calculation variables are ignored
"""
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "si.in"
    path.write_text(text)
    structure = load(str(path), format="abinit-input")

bohr = Fraction("0.52917720859")
side = Fraction("5.13") * bohr
assert structure.cell.basis == SurdVector([[0, side, side], [side, 0, side], [side, side, 0]])
assert [species.name for species in structure.species] == ["Si"]
```

The reader takes ABINIT's syntax: whitespace-separated `name value...`
pairs with case-insensitive names, `#` and `!` comments, `n*value`
repetition, Fortran `d` exponents, and the number expressions `a/b`,
`sqrt(x)` and `-sqrt(x)`, all kept exact (`sqrt(0.75)` is the surd
√3/2). It reads `acell` (default `3*1`, with an optional `Bohr` or `Angstrom`
unit), `rprim` (default identity) or `angdeg`, `scalecart`, `natom`, `ntypat`,
`typat`, `znucl`, and one of `xred` (the default, all zero), `xcart` (Bohr
unless a unit follows) or `xangst`. Cell vector *j* is `acell(j)` times row
*j* of `rprim`, with component *i* multiplied by `scalecart(i)`. All other
variables are ignored.

`angdeg` cells are built by ABINIT's documented construction (trigonal axis
along *z* for three equal angles, otherwise the first vector along *x*)
whenever that construction is exact: angles of 30, 45, 60, 90, 120, 135 or
150 degrees whose square roots stay simple surds. Other angles, such as
`angdeg 90 104.5 90`, are read as lattice parameters (`acell` lengths,
`angdeg` angles) and built as for CIF: the lengths and angles are ABINIT's
(with lengths in Bohr the approximation also shifts *b* and *c* by about
1e-11 Å), the basis is a deterministic rational approximation (about 1e-10) in the
lattice-parameter frame (*a* along *x*, *b* in the *xy* plane), and the cell
precision comes from the digits of the `acell` tokens. Fractional positions
do not depend on the frame. Cartesian positions (`xcart`, `xangst`) and a
`scalecart` other than `3*1` do, so with such angles they are refused; give
an explicit `rprim` instead.

Lengths in Bohr convert with ABINIT's own Bohr radius, 0.52917720859 Å
(ABINIT 10 output echoes `xcart 2.565` as `xangst 1.3573395400`), exactly.
The cell entries are exact products `acell × rprim × scalecart`; their
precision is propagated from the digits of those tokens. In `rprim` and
`scalecart`, which are dimensionless, `0.0`, `0.5` and `1.0` spell exact
fractions and claim nothing, as in POSCAR's direct coordinates; exact tokens
such as `1/2`, `sqrt(0.75)`, integers and the defaults claim nothing either.
For `si.abi`, `acell 3*10.26` with `rprim 0.0 0.5 ...` claims
0.01 × 0.5 = 0.005 Bohr. Position tokens keep their own digit precision.
Pass `precision=` (Å) to set the structure's Cartesian precision explicitly.

Species come from `typat` and `znucl` and are named by element. Two types with
the same element are numbered, `Fe1` and `Fe2`. The file cannot carry species
names, so names other than these do not survive a round trip, as with POSCAR.

Refused with a `ValueError` rather than misread: dataset- or image-dependent
structure variables (`acell1`, `xred:`, `xred+`, `xred?`, `xred_lastimg`, ...),
`spgroup`/`spgroupma`/`natrd`/`nobj` symmetry-generated inputs, the
`structure "..."` variable, `xyzfile`, `getxred`/`getxcart`/`getcell`, missing
`znucl` (ABINIT would take it from the pseudopotentials), counts that disagree
with `natom`/`ntypat`, non-integer `znucl`, positions given twice, duplicate
variables, unknown unit words, and non-UTF-8 files. Dataset-independent
structure variables in a multi-dataset input are read.

## Writing inputs

`save(structure, "x.abi")` writes only the structure variables; for the
FeO cell of the example below:

```text
acell 1 1 1 Angstrom
rprim
 3.5 0 0
 0 3.5 0
 0 0 4.25
natom 2
ntypat 2
typat
 1 2
znucl 26 8
xred
 0 0 0
 0.5 0.5 0.5
```

The cell rows are in Å and positions fractional, with POSCAR's number
tokens (finite decimals exact, other values to 16 significant digits). Add the
calculation variables yourself, or use `httk.codes.abinit` from
*httk-workflow-abinit*. Species masses, names and precision are not written;
disordered or partially occupied species and non-periodic cells are refused.
The writer takes no options. A payload loaded with `load(path, raw=True)` is
written back byte for byte.

```python
import tempfile
from pathlib import Path

from httk.atomistic import Species, UnitcellStructure
from httk.core import load, save

structure = UnitcellStructure(
    [["3.5", 0, 0], [0, "3.5", 0], [0, 0, "4.25"]],
    [[0, 0, 0], ["0.5", "0.5", "0.5"]],
    [Species("Fe", ("Fe",), (1,)), Species("O", ("O",), (1,))],
    ["Fe", "O"],
)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "feo.abi"
    save(structure, str(path))
    assert path.read_text().startswith("acell 1 1 1 Angstrom\n")
    back = load(str(path))

assert back.cell.basis == structure.cell.basis
assert back.sites.reduced_coords == structure.sites.reduced_coords
```

## Reading main outputs

`load("run.abo")` returns a `Trajectory`. Its first frame is the
`-outvars: echo values of preprocessed input variables` block; the
`-outvars: echo values of variables after computation` block is a second
frame only when its geometry differs, so a single-point run gives one frame
and `trajectory.frame(-1)` is always the final geometry. Only `acell`,
`rprim`, `xred` (or `xcart`), `natom`, `ntypat`, `typat` and `znucl` are read;
the rounded `xangst` echo never is.

Not read in this release: the per-iteration geometries of a relaxation or
molecular-dynamics run (only the initial and final echoes become frames), and
energies or forces (use `httk.codes.abinit` for those). Multi-dataset outputs
and outputs echoing `scalecart` are refused.
