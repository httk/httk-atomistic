# Quantum ESPRESSO pw.x files

`httk.core.load` reads the geometry of Quantum ESPRESSO `pw.x` inputs as an exact
`UnitcellStructure` and the geometries a `pw.x` run prints as a `Trajectory`;
`httk.core.save` writes the geometry part of a `pw.x` input. The readers live in
`httk.atomistic.integrations.qe.io` (`read_pw_input`, `read_pw_output`).

| File | Extension | Format tag | `load` gives | `save` |
| --- | --- | --- | --- | --- |
| `pw.x` input | `.pwi` | `qe-pw-input` | structure | geometry only |
| `pw.x` output | `.pwo` | `qe-pw-output` | trajectory | — |

Inputs and outputs are usually called `si.in` and `pw.out`, which no extension can
dispatch: pass the tag, `load("si.in", format="qe-pw-input")` or
`load("pw.out", format="qe-pw-output")`. Compressed files (`si.pwi.gz`) load directly.

## Inputs

```python
import tempfile
from pathlib import Path

import httk.core

text = """&control
   calculation = 'scf'   ! namelists other than &system are skipped
/
&system
   ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1, ecutwfc = 12
/
ATOMIC_SPECIES
 Si  28.085  Si.upf
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp) / "si.in"
    source.write_text(text)
    structure = httk.core.load(str(source), format="qe-pw-input")
    print(structure.cell.basis)

    httk.core.save(structure, str(Path(tmp) / "si.pwi"), pseudopotentials={"Si": "Si.pz-vbc.UPF"})
    print((Path(tmp) / "si.pwi").read_text())
```

The reader interprets the `&system` keys `ibrav`, `celldm(1..6)`, `A`, `B`, `C`,
`cosAB`, `cosAC`, `cosBC`, `nat` and `ntyp`, and the cards `ATOMIC_SPECIES`,
`CELL_PARAMETERS {alat|bohr|angstrom}` and `ATOMIC_POSITIONS {alat|bohr|angstrom|crystal}`.
It follows Fortran namelist syntax: case-insensitive names, `,` or newline separators,
`!` comments, quoted strings, `.true.`, `1.0d0` exponents, indexed names such as
`celldm(1)`, list assignments and `r*value` repeats. Other namelists and cards are parsed
or delimited and skipped.

- `ibrav = 0` and every lattice of the INPUT_PW `ibrav` table — 1, 2, 3, -3, 4, 5, -5,
  6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14 — are built exactly; square roots
  (hexagonal, trigonal, monoclinic and triclinic cells) stay exact surds when that is
  cheap. A root whose radicand is too long (e.g. a 10- or 16-digit `cosAB`) becomes a
  deterministic rational approximation, as for CIF cell parameters, and its error bound
  (about 1e-11 of the length) is added to the claimed precision.
- alat is `celldm(1)` (Bohr) or `A` (Å). With `CELL_PARAMETERS bohr|angstrom` it is the
  length of the first vector, computed only when an `alat`-unit card needs it (approximated,
  with its error claimed, when it is not cheaply exact). `CELL_PARAMETERS` without an
  option means `alat` when a lattice parameter is given and `bohr` otherwise;
  `ATOMIC_POSITIONS` without an option means `alat`. Trailing `if_pos` columns are accepted and ignored.
- Species names, masses and pseudopotential names come from `ATOMIC_SPECIES`. The element
  follows QE's label rule: an element symbol (any case) optionally followed by digits or by
  `_x`/`-x`, at most 3 characters, as in `Fe1` or `O_h`.

Refused with a `ValueError`: `space_group` and `ATOMIC_POSITIONS crystal_sg`, `ibrav` values
outside the table, `celldm` together with `A`-style parameters, a lattice parameter together
with `CELL_PARAMETERS bohr|angstrom`, a `CELL_PARAMETERS` card with `ibrav` other than 0,
algebraic expressions such as `1/3`, labels that name no element, site or species counts that
disagree with `nat`/`ntyp`, and non-UTF-8 or non-`pw.x` text.

Lengths in Bohr are converted with Quantum ESPRESSO's own constant,
`BOHR_ANGSTROM = "0.52917720859"` Å (CODATA 2006, as in QE 7.x), exactly. The precision
of decimal tokens is derived as for POSCAR (see {doc}`precision`). Values scaled by alat
claim the precision of their own digits and of the alat token
(`δt·alat + max|t|·δalat`); `ibrav` cells claim the alat token's precision times their
largest entry in alat units (the digits of `celldm(2..6)`, `B`, `C` and the cosines are not
propagated). Approximated square roots add their error bound. `precision=` overrides all of
these.

## Writing

`save(structure, "x.pwi")` writes `&system` (`ibrav = 0`, `nat`, `ntyp`), `ATOMIC_SPECIES`,
`CELL_PARAMETERS angstrom` and `ATOMIC_POSITIONS crystal`, with numbers exact where they are
finite decimals. Calculation parameters (`&control`, cut-offs, `K_POINTS`) are not written;
add them, or write complete inputs with `httk.codes.qe.write_pw_input` from
*httk-workflow-qe*. Writer keywords:

- `masses={"Si": 28.0855}`: masses by species name; otherwise the species mass, else `1.0`
  (pw.x needs a number there; it only matters for dynamics).
- `pseudopotentials={"Si": "Si.pbe.UPF"}`: file names by species name; otherwise
  `<Element>.UPF`.

Species names must be valid QE labels for their element (see above); rename others first.
Saving the raw payload of a loaded input, `save(load(p, format="qe-pw-input", raw=True),
q)`, writes the original text unchanged; writer keywords are refused for it.

## Outputs

`load("pw.out", format="qe-pw-output")` returns one frame per printed geometry:

- frame 0 is the header (`lattice parameter (alat)`, `crystal axes`, and the
  `site n. ... positions (alat units)` table). It is printed with only 4–7 decimals and is
  therefore lossy and claims that printed precision (digits and the 4-decimal alat); the
  `pw.x` input holds the exact initial geometry;
- every later `ATOMIC_POSITIONS` block is a frame on the most recent `CELL_PARAMETERS`
  cell (`vc-relax`) or else the header cell;
- the `Begin/End final coordinates` block is not repeated when it equals the last step.

A converged `relax` therefore gives the initial geometry plus one frame per ionic step,
and `trajectory.frame(-1)` is the relaxed structure. Species names and
masses come from the header's atomic species table. Energies, forces and other
observables are not read here.
