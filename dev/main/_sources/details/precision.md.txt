# Data precision in detail

A data file states its numbers to a definite number of digits, and that is a claim.
`0.3333` says a coordinate is known to about `1e-4` of a cell edge; `0.33` says only
`1e-2`. httk records that claim and uses it, so a matching tolerance or an spglib
`symprec` follows the data instead of being a constant somebody guessed.

```python
from httk.atomistic import structure_tolerance
from httk.core import load

asu = load("measured.cif")
asu.coordinate_precision      # Fraction(1, 10000) — from the file's own digits
asu.cell.precision            # Fraction(3, 10000) — widened by a stated esd
structure_tolerance(asu)      # a matching tolerance derived from both
```

## What is recorded

Two values, on the objects that own the numbers they describe.

| | Where | Units |
| --- | --- | --- |
| coordinate precision | {py:class}`~httk.atomistic.Sites` | fractional (dimensionless) |
| basis precision | {py:class}`~httk.atomistic.Cell` | the cell's length units |

Both are exact rationals — `1e-4` is `Fraction(1, 10000)` — so nothing picks up binary
noise on the way to becoming a tolerance. `None` means **unknown**, which is a real answer
and not the same as claiming exactness. A non-positive value is rejected: zero would assert
an exactness no measurement has.

`UnitcellStructure` reads both through, and adds the value a tolerance actually wants:

```python
structure.coordinate_precision     # fractional
structure.basis_precision          # a length
structure.cartesian_precision()    # the coordinate precision as a length
```

`cartesian_precision()` multiplies by the **longest** cell edge — the largest displacement
a fractional uncertainty can produce — and takes the cell's own precision as a floor, since
a cell known only to `1e-3` cannot place an atom better than that however many digits the
coordinates carry. The same `1e-4` coordinates therefore mean `5e-4` in a 5 Å cell and
`3e-3` in a 30 Å one, which is the entire reason a fractional precision has to be converted
before it is used as a distance.

```{admonition} Precision is not part of identity
:class: note

It takes no part in `==`, on `Cell`, `Sites`, `UnitcellStructure`, or in
{py:func}`~httk.atomistic.same_crystal`. The same atoms read from a more carefully written
file are still the same structure. (`Cell.__eq__` already ignored the `scale`/`unscaled`
factoring for the same reason.)
```

## Where it comes from

Any source that writes its numbers to a definite number of digits infers a precision from
them.

**CIF** states both a number of digits and, often, an explicit standard uncertainty. Where
both are present the **coarser** wins, that being the weaker and therefore safer claim — so
`5.6402(3)` is precise to `3e-4`, not to the `1e-4` its four decimals alone would suggest.

**POSCAR** has no uncertainty concept, so the digits are all there is. Two conversions
happen when the structure is built, because they need the assembled cell:

- Cartesian coordinates are a *length*, so they are divided by the shortest cell edge to
  become fractional.
- The cell vectors are multiplied by the scaling factor, so their precision scales with
  them. The scaling factor's **own** digits are deliberately not charged as uncertainty: it
  defines units rather than being measured apart from the rows it scales, and charging it
  would declare a 5.64 Å cell precise to only half an ångström because the file wrote
  `1.0`.

Across a set of values the **coarsest** wins: a structure is only as precisely stated as
its least precisely stated number, so one sloppy `0.5` in a table of six-decimal
coordinates really does mean the table is good to `1e-1`.

A value written as an exact rational — `1/3` — states a value rather than a measurement and
contributes nothing at all, rather than dragging the table down to the width of its last
digit.

The underlying primitive is in httk-core and is usable directly:

```python
from httk.core import combined_precision, decimal_precision

decimal_precision("0.3333")      # Fraction(1, 10000)
decimal_precision("1.2e-3")      # Fraction(1, 10000)
decimal_precision("1/3")         # None — exact, not measured
combined_precision(["0.123456", "0.5"])   # Fraction(1, 10) — the coarsest
```

## What it is used for

{py:func}`~httk.atomistic.structure_tolerance` turns a recorded precision into a matching
tolerance, and it is the default for every tolerant operation in the package:
{py:func}`~httk.atomistic.recognize_asu`, `ASUStructureView`,
{py:func}`~httk.atomistic.asu_structure_from_cif`, and spglib's `symprec`. Passing an
explicit `tolerance=` still overrides it.

The tolerance is the Cartesian precision times two — two independently rounded values can
differ by twice their precision, so anything tighter would reject data that is in fact
consistent — capped so that it can never reach half the closest approach between two sites,
which is what would let genuinely distinct atoms merge.

As a calibration, coordinates written to four decimals in a 5 Å cell derive exactly `1e-3`,
matching the fallback used when precision is unknown. Recorded precision lets other data
derive an appropriate value.

### Why it matters

Consider a CIF written to three decimals whose site is meant to sit on Wyckoff `4e` of
space group 15 (`0, y, 1/4`), rounded to `0.001 0.333 0.251` — 0.005 Å off in *x* and
0.007 Å off in *z*:

| tolerance | outcome |
| --- | --- |
| fixed `1e-3` | misses the special position, lands on the general position `8f`, generates **8 atoms** |
| derived `0.014 Å` | recognised as `4e`, generates **4 atoms** |

Using the recorded precision avoids silently producing a structure with twice the correct
number of atoms in this case.

```{admonition} Minimum separation is a cap, not a precision
:class: important

The minimum separation bounds a derived tolerance from above. Treating it as part of the
recorded precision would be incorrect: two accidentally close atoms would make a structure
look far more precisely stated than it is.
```

## Serving it

Two provider-specific OPTIMADE properties, described by published definitions from
[schemas.httk.org](https://schemas.httk.org) rather than paraphrased locally:

| Property | Definition | Unit |
| --- | --- | --- |
| `_httk_coordinate_precision` | `core/fractional_coordinate_precision` | dimensionless |
| `_httk_basis_precision` | `core/length_precision` | ångström |

Both serve `null` for a structure that states no precision. OPTIMADE standardises nothing
here — the only `standard_uncertainty` in its structures definition is CODATA metadata
inside a unit definition — so these carry the `_httk_` prefix OPTIMADE requires of a
database-specific property name, while the definitions keep their own `$id`.

Storing precision needs nothing special: `float` and `Fraction` are already first-class
scalar kinds in httk-store.

## See also

- {doc}`asu` — where the derived tolerance is used
- {doc}`/examples/precision` — a runnable walkthrough
