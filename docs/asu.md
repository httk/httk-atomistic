# Asymmetric units

A crystal is mostly repetition. `ASUStructure` represents the part that is not:
a space group, one site per symmetry orbit, and the values of that site's free
parameters. Expanding it produces the full unit cell; recognizing a full cell
produces it back.

```python
from httk.atomistic import UnitcellStructureView
from httk.core import load

asu = load("nacl.cif")                     # a CIF's native, declared symmetry
structure = UnitcellStructureView(asu)     # the full cell, exactly
```

`ASUStructure` is part of `StructureLike`, so it can be passed anywhere a
structure is accepted, and expansion is exact and lazy — reading the space
group never generates the cell.

## Canonicalization

Two crystals that are the same up to origin, cell-basis choice, site order, or
setting have many `ASUStructure` descriptions. `canonicalize` collapses that
freedom exactly: given an `ASUStructure` it returns the single deterministic,
highest-symmetry representative, using only exact rational arithmetic.

```python
from httk.atomistic import canonicalize

result = canonicalize(asu)          # exact input, exact answer
canonical = result.asu
```

For *measured* input — coordinates carrying noise — use `canonical_asu`, the
one-liner that recognizes the symmetry within a tolerance (with spglib) and then
canonicalizes the result exactly:

```python
from httk.atomistic import canonical_asu
from httk.core import load

asu = canonical_asu(load("measured.cif"))   # noisy input, canonical answer
```

It sweeps recognition over a few tolerance multiples and keeps the
highest-symmetry model whose atoms still sit within the base tolerance of the
input, so a boundary flip can be rescued without accepting extra noise. It always
works from the coordinates, so it can raise a declared symmetry the geometry
supports — or lower one it does not, at the derived tolerance.

Only the recognition step is floating-point: which symmetry is *accepted* near a
tolerance boundary can vary across platforms or spglib builds, but the exact
canonicalization erases spglib's representational freedom, so *how* an accepted
symmetry is represented never does. Free-parameter values are least-squares fits
of the measured coordinates: two noisy measurements of the same crystal reach the
same Wyckoff choices but slightly different rational parameter values.

The full guide, {doc}`details/asu`, covers what an asymmetric unit holds,
arbitrary and untabulated settings, the exactness contract of expansion,
tolerance-bearing recognition, round-tripping, reading CIFs, serving symmetry
over OPTIMADE, and the symmetry tables.
