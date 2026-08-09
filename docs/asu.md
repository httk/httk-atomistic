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

The full guide, {doc}`details/asu`, covers what an asymmetric unit holds,
arbitrary and untabulated settings, the exactness contract of expansion,
tolerance-bearing recognition, round-tripping, reading CIFs, serving symmetry
over OPTIMADE, and the symmetry tables.
