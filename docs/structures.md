# Structures

Crystal structures in `httk.atomistic` follow the *httk₂* view/backend pattern:
one family of backends (`UnitcellStructure`, `FundamentalDomainStructure`,
`PlainStructure` — the last defined in `httk.atomistic.models.structure.plain`, not
re-exported from `httk.atomistic` — records, ...) presented through views, with
`StructureLike` naming everything a function accepts. Loading is one call, and any member
converts to any view by class conversion:

```python
from httk.atomistic import PlainStructureView, UnitcellStructureView

unitcell = UnitcellStructureView("example.cif")   # the full cell, exactly
lattice, positions, numbers = PlainStructureView(unitcell)  # spglib-like triple
```

Every backend produces the same canonical quartet — `cell`, `sites`,
`species`, `species_at_sites` — and views build their presentation from it;
there is no pairwise conversion between representations. `unwrap()` always
recovers the exact original.

The full guide, {doc}`details/structures`, covers `DatastreamStructure` and
lazy remote sources, the component families (`Cell`, `Sites`, `Species`),
exact geometry (surd matrices, Cartesian positions), the numeric float/numpy
layer, POSCAR loading, supercells, serving structures over OPTIMADE, and
`unwrap`/`unview` semantics.
