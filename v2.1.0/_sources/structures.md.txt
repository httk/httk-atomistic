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

`Structure` is the exact-geometry, assigned-species cell of the three-row
material-information taxonomy; {doc}`prototypes` lays out that matrix and the
coarser levels (`Structuretype`, `Protostructure`, `Prototype`, and
`Formulatype`) that a structure projects to. A middle-row value may additionally carry an exact representative
and/or a discriminator when the user constructs it with them (recognition and
derivation return a base value); those optional fields affect equality and content
identity while remaining separate from its label.

`structure_delta(first, second)` is the public total Cartesian atom travel
between compatible exact representatives after common-subgroup alignment. It
uses each endpoint's cell and shortest periodic images; it is not a label or
content-id distance.

The full guide, {doc}`details/structures`, covers `DatastreamStructure` and
lazy remote sources, the component families (`Cell`, `Sites`, `Species`),
exact geometry (surd matrices, Cartesian positions), the numeric float/numpy
layer, POSCAR loading, supercells, serving structures over OPTIMADE, and
`unwrap`/`unview` semantics. Storage tables for the retained middle-row records
are `atomistic_prototype` and `atomistic_protostructure`; stores using the
retired layout must be rebuilt.
