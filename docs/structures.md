# Structures

This page documents practical usage of the structure classes in `httk.atomistic`.
It follows the same view/backend pattern as the datastream classes in `httk.core`.

## Overview

A crystal structure is available through one family of backends and views:

- backends: `StructureSimple` (wraps a `Structure`), `StructurePrimitive` (wraps an spglib-like triple)
- views: `StructureSimpleView` (presents any backend as a `Structure`), `StructurePrimitiveView` (presents any backend as a `(lattice, positions, numbers)` tuple)
- accepted union: `StructureLike`

Every backend produces the same canonical Simple quartet declared by `StructureAPI`:
`cell` (a `Cell` of 3x3 cell vectors), `sites` (a `Sites` of Nx3 reduced coordinates),
`species` (a tuple of `Species`), and `species_at_sites` (the species name at each site).
Views build their presentation from that quartet; there is no pairwise conversion between
representations.

In normal user code, you usually accept `StructureLike` and normalize immediately to one view.

## Common Calling Patterns

```python
from httk.atomistic import Structure, StructureSimpleView, StructurePrimitiveView

# A Structure (the Simple representation)
structure = Structure(
    cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    species=[
        {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
        {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
    ],
    species_at_sites=["Na", "Cl"],
)

# Structure in -> primitive triple out
lattice, positions, numbers = StructurePrimitiveView(structure)

# spglib-like triple in -> Structure out
triple = (
    [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    [11, 17],
)
as_structure = StructureSimpleView(triple)
```

## Component families

The `cell`, `sites`, and `species` components each get the same view/backend treatment as
`Structure` itself, mirroring the Structure family with `Class` in place of `Simple` (there
the word describes the representation, which still applies):

- `Cell`: backends `CellClass` / `CellPrimitive` / `CellParams`, views `CellClassView` /
  `CellPrimitiveView` / `CellParamsView`, union `CellLike`. `Cell` exposes `matrix` plus the
  derived `lengths`, `angles` (the crystallographic `alpha`/`beta`/`gamma` in degrees), and
  `volume`. The params representation is a flat `(a, b, c, alpha, beta, gamma)` 6-tuple
  (angles in degrees): a cell can be constructed from parameters anywhere a `CellLike` is
  accepted (the matrix is built with the standard orientation convention — first vector
  along x, second in the xy-plane), and `CellParamsView` presents any cell as its
  parameters, with the elements also available as the named properties `a`/`b`/`c`/
  `alpha`/`beta`/`gamma`. Note that parameters carry no orientation, so cell → params →
  cell reproduces lengths, angles, and volume but not the original orientation.
- `Sites`: backends `SitesClass` / `SitesPrimitive`, views `SitesClassView` /
  `SitesPrimitiveView`, union `SitesLike`. `Sites` exposes `reduced_coords` and is iterable,
  indexable, and sized over its rows.
- `Species` (one species; the OPTIMADE `species` object): backends `SpeciesClass` /
  `SpeciesPrimitive`, views `SpeciesClassView` / `SpeciesPrimitiveView`, union `SpeciesLike`.
  The class representation is the frozen `Species`; the primitive representation is an
  OPTIMADE species dict.

```python
from httk.atomistic import Cell, CellParamsView, CellPrimitiveView, SpeciesPrimitiveView

cell = structure.cell            # a Cell
cell.lengths, cell.angles, cell.volume
raw_matrix = tuple(CellPrimitiveView(cell))          # back to a raw 3x3 tuple
params = CellParamsView(cell)                        # (a, b, c, alpha, beta, gamma)
params.a, params.gamma
optimade = dict(SpeciesPrimitiveView(structure.species[0]))  # a species as an OPTIMADE dict

# Construct from parameters (standard orientation convention):
structure_from_params = Structure(cell=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
                                  sites=sites, species=species,
                                  species_at_sites=species_at_sites)
```

The kinds dispatch by type and shape: a `Cell`/`Sites`/`Species` goes to its `*Class`
backend, a raw matrix / dict to its `*Primitive` backend, and a flat 6-sequence to the
`CellParams` backend. Pass `kind="class"`, `kind="primitive"`, or `kind="params"` to force
an interpretation.

## Notes

- A `Structure` is dispatched to `StructureSimple` and a length-3 triple to
  `StructurePrimitive`. A malformed triple raises `TypeError` from `create`.
  Pass `kind="simple"` or `kind="primitive"` to force an interpretation.
- `StructureSimpleView` and `StructurePrimitiveView` are eager: they build their
  full presentation when constructed. The same holds for the component views. The
  `*ClassView` and `*PrimitiveView` immutable-subclass views are genuine instances of their
  class (a `Cell`, a tuple, ...); `SpeciesPrimitiveView` is a genuine — but detached and
  mutable — OPTIMADE `dict`.
- `StructurePrimitiveView` requires every site's species to be a single, unattached
  chemical element; alloy, vacancy, and attached species cannot be represented as a
  bare atomic number and raise `TypeError`. Such species survive in the Simple
  representation.
- Rewrapping a view returns the same object, and views built from the same backend
  share it. `unwrap(view)` returns the native raw object (a `Structure` or a triple, a
  `Cell` or a raw matrix, a `Species` or a dict).
- The numeric values are currently interim nested tuples of floats, and the derived cell
  quantities use plain float arithmetic. They are planned to be replaced by the httk exact
  vector representation; keep numeric access behind the quartet accessors. An ASU
  (asymmetric-unit) representation is also an upcoming addition.

## Shared Behavior and `unwrap`

`unwrap(obj)` returns the most raw representation available:

- for `StructureSimple` / `StructureSimpleView` this is the wrapped `Structure`
- for `StructurePrimitive` / `StructurePrimitiveView` this is the `(lattice, positions, numbers)` triple
- for non-view/backend objects it returns the object unchanged
