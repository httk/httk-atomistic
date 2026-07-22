# Structures

This page documents practical usage of the structure classes in `httk.atomistic`.
It follows the same view/backend pattern as the datastream classes in `httk.core`.

## Overview

A crystal structure is available through one family of backends and views:

- backends: `StructureSimple` (wraps a `Structure`), `StructurePrimitive` (wraps an spglib-like triple)
- views: `StructureSimpleView` (presents any backend as a `Structure`), `StructurePrimitiveView` (presents any backend as a `(lattice, positions, numbers)` tuple)
- accepted union: `StructureLike`

Every backend produces the same canonical Simple quartet declared by `StructureAPI`:
`basis` (3x3 cell vectors), `sites` (Nx3 reduced coordinates), `species` (a tuple of
`Species`), and `species_at_sites` (the species name at each site). Views build their
presentation from that quartet; there is no pairwise conversion between representations.

In normal user code, you usually accept `StructureLike` and normalize immediately to one view.

## Common Calling Patterns

```python
from httk.atomistic import Structure, StructureSimpleView, StructurePrimitiveView

# A Structure (the Simple representation)
structure = Structure(
    basis=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
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

## Notes

- A `Structure` is dispatched to `StructureSimple` and a length-3 triple to
  `StructurePrimitive`. A malformed triple raises `TypeError` from `create`.
  Pass `kind="simple"` or `kind="primitive"` to force an interpretation.
- `StructureSimpleView` and `StructurePrimitiveView` are eager: they build their
  full presentation when constructed.
- `StructurePrimitiveView` requires every site's species to be a single, unattached
  chemical element; alloy, vacancy, and attached species cannot be represented as a
  bare atomic number and raise `TypeError`. Such species survive in the Simple
  representation.
- Rewrapping a view returns the same object, and views built from the same backend
  share it. `unwrap(view)` returns the native raw object (a `Structure` or a triple).
- The numeric values are currently interim nested tuples of floats. They are planned
  to be replaced by the httk exact vector representation; keep numeric access behind
  the quartet accessors. An ASU (asymmetric-unit) representation is also an upcoming
  addition.

## Shared Behavior and `unwrap`

`unwrap(obj)` returns the most raw representation available:

- for `StructureSimple` / `StructureSimpleView` this is the wrapped `Structure`
- for `StructurePrimitive` / `StructurePrimitiveView` this is the `(lattice, positions, numbers)` triple
- for non-view/backend objects it returns the object unchanged
