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
from httk.atomistic import CellParamsView, CellPrimitiveView, SpeciesPrimitiveView, Structure

cell = structure.cell            # a Cell
cell.lengths                     # a triple of exact SurdScalar norms
cell.angles                      # (alpha, beta, gamma) as exact Fraction degrees
cell.volume                      # an exact SurdScalar
raw_matrix = tuple(CellPrimitiveView(cell))          # back to a raw 3x3 tuple of floats
params = CellParamsView(cell)                        # (a, b, c, alpha, beta, gamma) as floats
params.a, params.gamma
optimade = dict(SpeciesPrimitiveView(structure.species[0]))  # a species as an OPTIMADE dict

# Construct from parameters (standard orientation convention):
structure_from_params = Structure(
    cell=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
    sites=[[0.0, 0.0, 0.0]],
    species=[{"name": "Fe", "chemical_symbols": ["Fe"], "concentration": [1.0]}],
    species_at_sites=["Fe"],
)
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
- The numeric model is **exact**. A `Cell` stores its lattice vectors as a `httk.core.SurdVector`
  (the squarefree-radical field) factored as a positive `SurdScalar` `scale` times an
  `unscaled_matrix`, and its `lengths`/`volume` are exact `SurdScalar`s and `angles` exact
  `Fraction` degrees. A `Sites` stores its reduced coordinates as an exact rational
  `httk.core.FracVector`. Floats appear only at the presentation and JSON boundaries
  (`cell.matrix_floats()`, the `*PrimitiveView`s, and the OPTIMADE records). An ASU
  (asymmetric-unit) representation is an upcoming addition.

## Exact geometry: scale, surd matrices, and Cartesian positions

The numeric layer is exact and split by purpose. The **fractional** frame (reduced coordinates,
symmetry) is rational and lives in `Sites` as a `httk.core.FracVector`; the **Cartesian** frame —
where radicals such as the hexagonal $\sqrt3$ appear — is exact in the squarefree-radical field
(`httk.core.SurdVector`). Magnitudes (bond-length comparisons) stay rational-exact via the metric.

```python
import fractions

from httk.core import FracVector, SurdVector
from httk.atomistic import Cell, CellParams, Structure

F = fractions.Fraction

# Cell parameters -> an EXACT matrix: hexagonal a=b=3, c=5, gamma=120 carries a real sqrt(3).
cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).matrix)
assert 3 in cell.matrix.radicands                       # the sqrt(3) is exact, not a float

# Angles come back exactly through the reverse-Niven table; volume is (45/2)*sqrt(3):
assert cell.angles == (F(90), F(90), F(120))
assert cell.volume == SurdVector.from_radicand_map({3: F(45, 2)})

# The scale carries an overall length factor: unscaled rows scaled by 4 == the absolute matrix,
# and the volume scales as scale**3. Angles are scale-independent.
scaled = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], scale=4)
assert scaled.matrix == Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]).matrix
assert scaled.volume == SurdVector.create(64)

# Exact Cartesian positions: reduced (rational) coordinates times the surd cell matrix.
structure = Structure(
    cell=cell,
    sites=[[F(0), F(0), F(0)], [F(1, 3), F(1, 3), F(0)]],
    species=[{"name": "Mg", "chemical_symbols": ["Mg"], "concentration": [1.0]}],
    species_at_sites=["Mg", "Mg"],
)
cartesian = structure.cartesian_sites()                 # an exact (N, 3) SurdVector
assert 3 in cartesian.radicands                         # the sqrt(3) survives into Cartesian space

# A bond squared-length is rational-exact; the exact-Cartesian and rational-metric routes agree.
diff = FracVector.create([F(1, 3), F(1, 3), F(0)])
bond_sqr_cartesian = (SurdVector.create(diff) * cell.matrix).lengthsqr()
bond_sqr_metric = (SurdVector.create(diff) * cell.metric()).dot(SurdVector.create(diff))
assert bond_sqr_cartesian == bond_sqr_metric
assert bond_sqr_cartesian.is_rational

# Floats appear only at the presentation boundary:
cell.matrix_floats()
structure.cartesian_sites_floats()
```

## Shared Behavior and `unwrap`

`unwrap(obj)` returns the most raw representation available:

- for `StructureSimple` / `StructureSimpleView` this is the wrapped `Structure`
- for `StructurePrimitive` / `StructurePrimitiveView` this is the `(lattice, positions, numbers)` triple
- for non-view/backend objects it returns the object unchanged
