# Periodicity

Not everything worth describing is a crystal. A slab is periodic in two directions, a
nanowire in one, a molecule in none. A {py:class}`~httk.atomistic.Cell` records which of its
three directions actually repeat, so httk can represent all of them and reject operations
that only make sense for a crystal.

```python
from httk.atomistic import Cell, Structure

slab = Structure(
    Cell([[3, 0, 0], [0, 3, 0], [0, 0, 1]], periodicity=(True, True, False)),
    sites, species, species_at_sites,
)
slab.periodicity            # (True, True, False)
slab.nperiodic_dimensions   # 2
```

The default is `(True, True, True)`, representing a fully periodic crystal.

## The basis is a frame, not a box

This is the idea the rest follows from. A basis row flagged periodic is a lattice vector, in
the usual sense. A row flagged non-periodic is **not** a lattice vector — it is only a
*coordinate frame*, saying what a fractional coordinate means along that direction.

Three consequences, and they are the whole model:

- **Coordinates there are unbounded.** Freely below `0` or above `1`, and **never wrapped**.
  An atom at `z = 1.05` is a real position, one frame-vector out; it is not another way of
  writing `z = 0.05`.
- **There is no vacuum and no padding.** The frame vector is not a wall and the atoms are not
  inside a box. Nothing is being separated from its periodic image, because there is no
  periodic image.
- **A unit frame vector makes the coordinate a length.** Set the row to a unit vector and the
  fractional coordinate along it simply *is* a displacement in the basis's units.

That last point is what makes a molecule fall out for free. Give it the identity basis and
nothing periodic, and its "fractional" coordinates are literally its ångström coordinates:

```python
water = Structure(
    Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], periodicity=(False, False, False)),
    sites=[[0, 0, 0], [0.7576, 0.5865, 0], [-0.7576, 0.5865, 0]],   # angstrom
    species=[oxygen, hydrogen],
    species_at_sites=["O", "H", "H"],
)
water.cartesian_sites().to_floats() == water.sites.reduced_coords.to_floats()   # True
```

Note the negative coordinate surviving. In a crystal it would have been folded to the far
side of the cell; here there is no far side.

```{admonition} All three basis vectors are still required
:class: note

A cell is always a full, non-degenerate 3×3 basis, whatever its periodicity. A frame vector
must be a real vector — a unit one, not a zero one — or fractional coordinates along it mean
nothing and the basis cannot be inverted. This is also exactly what OPTIMADE mandates: three
lattice vectors regardless of `dimension_types`, with the ones in non-periodic directions
carrying "no significance beyond fulfilling these requirements".
```

## What changes

### Wrapping

Only the periodic directions are wrapped into `[0, 1)`. That matters most in
{py:func}`~httk.atomistic.same_crystal`, where folding a non-periodic direction would call the
top of a slab the same place as the bottom, and in the minimum-separation cap behind
{py:func}`~httk.atomistic.structure_tolerance`, where it would report two atoms 9 Å apart as
close neighbours and derive a tolerance far tighter than the data justifies.

### Identity

Periodicity takes part in `==`, on `Cell` and on `Structure`, and in `same_crystal`. It is not
provenance the way a recorded precision is — it says which rows are lattice vectors at all, so
a slab is never the same cell as the bulk that happens to share its basis.

The comparison stays exact in the frame as written. Two descriptions of the same slab that
differ in their non-periodic frame vector compare *unequal*, because reconciling them would
need an origin convention that `same_crystal` deliberately does not have.

### Volume

{py:attr}`~httk.atomistic.Cell.volume` raises for anything but a full 3D crystal. The
determinant of a reduced-periodicity basis mixes real lattice vectors with frame vectors, so
it changes when a frame vector is rescaled even though nothing about the material did — and
any density or packing fraction derived from it would inherit that.

{py:attr}`~httk.atomistic.Cell.periodic_measure` is the quantity that is always defined: the
size of the repeating unit, whatever its dimension.

| periodicity | `periodic_measure` |
| --- | --- |
| 3D | the cell volume (identical to `volume`) |
| 2D | the area of the periodic plane |
| 1D | the length of the periodic direction |
| 0D | `1` — the empty product, dimensionless |

It is exact in the crystallographic case, through the same machinery as
{py:attr}`~httk.atomistic.Cell.lengths`: the area of a hexagonal slab of `a = 3` comes back as
`(9/2)·√3` in the surd field, not as a float near it.

## What is refused

Space groups describe three-dimensionally periodic lattices, and every part of the symmetry
machinery here is built on that — 3×3 rational operations, coordinates reduced modulo `Z³`,
Wyckoff membership as a congruence in all three directions, and tables holding the 230 space
groups. A slab or a wire has symmetry too, but it is a **layer group** or a **rod group**,
which is different tabulated data and different mathematics. httk does not have either.

So these refuse a reduced-periodicity structure outright, rather than answering with machinery
whose assumptions it violates:

- {py:class}`~httk.atomistic.ASUStructure`
- {py:func}`~httk.atomistic.recognize_asu` and `ASUStructureView`
- {py:func}`~httk.atomistic.asu_structure_from_cif`
- {py:func}`~httk.atomistic.build_supercell`

```pycon
>>> recognize_asu(slab)
ValueError: recognize_asu requires a fully 3D-periodic structure, but this one is periodic
in 2 of 3 directions ((True, True, False)). ...
```

Rejecting these operations prevents silent failures: an orbit generated across a
non-periodic direction could *delete* atoms, and a distance folded across one could make an
unsymmetric structure pass as symmetric.

Everything else keeps working. `lengths`, `angles`, `metric()` and `cartesian_sites()` are
honest geometry of whatever the three vectors are; only their crystallographic reading
changes.

## Marking a structure as a slab

Neither CIF nor POSCAR can tell you a structure is not a crystal. VASP expresses a slab purely
by convention — a lot of empty space along **c** — and nothing in the file says so. The
periodicity has to come from you:

```python
from httk.atomistic import Cell, Structure, load_structure

relaxed = load_structure("relaxed.POSCAR")
slab = Structure(
    Cell(
        relaxed.cell.unscaled_basis,
        relaxed.cell.scale,
        relaxed.cell.precision,
        (True, True, False),
    ),
    relaxed.sites,
    relaxed.species,
    relaxed.species_at_sites,
)
```

Rebuilding the `Cell` through `unscaled_basis`/`scale` rather than `basis` keeps the exact
factoring, and passing `precision` through keeps the stated data precision — see
{doc}`precision`.

Note what this does *not* do: it does not remove the empty space. A relaxed VASP slab keeps
whatever **c** it was computed with, and that is correct — the vector is still a perfectly good
frame, and the atoms sit where they sit. Marking it non-periodic only stops httk from treating
that direction as repeating.

## Serving it

`nperiodic_dimensions` and `dimension_types` are standard OPTIMADE properties, served from the
structure's own periodicity. `dimension_types` is ordered by lattice vector, exactly as
`Cell.periodicity` is — entry *i* refers to basis row *i*, not to the Cartesian *x*, *y*, *z*.

`nperiodic_dimensions` carries query-support `all mandatory`, so it filters:

```
/v1/structures?filter=nperiodic_dimensions<=2
```

which is OPTIMADE's own documented example.

The space-group properties come out `null` for anything that is not a 3D crystal, which is
what OPTIMADE requires. That is not enforced separately — it falls out, because symmetry is
only ever served for an `ASUStructure` and one of those cannot hold a reduced-periodicity
cell.

`site_coordinate_span` stays `unit_cell` while anything is periodic, since translating the
given sites by the lattice vectors does reconstruct the whole system. With nothing periodic
there is no periodic system to reconstruct, and the `molecular_*` values do not fit either —
each additionally requires bonded atoms to be placed at a bond distance, which httk does not
know — so a molecule is served as `other`.

## See also

- {doc}`structures` — the structure classes themselves
- {doc}`asu` — the symmetry machinery this is refused by
- {doc}`precision` — the other thing a `Cell` records about its provenance
