# Lattice reduction (Niggli)

Niggli reduction puts a fully three-dimensional periodic cell into a canonical primitive
lattice setting. The operation changes the basis vectors, but not the lattice they span. It is
useful when cells need to be compared by lattice shape or passed to algorithms that expect a
reduced cell.

## Exactness

{py:func}`~httk.atomistic.niggli_reduce` reads the exact rational Gram matrix returned by
{py:meth}`~httk.atomistic.Cell.metric`. The Krivý–Gruber steps use `epsilon = 0`: every
comparison, including boundary equalities and special conditions, is an exact comparison of
{py:class}`fractions.Fraction` values. Recognition is a separate, tolerant operation; Niggli
reduction is not.

The input must have `periodicity == (True, True, True)`. A slab, wire, or molecular frame is not
a lattice in all three directions and is refused.

## The transform convention

Cell vectors are rows. If `C = result.transform`, then

```python
basis_reduced = C * basis_original
```

`C` is an integer matrix with determinant `+1`. For row-vector fractional coordinates, Cartesian
positions are preserved by

```python
coordinates_reduced = coordinates_original * C.inv()
```

The structure-level {py:func}`~httk.atomistic.niggli_reduced` operation applies that remapping
exactly and wraps the coordinates into `[0, 1)`.

## Cell and structure use

Reduce a cell directly:

```pycon
>>> from httk.atomistic import Cell, niggli_reduce, is_niggli_reduced
>>> result = niggli_reduce(Cell([[1, 0, 0], [0, 1, 0], [1, 0, 1]]))
>>> result.transform.det().to_fraction()
Fraction(1, 1)
>>> is_niggli_reduced(result.cell)
True
```

For a structure, use `niggli_reduced(structure)`. Site order and count are unchanged, species,
charge, composition, assemblies, and Cartesian site moments are carried through, and recorded
cell and coordinate precision is widened by the exact induced matrix norms.

To compare the original and reduced structures with the package's same-basis
{py:func}`~httk.atomistic.same_crystal` predicate, transform the reduced structure back first:

```python
from httk.atomistic import UnitcellStructureView, build_supercell, niggli_reduced, same_crystal

reduced = niggli_reduced(original)
inverse = reduced.transform.inv()
restored = build_supercell(reduced.structure, inverse)
same_crystal(UnitcellStructureView(original), UnitcellStructureView(restored.structure))
```

The algorithm follows [I. Krivý and B. Gruber](https://doi.org/10.1107/S0567739476000636),
“A unified algorithm for determining the reduced (Niggli) cell”, *Acta Crystallographica
Section A* **32** (1976), 297–298, and the exact change-of-basis and stabilization formulation
described by [R. W. Grosse-Kunstleve, N. K. Sauter, and P. D. Adams](https://doi.org/10.1107/S010876730302186X),
“Numerically stable algorithms for the computation of reduced unit cells”, *Acta Crystallographica
Section A* **60** (2004), 1–6.
