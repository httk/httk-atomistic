# Building a UnitcellStructure three ways, and reading its geometry back out exactly

A `UnitcellStructure` in *httk-atomistic* is always the same canonical quartet — a
`cell` of three lattice vectors, `sites` of reduced (fractional) coordinates, a
tuple of `species`, and the `species_at_sites` naming which species occupies
each site. What varies is how you *hand it in*. The cell argument dispatches on
type and shape, so all three constructions below produce the same kind of
object and are inspected with exactly the same code:

1. **An explicit 3x3 matrix** — rows are the lattice vectors $\mathbf{a}$,
   $\mathbf{b}$, $\mathbf{c}$. This is the direct route, and the only one that
   pins down the cell's orientation in space.
2. **Cell parameters** — the flat 6-tuple
   $(a, b, c, \alpha, \beta, \gamma)$, angles in degrees, wrapped by
   `CellParams`. A basis is built for you using the standard orientation
   convention: $\mathbf{a}$ along $x$, $\mathbf{b}$ in the $xy$-plane.
3. **An spglib-style triple** — `(lattice, positions, numbers)`, the shape
   symmetry libraries speak. `UnitcellStructureView` presents it as a `UnitcellStructure`,
   inventing one species per distinct atomic number.

The second half of this script is about the one thing that most often surprises
people, and it is worth being precise about: **cell parameters carry no
orientation.** Lengths and angles are invariant under rotation, so a cell and a
rotated copy of it have *identical* parameters. Going cell → parameters → cell
therefore cannot return the matrix you started from; it returns a
*different-but-equivalent* one, in the standard orientation, with the same
lengths, the same angles, and the same volume. If your workflow cares about how
the cell sits relative to a surface, a field, or a supercell it was cut from,
keep the matrix; if it only cares about the crystal, parameters are enough.

Inspection stays in the **exact** numeric model throughout. `cell.lengths` and
`cell.volume` are exact `SurdScalar`s in the squarefree-radical field,
`cell.angles` are exact `Fraction` degrees, `cell.metric()` is the exact Gram
matrix $G_{ij} = \mathbf{a}_i \cdot \mathbf{a}_j$ (rational for any cell
whose parameters are), and `structure.cartesian_sites()` multiplies the rational
reduced coordinates by the surd basis, so a hexagonal cell's $\sqrt3$ survives
into Cartesian space instead of decaying into 0.8660254. Floats are produced
only at the very end, by rendering: `.to_floats()` on any exact vector,
`float(...)` on any exact scalar.

The related guide page {doc}`../structures` covers the same model in reference
form; this script is the runnable counterpart.

```{literalinclude} ../../examples/build_a_structure.py
:language: python
:lines: 44-
```
