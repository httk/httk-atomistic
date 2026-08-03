"""Building a UnitcellStructure three ways, and reading its geometry back out exactly

A `UnitcellStructure` in *httk-atomistic* is always the same canonical quartet — a
`cell` of three lattice vectors, `sites` of reduced (fractional) coordinates, a
tuple of `species`, and the `species_at_sites` naming which species occupies
each site. What varies is how you *hand it in*. The cell argument dispatches on
type and shape, so all three constructions below produce the same kind of
object and are inspected with exactly the same code:

1. **An explicit 3x3 matrix** — rows are the lattice vectors $\\mathbf{a}$,
   $\\mathbf{b}$, $\\mathbf{c}$. This is the direct route, and the only one that
   pins down the cell's orientation in space.
2. **Cell parameters** — the flat 6-tuple
   $(a, b, c, \\alpha, \\beta, \\gamma)$, angles in degrees, wrapped by
   `CellParams`. A basis is built for you using the standard orientation
   convention: $\\mathbf{a}$ along $x$, $\\mathbf{b}$ in the $xy$-plane.
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
matrix $G_{ij} = \\mathbf{a}_i \\cdot \\mathbf{a}_j$ (rational for any cell
whose parameters are), and `structure.cartesian_sites()` multiplies the rational
reduced coordinates by the surd basis, so a hexagonal cell's $\\sqrt3$ survives
into Cartesian space instead of decaying into 0.8660254. Floats are produced
only at the very end, by rendering: `.to_floats()` on any exact vector,
`float(...)` on any exact scalar.

The related guide page {doc}`../structures` covers the same model in reference
form; this script is the runnable counterpart.
"""

import fractions

from httk.atomistic import (
    Cell,
    CellParams,
    CellParamsView,
    PlainStructureView,
    UnitcellStructure,
    UnitcellStructureView,
)

F = fractions.Fraction

#: Rocksalt NaCl, as OPTIMADE-style species dicts.
NACL_SPECIES = [
    {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
    {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
]
NACL_SITES = [[F(0), F(0), F(0)], [F(1, 2), F(1, 2), F(1, 2)]]


def from_explicit_matrix() -> UnitcellStructure:
    """Route 1: hand in the three lattice vectors as rows of a 3x3 matrix."""
    return UnitcellStructure(
        cell=[[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        sites=NACL_SITES,
        species=NACL_SPECIES,
        species_at_sites=["Na", "Cl"],
    )


def from_cell_parameters() -> UnitcellStructure:
    """Route 2: hand in (a, b, c, alpha, beta, gamma); the basis is built for you."""
    # A flat 6-sequence is dispatched to the CellParams backend automatically...
    return UnitcellStructure(
        cell=(5.64, 5.64, 5.64, 90.0, 90.0, 90.0),
        sites=NACL_SITES,
        species=NACL_SPECIES,
        species_at_sites=["Na", "Cl"],
    )


def from_primitive_triple() -> UnitcellStructure:
    """Route 3: an spglib-style (lattice, positions, numbers) triple, viewed as a UnitcellStructure."""
    triple = (
        [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        [11, 17],  # Na, Cl
    )
    return UnitcellStructureView(triple)


def describe(label: str, structure: UnitcellStructure) -> None:
    """Print the exact geometry of a structure, rendering to floats only at the end."""
    cell = structure.cell
    print(f"{label}:")
    print("  basis            ", cell.basis.to_floats())
    print("  lengths          ", [float(length) for length in cell.lengths])
    print("  angles (degrees) ", [float(angle) for angle in cell.angles])
    print("  volume           ", float(cell.volume))
    print("  species_at_sites ", structure.species_at_sites)
    print("  cartesian sites  ", structure.cartesian_sites().to_floats())


def exact_hexagonal() -> None:
    """A hexagonal cell: the sqrt(3) is carried exactly, all the way into Cartesian space."""
    # CellParams turns (3, 3, 5, 90, 90, 120) into a basis with a genuine sqrt(3) in it,
    # not a float approximation of one. `radicands` lists the radicals actually present.
    cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis)
    structure = UnitcellStructure(
        cell=cell,
        sites=[[F(0), F(0), F(0)], [F(1, 3), F(1, 3), F(0)]],
        species=[{"name": "Mg", "chemical_symbols": ["Mg"], "concentration": [1.0]}],
        species_at_sites=["Mg", "Mg"],
    )
    print("hexagonal a=b=3, c=5, gamma=120:")
    print("  radicands in basis   ", cell.basis.radicands, "(3 present => a real sqrt(3))")
    print("  angles are exact     ", cell.angles, "== (90, 90, 120) exactly:", cell.angles == (F(90), F(90), F(120)))
    print("  volume (exact)       ", cell.volume, "=", float(cell.volume))
    # The metric (Gram) matrix G[i][j] = a_i . a_j is rational here, so squared bond
    # lengths computed through it are rational-exact -- no radicals, no rounding.
    print("  metric()             ", cell.metric().to_floats())
    print("  radicands in metric  ", cell.metric().radicands, "(only 1 => fully rational)")
    cartesian = structure.cartesian_sites()
    print("  cartesian sites      ", cartesian.to_floats())
    print("  radicands in cartesian", cartesian.radicands, "(the sqrt(3) survived)")


def parameters_carry_no_orientation() -> None:
    """The same cubic cell, rotated 90 degrees about z: different basis, identical parameters."""
    upright = Cell([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    rotated = Cell([[0.0, 4.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

    print("orientation:")
    print("  upright basis        ", upright.basis.to_floats())
    print("  rotated basis        ", rotated.basis.to_floats())
    print("  bases are equal?     ", upright.basis == rotated.basis)

    upright_params = CellParamsView(upright)
    rotated_params = CellParamsView(rotated)
    print("  upright parameters   ", tuple(upright_params))
    print("  rotated parameters   ", tuple(rotated_params))
    print("  parameters are equal?", tuple(upright_params) == tuple(rotated_params))
    # The named accessors are the same six numbers under friendlier names.
    print("  rotated.a, .gamma    ", rotated_params.a, rotated_params.gamma)

    # Round-tripping the rotated cell through its parameters cannot recover the rotation:
    # the reconstruction comes back in the standard orientation.
    reconstructed = Cell(CellParams(tuple(rotated_params)).basis)
    print("  reconstructed basis  ", reconstructed.basis.to_floats())
    print("  == rotated basis?    ", reconstructed.basis == rotated.basis, "(lossy: the rotation is gone)")
    print(
        "  ...but equivalent:   ",
        "lengths",
        [float(length) for length in reconstructed.lengths] == [float(length) for length in rotated.lengths],
        "angles",
        reconstructed.angles == rotated.angles,
        "volume",
        float(reconstructed.volume) == float(rotated.volume),
    )


def main() -> None:
    matrix_structure = from_explicit_matrix()
    describe("from an explicit matrix", matrix_structure)
    describe("from cell parameters", from_cell_parameters())
    describe("from an spglib-style triple", from_primitive_triple())

    # Every one of them goes back out as an spglib-style triple, too: the primitive
    # view is the same presentation in reverse (this is what the former
    # examples/structure_to_primitive.py showed).
    lattice, positions, numbers = PlainStructureView(matrix_structure)
    print("back to a primitive triple:")
    print("  lattice  ", lattice)
    print("  positions", positions)
    print("  numbers  ", numbers)

    print()
    exact_hexagonal()
    print()
    parameters_carry_no_orientation()


if __name__ == "__main__":
    main()
