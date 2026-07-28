"""Data precision: letting the file choose the tolerance

Every tolerance in a structure pipeline is a guess about how good the data is.
Match two coordinates within `1e-3`? Hand spglib a `symprec` of `1e-5`? Those
numbers usually come from habit rather than from the data, and when they are
wrong they are wrong silently.

But a file already says how good it is. A coordinate written `0.3333` claims a
precision of about `1e-4` of a cell edge; written `0.33` it claims only `1e-2`.
A cell edge written `5.6402(3)` states its uncertainty outright. httk records
those claims and derives its tolerances from them.

This example shows where the numbers come from, how they are converted, and one
case where deriving the tolerance changes the answer from wrong to right.

## The two values

A structure records two precisions, each on the object that owns the numbers it
describes:

- **coordinate precision** on `Sites`, in *fractional* units — reduced
  coordinates are dimensionless, and a `Sites` has no cell to convert with;
- **basis precision** on `Cell`, in the cell's *length* units.

Both are exact rationals, so nothing picks up binary noise on the way to
becoming a tolerance. `None` means unknown, which is a real answer and not the
same as claiming exactness.

`Structure.cartesian_precision()` combines them into the number a tolerance
actually wants: a distance. That conversion is not cosmetic — the same `1e-4`
coordinates mean `5e-4` in a 5 A cell and `3e-3` in a 30 A one.

## The rules

Three, and each exists because the naive alternative is wrong:

- **The coarsest value wins.** A structure is only as precisely stated as its
  least precisely stated number.
- **A stated uncertainty widens the claim.** `5.6402(3)` is good to `3e-4`, not
  to the `1e-4` its four decimals alone suggest.
- **An exact rational makes no claim at all.** `1/3` states a value rather than a
  measurement, and must not drag a table down to the width of its last digit.

Run this file to see every step printed.
"""

import fractions
import tempfile
from pathlib import Path

from httk.core import combined_precision, decimal_precision, load

from httk.atomistic import (
    Cell,
    Sites,
    Spacegroup,
    Species,
    Structure,
    UnitcellStructureView,
    structure_tolerance,
)
from httk.atomistic.cif_structures import asu_structure_from_cif

HTTK_EXAMPLE_REQUIRES = ["httk.io"]

F = fractions.Fraction


def angstrom(precision: fractions.Fraction | None) -> str:
    """Render a precision as a distance, saying so when there is none to render."""
    return "unknown" if precision is None else f"{float(precision):.1e} A"


def show_what_a_literal_claims() -> None:
    """The primitive underneath everything else, in httk-core."""
    print("== What a written number claims ==")
    for literal in ("0.3333", "0.33", "5.6402", "10", "1.2e-3", "-0.5", "1/3"):
        precision = decimal_precision(literal)
        note = "exact, not measured" if precision is None else f"= {float(precision):g}"
        print(f"   {literal:>8}  ->  {str(precision):>10}   {note}")
    print()
    print("Across a set, the coarsest wins — one sloppy value really does mean")
    print("the whole table is only that good:")
    print("   ['0.123456', '0.5'] ->", combined_precision(["0.123456", "0.5"]))
    print("   ['0.1234', '1/3']   ->", combined_precision(["0.1234", "1/3"]), "(the fraction abstains)")
    print()


def show_the_conversion_to_a_distance() -> None:
    """A fractional precision is not a tolerance until it becomes a length."""
    print("== Fractional precision is not yet a distance ==")
    sodium = [Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))]
    for edge in (5, 30):
        cell = [[edge, 0, 0], [0, edge, 0], [0, 0, edge]]
        structure = Structure(Cell(cell), Sites([[0, 0, 0]], F(1, 10000)), sodium, ["Na"])
        print(f"   coordinates good to 1e-4 in a {edge:>2} A cell  ->  {angstrom(structure.cartesian_precision())}")
    print("Same coordinates, same claim, six times the physical uncertainty.")
    print()

    print("The cell's own precision is a floor — a cell known to 1e-3 cannot")
    print("place an atom better than that, however many decimals the sites have:")
    coarse_cell = Structure(
        Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]], 1, F(1, 1000)),
        Sites([[0, 0, 0]], F(1, 10000)),
        sodium,
        ["Na"],
    )
    print(f"   -> {angstrom(coarse_cell.cartesian_precision())}")
    print()


def write_cif(directory: Path, coordinates: str) -> Path:
    """A monoclinic CIF whose one site is meant to sit on Wyckoff 4e (0, y, 1/4)."""
    spacegroup = Spacegroup.for_setting("15:b1")
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = directory / "measured.cif"
    path.write_text(
        "data_measured\n"
        "_cell_length_a 5.000\n_cell_length_b 6.000\n_cell_length_c 7.000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 15\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        f"Si1 Si {coordinates}\n",
        encoding="utf-8",
    )
    return path


def show_the_payoff(directory: Path) -> None:
    """Where a derived tolerance turns a wrong answer into a right one."""
    print("== Why it matters ==")
    # Rounded to three decimals: 0.005 A off in x, 0.007 A off in z.
    path = write_cif(directory, "0.001 0.333 0.251")
    block = load(str(path))["blocks"][0]

    print(f"the file states its coordinates to {block['coordinate_precision']}")
    print("the site is meant to be on Wyckoff 4e of SG 15 (0, y, 1/4), but rounding")
    print("has moved it 0.005 A off in x and 0.007 A off in z.")
    print()

    for label, tolerance in (("a fixed 1e-3 tolerance", 1e-3), ("the derived tolerance", None)):
        asu = asu_structure_from_cif(block, tolerance=tolerance)
        position = asu.spacegroup.wyckoff_position(asu.asu_sites[0].wyckoff)
        atoms = len(UnitcellStructureView(asu).sites)
        print(f"   {label:<24} -> Wyckoff {position.multiplicity}{position.letter}, {atoms} atoms")

    print()
    print("The fixed tolerance is too tight for data this coarsely written, so the")
    print("site misses its special position, falls back to the general position, and")
    print("generates twice the atoms the structure has — silently. The file said how")
    print("good it was; nothing was listening.")
    print()


def show_the_tolerance_is_bounded() -> None:
    """A coarse claim must still not be allowed to merge distinct atoms."""
    print("== The tolerance is capped, not merely derived ==")
    species = [
        Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,)),
        Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,)),
    ]
    cell = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]

    far = Structure(Cell(cell), Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], F(1, 10)), species, ["Na", "Cl"])
    close = Structure(Cell(cell), Sites([[0, 0, 0], [F(1, 10), 0, 0]], F(1, 10)), species, ["Na", "Cl"])

    print(f"   coarse data, atoms 4.33 A apart -> {structure_tolerance(far):.2f} A  (as derived)")
    print(f"   coarse data, atoms 0.50 A apart -> {structure_tolerance(close):.2f} A  (capped)")
    print("Half the closest approach is the ceiling: beyond it a tolerance would")
    print("merge atoms that are genuinely distinct.")
    print()

    unknown = Structure(Cell(cell), Sites([[0, 0, 0]]), species[:1], ["Na"])
    print(f"   a structure that states no precision  -> {angstrom(unknown.cartesian_precision())}")
    print(f"   ...so its tolerance is                -> {structure_tolerance(unknown)} (the documented fallback)")
    print()


def main() -> None:
    show_what_a_literal_claims()
    show_the_conversion_to_a_distance()
    with tempfile.TemporaryDirectory() as tmp:
        show_the_payoff(Path(tmp))
    show_the_tolerance_is_bounded()


if __name__ == "__main__":
    main()
