"""Asymmetric units: reading a CIF, expanding it, and going back again

A crystal is mostly repetition. A structure with 192 atoms in its cell may have
only two that are genuinely independent — the rest are copies placed by
symmetry. The *asymmetric unit* is that irreducible part, and
`httk.atomistic.ASUStructure` is httk's representation of it: a space group, one
site per orbit, and the values of each site's free parameters.

This example reads a CIF into an ASU, expands it to the full cell, recovers the
ASU from the expansion, and checks the crystal survived the trip.

## Why not just store the atoms?

Three reasons, and the third is the interesting one.

- It is smaller, and it says what the structure *means*: "sodium on Wyckoff `a`"
  rather than four coordinates that happen to be related.
- Symmetry is preserved rather than inferred. Expanding and re-detecting a
  structure is lossy in a way that storing the ASU is not.
- **It is exact.** Reduced coordinates, symmetry operations, and Wyckoff
  parameters are all rationals, and httk keeps them that way — as
  `Fraction`s, never as floats. Expanding an ASU is affine arithmetic over the
  rationals followed by an exact equality test, so it needs no coordinate grid,
  no snapping, and no tolerance at all.

## Any setting, including ones in no table

A space group can be written in many *settings* — different axis choices,
different origins. They are genuinely different coordinate systems: Wyckoff
letter `e` of space group 15 is `0,y,1/4` in the standard setting `15:b1` and
`1/4,0,z` in `15:c1`.

httk always records the Wyckoff data against the International Tables
**standard** setting, and carries a `SettingTransform` saying how to get from
there to the setting the structure is actually in. Because that transform is
*stored* rather than looked up, a setting that appears in no table is
representable just as well as a tabulated one — which is the point.

The transform is never *derived*. Searching for one that maps a group onto
itself is massively underdetermined: for space group 15 alone there are 192
valid candidates among just the small integer matrices, and they describe
*different crystals*. So httk stores the transform it was given and refuses to
guess one it was not.

## Where tolerance lives

In exactly one place: recognizing a structure that does not already carry its
symmetry.

A file writes `0.3333`. httk embeds that as the rational 3333/10000 — what the
file says — not as `float("0.3333")`, whose exact value is
6004199023210345/18014398509481984 and claims a precision nobody stated. The
recognition step then decides which Wyckoff position each site is *meant* to be
on, and the position supplies exact values for its fixed components. After that
everything is exact again.

So the two directions are not symmetric, and the example prints both:

- **expansion is lossless**, and
- **recognition is lossy at the tolerance level** — it snaps a measured structure onto an
  idealised one.

Run this file to see every step printed.
"""

import tempfile
from pathlib import Path

from httk.core import FracVector, load

from httk.atomistic import (
    ASUStructure,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    recognize_asu,
    same_crystal,
)

NACL_CIF = """\
data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 225
_space_group_name_H-M_alt 'F m -3 m'
loop_
_space_group_symop_operation_xyz
{symops}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0.0 0.0 0.0 1.0
Cl1 Cl 0.5 0.5 0.5 1.0
"""


def write_nacl_cif(directory: Path) -> Path:
    """A rock-salt CIF, with the full symmetry-operation list Fm-3m requires."""
    operations = Spacegroup.standard(225).symmetry_operations
    symops = "\n".join(f"'{operation.wrapped().to_xyz()}'" for operation in operations)
    path = directory / "nacl.cif"
    path.write_text(NACL_CIF.format(symops=symops), encoding="utf-8")
    return path


def read_the_asymmetric_unit(path: Path) -> ASUStructure:
    """A CIF states its own symmetry, so no symmetry search is needed — and no spglib."""
    print("== Reading a CIF into an asymmetric unit ==")
    asu = load(str(path))

    print(f"space group:       {asu.spacegroup.it_number} ({asu.spacegroup.hermann_mauguin})")
    print(f"standard setting:  {asu.spacegroup.setting}")
    print(f"written in:        {'the standard setting' if asu.is_standard_setting else asu.setting().setting}")
    print(f"asymmetric unit:   {len(asu.wyckoff_sites)} site(s) for {sum(asu.multiplicities())} atoms")
    for site, multiplicity in zip(asu.wyckoff_sites, asu.multiplicities()):
        position = asu.spacegroup.wyckoff_position(site.wyckoff)
        print(
            f"   {site.species:<4} on Wyckoff {position.multiplicity}{position.letter}"
            f" (site symmetry {position.site_symmetry}, {position.free_count} free parameter(s))"
            f" -> {multiplicity} atoms"
        )
    print()
    return asu


def expand_to_the_full_cell(asu: ASUStructure) -> None:
    """`UnitcellStructureView` turns an ASU into an ordinary UnitcellStructure, exactly."""
    print("== Expanding to the full unit cell ==")
    structure = UnitcellStructureView(asu)

    print(f"sites: {len(structure.sites)}")
    for name, coordinate in zip(structure.species_at_sites, structure.sites.reduced_coords.to_fractions()):
        rendered = ", ".join(str(value) for value in coordinate)
        print(f"   {name:<4} ({rendered})")
    print("Every coordinate above is an exact Fraction; no float ever took part.")
    print()


def go_back_to_the_asymmetric_unit(asu: ASUStructure) -> None:
    """The round trip, and the guarantee that comes with it."""
    print("== Round trip: ASU -> cell -> ASU -> cell ==")
    expanded = UnitcellStructureView(asu)
    recovered = recognize_asu(expanded, setting=asu.setting())
    rebuilt = UnitcellStructureView(recovered)

    print("recovered:", ", ".join(f"{site.species} on {site.wyckoff}" for site in recovered.wyckoff_sites))
    print("same crystal:               ", same_crystal(expanded, rebuilt))
    print("coordinates bitwise identical:", expanded.sites.reduced_coords == rebuilt.sites.reduced_coords)
    print("The input was already exact, so nothing moved at all.")
    print()


def show_a_non_standard_setting() -> None:
    """The same crystal, described in two settings, with the transform between them."""
    print("== A non-standard setting ==")
    standard = Spacegroup.standard(15)
    other = Spacegroup.from_setting("15:c1")

    print(f"Wyckoff e of {standard.setting}: {standard.wyckoff_position('e').representative.operation.to_xyz()}")
    print(f"Wyckoff e of {other.setting}: {other.wyckoff_position('e').representative.operation.to_xyz()}")
    print("Same position, different coordinates — the settings are different frames.")

    silicon = [Species(name="Si", chemical_symbols=("Si",), concentration=(1.0,))]
    site = [WyckoffSite("e", FracVector(["1/3"]), "Si")]
    cell = [[5, 0, 0], [0, 6, 0], [0, 0, 7]]

    in_standard = ASUStructure(cell, 15, site, silicon)
    in_other = ASUStructure(cell, 15, site, silicon, transform=other.transform_from_standard)

    for label, structure in (("standard", in_standard), ("15:c1", in_other)):
        coordinates = UnitcellStructureView(structure).sites.reduced_coords.to_fractions()
        rendered = "  ".join("(" + ", ".join(str(v) for v in row) + ")" for row in coordinates)
        print(f"   {label:<9} {rendered}")

    transform = other.transform_from_standard
    print(f"transform standard -> 15:c1: {transform.operation.to_xyz()}")
    print("The Wyckoff letter and free parameter are stored once, in the standard")
    print("setting; the transform is what places them in the frame you asked for.")
    print()


def show_that_recognition_is_the_lossy_direction() -> None:
    """Measured coordinates carry rounding, and recognition idealises it away."""
    print("== The one place tolerance lives ==")
    silicon = [Species(name="Si", chemical_symbols=("Si",), concentration=(1.0,))]
    exact = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si")],
        silicon,
    )
    expanded = UnitcellStructureView(exact)

    # Nudge every coordinate, as a refinement would.
    nudged = [
        [value + FracVector(f"{index + 1}/100000").to_fraction() for value in row]
        for index, row in enumerate(expanded.sites.reduced_coords.to_fractions())
    ]
    measured = UnitcellStructure(expanded.cell, nudged, expanded.species, expanded.species_at_sites)

    recovered = recognize_asu(measured, setting=Spacegroup.standard(15))
    idealised = UnitcellStructureView(recovered)

    print(f"measured x of first site:  {measured.sites.reduced_coords.to_fractions()[0][0]}")
    print(f"idealised x of first site: {idealised.sites.reduced_coords.to_fractions()[0][0]}")
    print("The position fixes x at 0 exactly; only the free parameter keeps what")
    print("was measured. That is why recognition is lossy and expansion is not:")
    print(f"   same crystal as the measured input? {same_crystal(measured, idealised)}")
    print(
        f"   recognizing the idealised one again changes nothing? "
        f"{UnitcellStructureView(recognize_asu(idealised, setting=Spacegroup.standard(15))).sites.reduced_coords == idealised.sites.reduced_coords}"
    )
    print()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_nacl_cif(Path(tmp))
        asu = read_the_asymmetric_unit(path)
        expand_to_the_full_cell(asu)
        go_back_to_the_asymmetric_unit(asu)
    show_a_non_standard_setting()
    show_that_recognition_is_the_lossy_direction()


if __name__ == "__main__":
    main()
