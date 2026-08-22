"""Run the exhaustive P1 scramble invariance benchmark."""

import runpy
import time
from pathlib import Path
from typing import Any


def main() -> int:
    """Run the correctness battery and report its elapsed time."""
    test_support: dict[str, Any] = runpy.run_path(Path(__file__).resolve().parents[1] / "tests" / "test_lift_p1.py")
    canonicalize = test_support["canonicalize"]
    asu_structure = test_support["ASUStructure"]
    cell = test_support["Cell"]
    cscl_sites = test_support["_cscl_sites"]
    expanded_p1 = test_support["_expanded_p1"]
    fraction = test_support["F"]
    frac_vector = test_support["FracVector"]
    invariance_variants = test_support["_invariance_variants"]
    result_key = test_support["_result_key"]
    same_crystal = test_support["same_crystal"]
    scramble_battery = test_support["_SCRAMBLE_BATTERY"]
    scrambled_p1 = test_support["_scrambled_p1"]
    unitcell_view = test_support["UnitcellStructureView"]
    wyckoff_site = test_support["WyckoffSite"]
    pytest = test_support["pytest"]
    p1 = test_support["_p1"]
    species = test_support["_species"]
    supercell_result_key = test_support["_supercell_result_key"]
    surd_vector = test_support["SurdVector"]

    started = time.perf_counter()
    checks = 0
    for name, reference in scramble_battery.items():
        expected = result_key(canonicalize(expanded_p1(reference), tolerance=1e-3))
        for seed in (1, 2, 3):
            assert result_key(canonicalize(scrambled_p1(reference, seed), tolerance=1e-3)) == expected, (name, seed)
            checks += 1

    variants = invariance_variants(((4, 0, 0), (0, 4, 0), (0, 0, 4)), cscl_sites(), ["Cs", "Cl"])
    reference = result_key(canonicalize(variants["base"], tolerance=1e-3))
    assert reference[0] == 221 and reference[1] == (("Cl", "a", ()), ("Cs", "b", ()))
    for name, structure in variants.items():
        assert result_key(canonicalize(structure, tolerance=1e-3)) == reference, name
        checks += 1

    variants = invariance_variants(
        ((5, 0, 0), (0, 5, 0), (0, 0, 5)), [wyckoff_site("a", frac_vector((0, 0, 0)), "Po")], ["Po"]
    )
    reference = result_key(canonicalize(variants["base"], tolerance=1e-3))
    assert reference[0] == 221 and reference[1] == (("Po", "a", ()),)
    for name, structure in variants.items():
        assert result_key(canonicalize(structure, tolerance=1e-3)) == reference, name
        checks += 1
    canonical = canonicalize(variants["base"], tolerance=1e-3)
    assert same_crystal(unitcell_view(canonical.asu), unitcell_view(variants["base"]))
    checks += 2

    po_base = p1(cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [wyckoff_site("a", frac_vector((0, 0, 0)), "Po")])
    po_reference = supercell_result_key(po_base)
    po_112 = p1(
        cell(((5, 0, 0), (0, 5, 0), (0, 0, 10))),
        [
            wyckoff_site("a", frac_vector((0, 0, 0)), "Po"),
            wyckoff_site("a", frac_vector((0, 0, fraction(1, 2))), "Po"),
        ],
    )
    po_113 = p1(
        cell(((5, 0, 0), (0, 5, 0), (0, 0, 15))),
        [
            wyckoff_site("a", frac_vector((0, 0, value)), "Po")
            for value in (fraction(0), fraction(1, 3), fraction(2, 3))
        ],
    )
    shear = frac_vector(((1, 0, 0), (0, 1, 0), (1, 0, 2)))
    inverse = shear.inv()
    po_shear = asu_structure(
        cell(surd_vector(shear) * cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))).basis),
        1,
        [wyckoff_site("a", (frac_vector(point) * inverse).normalize(), "Po") for point in ((0, 0, 0), (0, 0, 1))],
        species("Po"),
    )
    assert supercell_result_key(po_112) == po_reference
    assert supercell_result_key(po_113) == po_reference
    assert supercell_result_key(po_shear) == po_reference
    cscl_base = p1(
        cell(((4, 0, 0), (0, 4, 0), (0, 0, 4))),
        [
            wyckoff_site("a", frac_vector((0, 0, 0)), "Cs"),
            wyckoff_site("a", frac_vector((fraction(1, 2), fraction(1, 2), fraction(1, 2))), "Cl"),
        ],
    )
    cscl_112 = p1(
        cell(((4, 0, 0), (0, 4, 0), (0, 0, 8))),
        [
            wyckoff_site("a", frac_vector((0, 0, 0)), "Cs"),
            wyckoff_site("a", frac_vector((0, 0, fraction(1, 2))), "Cs"),
            wyckoff_site("a", frac_vector((fraction(1, 2), fraction(1, 2), fraction(1, 4))), "Cl"),
            wyckoff_site("a", frac_vector((fraction(1, 2), fraction(1, 2), fraction(3, 4))), "Cl"),
        ],
    )
    assert supercell_result_key(cscl_112) == supercell_result_key(cscl_base)
    checks += 4

    with pytest.MonkeyPatch.context() as monkeypatch:
        recell_searches = test_support["_count_recell_searches"](monkeypatch)
        recell_applications = test_support["_capture_recell_applications"](monkeypatch)
        result = canonicalize(expanded_p1(scramble_battery["NaCl-225-fcc"]), tolerance=1e-3)
    assert result.spacegroup.it_number == 225
    assert recell_searches[0] > 0
    assert recell_applications
    assert all(operation.determinant() == 1 for operation in recell_applications)
    checks += 4

    elapsed = time.perf_counter() - started
    print(f"checks={checks} seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
