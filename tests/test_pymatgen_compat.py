import fractions
import subprocess
import sys
from typing import Any, ClassVar

import pytest

from httk.atomistic import PymatgenStructure, PymatgenStructureProtocol, StructureBackend, UnitcellStructureView
from httk.atomistic.integrations.ase import ASEAtomsProtocol


class Lattice:
    matrix: ClassVar[list[list[float]]] = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    pbc: ClassVar[tuple[bool, bool, bool]] = (True, True, False)


class Element:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def __str__(self) -> str:
        return self.symbol


class Species:
    def __init__(self, symbol: str, oxi_state: int, spin: int | None = None) -> None:
        self.symbol = symbol
        self.oxi_state = oxi_state
        self.spin = spin

    def __str__(self) -> str:
        return f"{self.symbol}{self.oxi_state:+}"


class DummySpecies:
    def __init__(self, symbol: str, oxi_state: int = 0) -> None:
        self.symbol = symbol
        self.oxi_state = oxi_state

    def __str__(self) -> str:
        return self.symbol


class FakePymatgenStructure:
    lattice: ClassVar[Lattice] = Lattice()
    frac_coords: ClassVar[list[list[int]]] = [[0, 0, 0]]
    species_and_occu: ClassVar[list[dict[Element, int]]] = [{Element("Na"): 1}]


def test_pymatgen_protocol_is_duck_typed_and_disjoint_from_ase() -> None:
    from test_compat import FakeAtoms

    assert isinstance(FakePymatgenStructure(), PymatgenStructureProtocol)
    assert not isinstance(FakePymatgenStructure(), ASEAtomsProtocol)
    assert isinstance(FakeAtoms(), ASEAtomsProtocol)
    assert not isinstance(FakeAtoms(), PymatgenStructureProtocol)


def test_fake_pymatgen_backend_and_declines() -> None:
    fake = FakePymatgenStructure()
    backend = StructureBackend.create(fake)
    assert isinstance(backend, PymatgenStructure)
    assert backend.unwrap() is fake
    assert backend.cell.periodicity == (True, True, False)
    assert backend.species_at_sites == ("Na",)
    assert PymatgenStructure(UnitcellStructureView(fake)) is None


def test_pymatgen_absence_keeps_backend_importable() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['pymatgen'] = None
import httk.atomistic
import httk.atomistic.integrations.pymatgen.models as models
try:
    models.PymatgenStructureView
except ImportError as exc:
    assert 'pymatgen' in str(exc)
else:
    raise AssertionError('optional view unexpectedly imported')
""",
        ],
        check=True,
    )


def test_pymatgen_round_trip_and_exact_charge() -> None:
    pmg = pytest.importorskip("pymatgen")
    from pymatgen.core import IStructure, Structure

    from httk.atomistic import PymatgenStructureView

    original = Structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    backend = PymatgenStructure(original)
    assert backend.charge is None
    assert UnitcellStructureView(backend).species_at_sites == ("Na", "Cl")
    view = PymatgenStructureView(backend)
    assert view.lattice.pbc == (True, True, True)
    assert type(view.unview()) is pmg.core.Structure
    assert view.unwrap() is original

    immutable = IStructure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["Na"], [[0, 0, 0]])
    assert isinstance(immutable, PymatgenStructureProtocol)
    assert isinstance(PymatgenStructure(immutable), PymatgenStructure)

    charged = Structure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        ["Na"],
        [[0, 0, 0]],
        charge=2,
    )
    assert PymatgenStructure(charged).charge == fractions.Fraction(2)


def test_pymatgen_disorder_spin_dummy_and_moments() -> None:
    pytest.importorskip("pymatgen")
    from pymatgen.core import DummySpecies, Species, Structure
    from pymatgen.electronic_structure.core import Magmom

    from httk.atomistic import CartesianSiteMoments, CollinearSiteMoments, PymatgenStructureView, same_crystal

    disorder = Structure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [{Species("Fe", oxidation_state=2): 0.3, Species("Fe", oxidation_state=3): 0.4}],
        [[0, 0, 0]],
    )
    backend = PymatgenStructure(disorder)
    species = backend.species[0]
    assert species.concentration == (fractions.Fraction(2, 5), fractions.Fraction(3, 10), fractions.Fraction(3, 10))
    assert species.chemical_symbols[-1] == "vacancy"
    assert set(species.charges or ()) == {fractions.Fraction(2), fractions.Fraction(3), None}
    exported = PymatgenStructureView(backend)
    assert "vacancy" not in str(exported.species_and_occu[0])
    assert sum(float(value) for value in exported.species_and_occu[0].values()) == pytest.approx(0.7)
    original_httk = UnitcellStructureView(backend)
    httk_roundtrip = UnitcellStructureView(PymatgenStructure(exported.unview()))
    assert same_crystal(httk_roundtrip, original_httk)

    spin = PymatgenStructure(
        Structure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            [Species("Fe", oxidation_state=2, spin=2)],
            [[0, 0, 0]],
        )
    )
    assert spin.species[0].spins == (fractions.Fraction(2),)

    dummies = PymatgenStructure(
        Structure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            [DummySpecies("X"), DummySpecies("Xfoo"), DummySpecies("Q")],
            [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]],
        )
    )
    assert tuple(value.labels for value in dummies.species) == (None, ("foo",), ("Q",))
    assert dummies.species[0].charges is None
    with pytest.raises(ValueError, match="label 'Q'"):
        PymatgenStructureView(dummies)

    valid_dummies = PymatgenStructure(
        Structure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            [DummySpecies("X"), DummySpecies("Xfoo")],
            [[0, 0, 0], [0.2, 0.2, 0.2]],
        )
    )
    dummy_export = PymatgenStructureView(valid_dummies)
    assert str(next(iter(dummy_export.sites[0].species))) == "X"
    assert next(iter(dummy_export.sites[1].species)).symbol == "Xfoo"

    scalar = PymatgenStructure(
        Structure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            ["Fe", "Fe"],
            [[0, 0, 0], [0.2, 0.2, 0.2]],
            site_properties={"magmom": [1, -2]},
        )
    )
    assert isinstance(scalar.site_moments, CollinearSiteMoments)
    assert PymatgenStructureView(scalar).site_properties["magmom"] == [1.0, -2.0]

    vector = PymatgenStructure(
        Structure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            ["Fe", "Fe"],
            [[0, 0, 0], [0.2, 0.2, 0.2]],
            site_properties={"magmom": [[1, 2, 3], Magmom(2)]},
        )
    )
    assert isinstance(vector.site_moments, CartesianSiteMoments)
    assert PymatgenStructureView(vector).site_properties["magmom"] == [[1.0, 2.0, 3.0], [0.0, 0.0, 2.0]]


def test_pymatgen_partial_periodicity_and_discarded_metadata(caplog: Any) -> None:
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    from httk.atomistic import PymatgenStructureView

    original = Structure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        site_properties={"magmom": [None, None], "foreign": ["discard", "discard"]},
        labels=["site-a", "site-b"],
        properties={"source": "discard"},
    )
    original.lattice.pbc = (True, True, False)
    caplog.clear()
    backend = PymatgenStructure(original)
    assert backend.cell.periodicity == (True, True, False)
    assert backend.unwrap() is original
    assert backend.site_moments is not None
    warnings = [record for record in caplog.records if getattr(record, "context", None) == "pymatgen"]
    assert len(warnings) == 1
    assert "foreign" not in PymatgenStructureView(backend).site_properties


def test_pymatgen_exact_fraction_round_trip_without_vacancy() -> None:
    pytest.importorskip("pymatgen")
    from httk.atomistic import PymatgenStructureView, Species, UnitcellStructure

    fraction = fractions.Fraction
    species = Species("Fe2+Fe3+", ("Fe", "Fe"), (fraction(1, 3), fraction(2, 3)), charges=(2, 3))
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0]],
        (species,),
        (species.name,),
        charge=fraction(2, 3),
    )
    exported = PymatgenStructureView(structure).unview()
    assert tuple(exported.species_and_occu[0].values()) == (fraction(1, 3), fraction(2, 3))
    assert exported._charge == fraction(2, 3)
    backend = PymatgenStructure(exported)
    assert backend.species[0].concentration == (fraction(1, 3), fraction(2, 3))
    assert "vacancy" not in backend.species[0].chemical_symbols


def test_pymatgen_view_rejects_masses_and_empty_labels() -> None:
    pytest.importorskip("pymatgen")
    from httk.atomistic import PymatgenStructureView, Species, UnitcellStructure

    cell = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    single = Species("C13", ("C",), (1,), mass=(13.0,))
    single_structure = UnitcellStructure(cell, [[0, 0, 0]], (single,), (single.name,))
    with pytest.raises(ValueError, match="explicit constituent masses"):
        PymatgenStructureView(single_structure)

    isotopes = Species("C12C13", ("C", "C"), (fractions.Fraction(1, 2),) * 2, mass=(12.0, 13.0))
    isotope_structure = UnitcellStructure(cell, [[0, 0, 0]], (isotopes,), (isotopes.name,))
    with pytest.raises(ValueError, match="explicit constituent masses"):
        PymatgenStructureView(isotope_structure)

    empty = Species("X-empty", ("X",), (1,), labels=("",))
    empty_structure = UnitcellStructure(cell, [[0, 0, 0]], (empty,), (empty.name,))
    with pytest.raises(ValueError, match="empty label"):
        PymatgenStructureView(empty_structure)


def test_pymatgen_name_collision_gets_stable_suffix() -> None:
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    source = Structure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [
            {"Na": fractions.Fraction(1, 2), "Cl": fractions.Fraction(1, 2)},
            {"Na": fractions.Fraction(1, 4), "Cl": fractions.Fraction(3, 4)},
        ],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    backend = PymatgenStructure(source)
    assert backend.species_at_sites == ("ClNa", "ClNa_2")


def test_pymatgen_ndarray_magnetic_vectors() -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    from httk.atomistic import PymatgenStructureView

    backend = PymatgenStructure(
        Structure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            ["Fe", "Fe"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            site_properties={"magmom": numpy.array([[1, 2, 3], [4, 5, 6]])},
        )
    )
    assert PymatgenStructureView(backend).site_properties["magmom"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_pymatgen_view_rejects_unrepresentable_semantics() -> None:
    pytest.importorskip("pymatgen")
    from httk.atomistic import Assembly, PymatgenStructureView, Species, UnitcellStructure
    from httk.atomistic.composition import ChemicalComposition

    species = Species("C", ("C",), (1,), labels=("bad",))
    structure = UnitcellStructure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], (species,), ("C",))
    with pytest.raises(ValueError, match="labels on elements"):
        PymatgenStructureView(structure)
    plain_species = Species("C", ("C",), (1,))
    assembled = UnitcellStructure(
        structure.cell,
        structure.sites,
        (plain_species,),
        (plain_species.name,),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(TypeError, match="assemblies"):
        PymatgenStructureView(assembled)

    attached = Species("attached", ("C",), (1,), attached=("H",), nattached=(1,))
    attached_structure = UnitcellStructure(structure.cell, structure.sites, (attached,), (attached.name,))
    with pytest.raises(TypeError, match="attached species"):
        PymatgenStructureView(attached_structure)

    declared = UnitcellStructure(
        structure.cell,
        structure.sites,
        (plain_species,),
        (plain_species.name,),
        chemical_composition=ChemicalComposition({"C": 1}, mode="implicit"),
    )
    with pytest.raises(TypeError, match="declared chemical composition"):
        PymatgenStructureView(declared)
