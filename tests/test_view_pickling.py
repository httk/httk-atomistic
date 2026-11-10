"""Every atomistic view pickles by reconstructing from its backend (parallel bulk ingest needs it)."""

import copy
import importlib
import pickle
import pkgutil
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import View

import httk.atomistic
from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    BareProtostructureLabel,
    BareProtostructureView,
    BarePrototypeLabel,
    BarePrototypeView,
    CartesianSiteMomentsView,
    CellParamsView,
    CellView,
    ChemicalFormulaView,
    CompositionView,
    CrystalAxisSiteMomentsView,
    FormulatypeView,
    FundamentalDomainTemplateView,
    NumericUnitcellStructureView,
    PlainSpeciesView,
    PlainStructureView,
    ProtonetstructureLabel,
    ProtonetstructureView,
    ProtonettypeLabel,
    ProtonettypeView,
    Protostructure,
    ProtostructureLabel,
    ProtostructureView,
    Prototype,
    PrototypeLabel,
    PrototypeView,
    SitesView,
    SpeciesView,
    StructuretypeView,
    TrajectoryView,
    UnitcellStructureView,
)
from httk.atomistic.models.cell.numeric_view import CellNumericView
from httk.atomistic.models.cell.plain_view import PlainCellView
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.sites.numeric_view import SitesNumericView
from httk.atomistic.models.sites.plain_view import PlainSitesView
from httk.atomistic.models.trajectory import PlainTrajectory

COD = Path(__file__).parent / "fixtures" / "cod" / "1000008.cif"
MOMENTS = [[0, 0, 1], [0, 0, -1]]


@cache
def _asu() -> ASUStructure:
    return httk.core.load(str(COD))


def _protostructure() -> Protostructure:
    return Protostructure(representative=_asu())


def _prototype() -> Prototype:
    return Prototype(225, [("a", "A"), ("b", "B")], discriminator="001")


def _trajectory() -> PlainTrajectory:
    return PlainTrajectory(
        {
            "nframes": 2,
            "lattice_vectors": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            "fractional_site_positions": [[[0, 0, 0]], [[1, 0, 0]]],
            "species_at_sites": [["Si"]],
            "species": [[{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1]}]],
        }
    )


BUILDERS: dict[type[View[Any]], Callable[[], Any]] = {
    UnitcellStructureView: lambda: UnitcellStructureView(_asu(), immutable_id="mat-1"),
    ASUStructureView: lambda: ASUStructureView(_asu()),
    PlainStructureView: lambda: PlainStructureView(_asu()),
    NumericUnitcellStructureView: lambda: NumericUnitcellStructureView(_asu()),
    SpeciesView: lambda: SpeciesView("Si"),
    PlainSpeciesView: lambda: PlainSpeciesView(SpeciesView("Si")),
    SitesView: lambda: SitesView(_asu().sites),
    PlainSitesView: lambda: PlainSitesView(_asu().sites),
    SitesNumericView: lambda: SitesNumericView(_asu().sites),
    CellView: lambda: CellView(_asu().cell),
    PlainCellView: lambda: PlainCellView(_asu().cell),
    CellParamsView: lambda: CellParamsView(_asu().cell),
    CellNumericView: lambda: CellNumericView(_asu().cell),
    ChemicalFormulaView: lambda: ChemicalFormulaView(_asu()),
    CompositionView: lambda: CompositionView(_asu()),
    FormulatypeView: lambda: FormulatypeView(_asu()),
    BarePrototypeView: lambda: BarePrototypeView(_asu()),
    BarePrototypeLabel: lambda: BarePrototypeLabel(_asu()),
    BareProtostructureView: lambda: BareProtostructureView(_asu()),
    BareProtostructureLabel: lambda: BareProtostructureLabel(_asu()),
    PrototypeView: lambda: PrototypeView(_prototype()),
    PrototypeLabel: lambda: PrototypeLabel(_prototype()),
    ProtostructureView: lambda: ProtostructureView(_protostructure()),
    ProtostructureLabel: lambda: ProtostructureLabel(_protostructure()),
    ProtonettypeView: lambda: ProtonettypeView("AB_cF8_225_a_b__1~2"),
    ProtonettypeLabel: lambda: ProtonettypeLabel("AB_cF8_225_a_b__1~2"),
    ProtonetstructureView: lambda: ProtonetstructureView("AB_cF8_225_a_b__1~2:Cl-Na"),
    ProtonetstructureLabel: lambda: ProtonetstructureLabel("AB_cF8_225_a_b__1~2:Cl-Na"),
    StructuretypeView: lambda: StructuretypeView(_asu()),
    FundamentalDomainTemplateView: lambda: FundamentalDomainTemplateView(_asu()),
    CartesianSiteMomentsView: lambda: CartesianSiteMomentsView(CartesianSiteMoments(MOMENTS)),
    CrystalAxisSiteMomentsView: lambda: CrystalAxisSiteMomentsView(CrystalAxisSiteMoments(MOMENTS, _asu().cell)),
    TrajectoryView: lambda: TrajectoryView(_trajectory()),
}
OPTIONAL = {"ASEAtomsView": "ase", "PymatgenStructureView": "pymatgen"}


def _optional_builder(name: str) -> Callable[[], Any]:
    def build() -> Any:
        pytest.importorskip(OPTIONAL[name])
        return getattr(httk.atomistic, name)(_asu())

    return build


CASES = [pytest.param(builder, id=cls.__name__) for cls, builder in BUILDERS.items()] + [
    pytest.param(_optional_builder(name), id=name) for name in OPTIONAL
]


def _key(view: Any) -> tuple[str, Any]:
    # repr round-trips exactly and also compares numpy-backed values elementwise.
    try:
        plain = view.unview()
    except TypeError:  # interface-only views: compare the presented backend
        plain = view.unwrap()
    return repr(plain), getattr(plain, "id", None)


@pytest.mark.parametrize("build", CASES)
def test_view_round_trips_through_pickle(build: Callable[[], Any]) -> None:
    view = build()
    restored = pickle.loads(pickle.dumps(view))
    assert type(restored) is type(view)
    assert _key(restored) == _key(view)


def test_view_metadata_survives_pickling() -> None:
    restored = pickle.loads(pickle.dumps(UnitcellStructureView(_asu(), immutable_id="mat-1")))
    assert restored.immutable_id == "mat-1"


def test_stated_space_group_survives_pickling_and_deepcopy() -> None:
    from httk.atomistic import StructureSymmetry, UnitcellStructure

    symmetry = StructureSymmetry(space_group_it_number=225)
    for copied in (pickle.loads(pickle.dumps(symmetry)), copy.deepcopy(symmetry)):
        assert copied == symmetry
        assert [record["hall_entry"] for record in copied.matched_settings] == [
            record["hall_entry"] for record in symmetry.matched_settings
        ]
    view = UnitcellStructureView(
        UnitcellStructure(
            [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            [[0, 0, 0], ["1/2", "1/2", "1/2"]],
            species_at_sites=["Na", "Cl"],
            symmetry=symmetry,
        )
    )
    assert view.symmetry == symmetry
    for copied in (pickle.loads(pickle.dumps(view)), copy.deepcopy(view)):
        assert copied.symmetry == symmetry
        assert copied.space_group_it_number == 225


def test_every_atomistic_view_is_covered() -> None:
    for module in pkgutil.walk_packages(httk.atomistic.__path__, "httk.atomistic."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # optional integration dependency missing
            continue
    seen: set[type] = set()
    pending: list[type] = [View]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                pending.append(sub)
    covered = {cls.__name__ for cls in BUILDERS} | set(OPTIONAL)
    concrete = {cls.__name__ for cls in seen if cls.__module__.startswith("httk.atomistic.") and "__new__" in vars(cls)}
    assert concrete - covered == set()
