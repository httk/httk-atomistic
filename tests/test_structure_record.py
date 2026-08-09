"""Tests for representation-specific atomistic storage records."""

import datetime
import math
from dataclasses import fields
from fractions import Fraction

import pytest
from httk.core import FracVector
from httk.core.storage import content_id, project_storage_record, storage_identity_name

from httk.atomistic import (
    ASUStructure,
    ASUStructureRecord,
    ASUStructureView,
    FundamentalDomainStructure,
    FundamentalDomainStructureRecord,
    SettingTransform,
    Species,
    StructureEntry,
    UnitcellStructure,
    UnitcellStructureRecord,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.models.moments import CartesianSiteMoments, CollinearSiteMoments, CrystalAxisSiteMoments
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.storage.records import (
    AssemblyGroupRecord,
    AssemblyRecord,
    CellRecord,
    ChemicalCompositionRecord,
    CompositionAmountRecord,
    NormalizedCompositionAmountRecord,
    NormalizedCompositionRecord,
    SettingTransformRecord,
    SitesRecord,
    SpeciesConstituentRecord,
    SpeciesRecord,
    SymmetryRecord,
    WyckoffSiteRecord,
    _domain_structure_from_record,
    _structure_from_record,
    validate_structure_record,
)


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))


def _mixed_precision_unitcell() -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0]],
        (
            Species(
                "FeNi",
                ("Fe", "Ni"),
                (Fraction(1, 2), Fraction(1, 2)),
                concentration_precision=(None, Fraction(1, 1000)),
            ),
        ),
        ("FeNi",),
    )


def _unitcell(**metadata: object) -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        _species(),
        ("Na", "Cl"),
        **metadata,
    )


def _domain(
    record_type: type[FundamentalDomainStructure] = ASUStructure, **metadata: object
) -> FundamentalDomainStructure:
    return record_type(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        225,
        (
            WyckoffSite("a", FracVector(()), "Na"),
            WyckoffSite("b", FracVector(()), "Cl"),
        ),
        _species(),
        **metadata,
    )


def _cell_record(source: object) -> CellRecord:
    return CellRecord(**project_storage_record(CellRecord, source))  # type: ignore[arg-type]


def _common(source: object) -> dict[str, object]:
    value = source  # keep the helper compact without weakening public annotations
    return {
        "cell": _cell_record(value.cell),  # type: ignore[attr-defined]
        "species": tuple(SpeciesRecord(**SpeciesRecord.__httk_project__(item)) for item in value.species),  # type: ignore[attr-defined]
        "normalized_composition": NormalizedCompositionRecord(
            tuple(
                NormalizedCompositionAmountRecord(*item)
                for item in NormalizedCompositionRecord.__httk_project__(value.composition)["amounts"]  # type: ignore[index]
            ),
            value.composition.complete,  # type: ignore[attr-defined]
        ),
        "charge": value.charge,  # type: ignore[attr-defined]
        "molecular": value.molecular,  # type: ignore[attr-defined]
        "assemblies": (
            None
            if value.assemblies is None  # type: ignore[attr-defined]
            else tuple(AssemblyRecord(**AssemblyRecord.__httk_project__(item)) for item in value.assemblies)  # type: ignore[attr-defined]
        ),
        "chemical_composition": (
            None
            if value.chemical_composition is None  # type: ignore[attr-defined]
            else ChemicalCompositionRecord(**ChemicalCompositionRecord.__httk_project__(value.chemical_composition))  # type: ignore[attr-defined]
        ),
        "chemical_formula_descriptive": value.chemical_formula_descriptive,  # type: ignore[attr-defined]
        "chemical_formula_hill": value.chemical_formula_hill,  # type: ignore[attr-defined]
        "optimization_type": value.optimization_type,  # type: ignore[attr-defined]
        "immutable_id": value.immutable_id,  # type: ignore[attr-defined]
        "last_modified": value.last_modified,  # type: ignore[attr-defined]
    }


def _unitcell_record(source: UnitcellStructure) -> UnitcellStructureRecord:
    common = _common(source)
    return UnitcellStructureRecord(
        **common,
        sites=SitesRecord(**project_storage_record(SitesRecord, source.sites)),
        species_at_sites=source.species_at_sites,
        site_moments_kind=None if source.site_moments is None else source.site_moments.kind,
        site_moments=None
        if source.site_moments is None
        else project_storage_record(UnitcellStructureRecord, source)["site_moments"],
        site_moments_precision=None if source.site_moments is None else source.site_moments.precision,
        symmetry=None
        if source.symmetry is None
        else SymmetryRecord(**SymmetryRecord.__httk_project__(source.symmetry)),
    )


def _domain_record(
    source: FundamentalDomainStructure,
) -> FundamentalDomainStructureRecord | ASUStructureRecord:
    record_type = ASUStructureRecord if isinstance(source, ASUStructure) else FundamentalDomainStructureRecord
    return record_type(
        **_common(source),
        domain_sites=tuple(
            WyckoffSiteRecord(**project_storage_record(WyckoffSiteRecord, site)) for site in source.domain_sites
        ),
        spacegroup_it_number=source.spacegroup.it_number,
        setting_transform=SettingTransformRecord(**SettingTransformRecord.__httk_project__(source.transform)),
        coordinate_precision=source.coordinate_precision,
    )


def test_exact_source_bindings_are_representation_local() -> None:
    assert vars(UnitcellStructure)["__httk_storage_record__"] is UnitcellStructureRecord
    assert vars(UnitcellStructureView)["__httk_storage_record__"] is UnitcellStructureRecord
    assert vars(FundamentalDomainStructure)["__httk_storage_record__"] is FundamentalDomainStructureRecord
    assert vars(ASUStructure)["__httk_storage_record__"] is ASUStructureRecord
    assert vars(ASUStructureView)["__httk_storage_record__"] is ASUStructureRecord
    assert "__httk_storage_record__" not in vars(StructureView)


def test_projected_and_materialized_unitcell_have_the_same_content_id() -> None:
    source = _unitcell()
    record = _unitcell_record(source)
    assert (record.site_moments_kind, record.site_moments, record.site_moments_precision) == (None, None, None)
    assert content_id(source) == content_id(record) == source.id == record.id


@pytest.mark.parametrize(
    "moments",
    (
        CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]], precision=Fraction(1, 100)),
        CrystalAxisSiteMoments([[1, 2, 3], [-1, 0, 1]], _unitcell().cell, precision=Fraction(1, 100)),
        CollinearSiteMoments([1, -1], precision=Fraction(1, 100)),
    ),
)
def test_unitcell_moments_round_trip_exactly(
    moments: CartesianSiteMoments | CrystalAxisSiteMoments | CollinearSiteMoments,
) -> None:
    source = _unitcell(site_moments=moments)
    record = _unitcell_record(source)
    rebuilt = _structure_from_record(record)
    assert rebuilt.site_moments == source.site_moments
    assert rebuilt.site_moments.precision == source.site_moments.precision
    viewed = UnitcellStructureView(record)
    assert viewed.site_moments == source.site_moments
    assert viewed.site_moments.precision == source.site_moments.precision


def test_domain_site_moments_round_trip_exactly() -> None:
    source = ASUStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        225,
        (
            WyckoffSite("a", (), "Na", moment=CartesianSiteMoments([[1, 2, 3]], precision=Fraction(1, 100))),
            WyckoffSite("b", (), "Cl", moment=CartesianSiteMoments([[-1, 0, 1]], precision=Fraction(1, 100))),
        ),
        _species(),
    )
    record = _domain_record(source)
    rebuilt = _domain_structure_from_record(record)
    assert rebuilt.domain_sites == source.domain_sites
    assert all(site.moment.precision == Fraction(1, 100) for site in rebuilt.domain_sites if site.moment is not None)
    assert UnitcellStructureView(record).site_moments == UnitcellStructureView(source).site_moments


@pytest.mark.parametrize(
    "record_type",
    (
        SpeciesRecord,
        SpeciesConstituentRecord,
        AssemblyGroupRecord,
        AssemblyRecord,
        WyckoffSiteRecord,
        SettingTransformRecord,
        SymmetryRecord,
        CompositionAmountRecord,
        ChemicalCompositionRecord,
        NormalizedCompositionRecord,
        NormalizedCompositionAmountRecord,
        CellRecord,
        SitesRecord,
        UnitcellStructureRecord,
        FundamentalDomainStructureRecord,
        ASUStructureRecord,
    ),
)
def test_record_identity_names_are_storage_names(record_type: type[object]) -> None:
    identity_name = storage_identity_name(record_type)
    assert identity_name == record_type.__httk_storage__.storage_name  # type: ignore[attr-defined]
    assert "httk.atomistic" not in identity_name


def test_content_id_is_layout_independent_without_sqlalchemy() -> None:
    source = _unitcell()

    # 2026-08-06 species constituent child record + structure charge fields:
    # pre-existing stores must be rebuilt; a future move must not change this value.
    assert content_id(source) == "dfe119d07685701a738b7188f360be9707982a2ecb457027b5c2b9916d7129da"


def test_sql_store_unitcell_rename_preserves_content_id() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore

    source = _unitcell()
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        store.save(source)

    assert content_id(source) == "dfe119d07685701a738b7188f360be9707982a2ecb457027b5c2b9916d7129da"


def test_sql_store_round_trips_site_moments() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore

    source = _unitcell(site_moments=CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]], precision=Fraction(1, 100)))
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        fetched = store.fetch(UnitcellStructureRecord, store.save(source))

    assert _structure_from_record(fetched).site_moments == source.site_moments
    assert fetched.site_moments_precision == Fraction(1, 100)


@pytest.mark.parametrize(
    "structure_type",
    (FundamentalDomainStructure, ASUStructure),
)
def test_setting_transform_provenance_does_not_change_structure_identity(
    structure_type: type[FundamentalDomainStructure],
) -> None:
    first = _domain(structure_type, transform=SettingTransform([[1, 0, 0], [0, 1, 0], [0, 0, 1]], hall_entry="one"))
    second = _domain(structure_type, transform=SettingTransform([[1, 0, 0], [0, 1, 0], [0, 0, 1]], hall_entry="two"))
    first_record = _domain_record(first)
    second_record = _domain_record(second)

    assert first.transform == second.transform
    assert first == second
    assert first_record == second_record
    assert first.id == second.id == first_record.id == second_record.id


def test_metadata_round_trips_without_changing_identity_or_equality() -> None:
    stamp = datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.UTC)
    plain = _unitcell()
    annotated = _unitcell(immutable_id="source-7", last_modified=stamp)
    assert plain == annotated
    assert plain.id == annotated.id

    plain_record = _unitcell_record(plain)
    record = _unitcell_record(annotated)
    assert plain_record == record
    assert plain_record.id == record.id
    rebuilt = UnitcellStructureView(record)
    assert rebuilt.immutable_id == "source-7"
    assert rebuilt.last_modified == stamp


def test_views_inherit_metadata_and_reject_explicit_conflicts() -> None:
    stamp = datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.UTC)
    same_instant = stamp.astimezone(datetime.timezone(datetime.timedelta(hours=2)))

    unitcell = UnitcellStructureView(_unitcell(immutable_id="unit-7", last_modified=stamp))
    assert unitcell.immutable_id == "unit-7"
    assert unitcell.last_modified == stamp
    assert UnitcellStructureView(unitcell, immutable_id="unit-7", last_modified=same_instant) is unitcell
    with pytest.raises(ValueError, match="immutable_id conflicts"):
        UnitcellStructureView(unitcell, immutable_id="other")
    with pytest.raises(ValueError, match="last_modified conflicts"):
        UnitcellStructureView(unitcell, last_modified=None)

    bare_unitcell = UnitcellStructureView(_unitcell())
    attached_unitcell = UnitcellStructureView(bare_unitcell, immutable_id="attached", last_modified=stamp)
    assert attached_unitcell is not bare_unitcell
    assert bare_unitcell.immutable_id is None
    assert bare_unitcell.last_modified is None
    assert attached_unitcell.immutable_id == "attached"
    assert attached_unitcell.last_modified == stamp

    asu = ASUStructureView(_domain(immutable_id="asu-7", last_modified=stamp))
    assert asu.immutable_id == "asu-7"
    assert asu.last_modified == stamp
    assert ASUStructureView(asu, immutable_id="asu-7", last_modified=same_instant) is asu
    with pytest.raises(ValueError, match="immutable_id conflicts"):
        ASUStructureView(asu, immutable_id=None)

    bare_asu = ASUStructureView(_domain())
    attached_asu = ASUStructureView(bare_asu, immutable_id="attached", last_modified=stamp)
    assert attached_asu is not bare_asu
    assert bare_asu.immutable_id is None
    assert bare_asu.last_modified is None
    assert attached_asu.immutable_id == "attached"
    assert attached_asu.last_modified == stamp
    expanded = UnitcellStructureView(attached_asu)
    assert expanded.immutable_id == "attached"
    assert expanded.last_modified == stamp


def test_unitcell_record_is_a_structure_like_view_source() -> None:
    source = _unitcell()
    record = _unitcell_record(source)
    view = UnitcellStructureView(record)
    assert view == source
    assert view.unwrap() is record
    assert view.id == record.id


def test_mixed_species_precision_survives_record_and_sql_views() -> None:
    source = _mixed_precision_unitcell()
    expected = (None, Fraction(1, 1000))
    record = _unitcell_record(source)

    assert UnitcellStructureView(record).species[0].concentration_precision == expected

    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore

    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        sid = store.save(source)
        fetched = store.fetch(UnitcellStructureRecord, sid)
        assert UnitcellStructureView(fetched).species[0].concentration_precision == expected


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_decorated_repeated_species_and_structure_charge_round_trip(dialect: str) -> None:
    decorated = Species(
        "FeFe",
        ("Fe", "Fe"),
        (Fraction(1, 2), Fraction(1, 2)),
        concentration_precision=(None, Fraction(1, 1000)),
        mass=(55.845, 55.846),
        charges=(None, Fraction(0)),
        spins=(None, Fraction(1, 2)),
        labels=("left", None),
        attached=("H",),
        nattached=(1,),
    )
    source = UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0]],
        (decorated,),
        ("FeFe",),
        charge=Fraction(-3, 2),
    )
    record = _unitcell_record(source)
    assert record.charge == Fraction(-3, 2)
    assert record.species[0].constituents[0].charge is None
    assert record.species[0].constituents[1].charge == Fraction(0)
    assert _structure_from_record(record).species == (decorated,)

    pytest.importorskip("sqlalchemy")
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.data.db import Database, SqlStore

    database = Database.duckdb() if dialect == "duckdb" else Database.sqlite()
    with database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        fetched = store.fetch(UnitcellStructureRecord, store.save(source))
        viewed = UnitcellStructureView(fetched)
        assert viewed.charge == source.charge
        assert viewed.species == source.species


@pytest.mark.parametrize("mass", (math.inf, -math.inf, math.nan))
def test_species_constituent_rejects_nonfinite_mass(mass: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SpeciesConstituentRecord("Fe", Fraction(1), mass=mass)


def test_assembly_record_rejects_mutable_group_impostors() -> None:
    class Impostor:
        def __init__(self) -> None:
            self.sites = [0]

    with pytest.raises(TypeError, match="AssemblyGroupRecord"):
        AssemblyRecord((Impostor(),), (Fraction(1),))  # type: ignore[arg-type]

    class MutableGroup(AssemblyGroupRecord):
        def __init__(self) -> None:
            object.__setattr__(self, "sites", [0])

    with pytest.raises(TypeError, match="AssemblyGroupRecord"):
        AssemblyRecord((MutableGroup(),), (Fraction(1),))


def test_native_root_schemas_are_distinct_and_not_tagged() -> None:
    unit_fields = {item.name for item in fields(UnitcellStructureRecord)}
    domain_fields = {item.name for item in fields(FundamentalDomainStructureRecord)}
    asu_fields = {item.name for item in fields(ASUStructureRecord)}
    assert {"cell", "sites", "species_at_sites"} <= unit_fields
    assert {"cell", "domain_sites", "spacegroup_it_number", "setting_transform"} <= domain_fields
    assert domain_fields == asu_fields
    assert "representation" not in unit_fields | domain_fields | asu_fields
    assert "sites" not in domain_fields


def test_domain_and_asu_records_keep_representation_identity_distinct() -> None:
    domain = _domain(FundamentalDomainStructure)
    asu = _domain(ASUStructure)
    domain_record = _domain_record(domain)
    asu_record = _domain_record(asu)
    assert isinstance(domain_record, FundamentalDomainStructureRecord)
    assert not isinstance(domain_record, ASUStructureRecord)
    assert isinstance(asu_record, ASUStructureRecord)
    assert domain.id != asu.id
    assert domain_record.id != asu_record.id
    assert content_id(domain) == domain_record.id
    assert content_id(asu) == asu_record.id


def test_asu_record_view_adopts_native_asu_without_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _domain_record(_domain())
    assert isinstance(record, ASUStructureRecord)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("recognition must not run for an ASU record")

    monkeypatch.setattr("httk.atomistic.models.structure.asu_view.recognize_asu", fail)
    view = ASUStructureView(record)
    assert view.space_group_it_number == 225
    assert view.domain_sites == ASUStructureView(record).domain_sites
    assert view.unwrap() is record


def test_asu_compares_equal_to_its_view_but_not_a_fundamental_domain() -> None:
    asu = _domain()
    view = ASUStructureView(asu)
    domain = _domain(FundamentalDomainStructure)

    assert asu == view
    assert view == asu
    assert asu.id == view.id
    assert domain != asu
    assert domain != view


def test_fundamental_domain_record_cannot_be_promoted_to_asu() -> None:
    record = _domain_record(_domain(FundamentalDomainStructure))
    with pytest.raises(ValueError, match="cannot promote"):
        ASUStructureView(record)


def test_unitcell_view_of_asu_record_preserves_expanded_symmetry() -> None:
    record = _domain_record(_domain())
    view = UnitcellStructureView(record)
    assert len(view.sites) == 8
    assert view.space_group_it_number == 225
    assert view.space_group_symmetry_operations_xyz is not None
    assert view.wyckoff_positions is not None
    assert len(view.wyckoff_positions) == len(view.sites)


def test_record_rejects_naive_last_modified() -> None:
    source = _unitcell()
    values = _common(source)
    values["last_modified"] = datetime.datetime(2026, 8, 2, tzinfo=datetime.UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        UnitcellStructureRecord(
            **values,
            sites=SitesRecord(**project_storage_record(SitesRecord, source.sites)),
            species_at_sites=source.species_at_sites,
            symmetry=None,
        )


def test_record_construction_defers_normalized_composition_validation() -> None:
    source = _unitcell()
    values = _common(source)
    values["normalized_composition"] = NormalizedCompositionRecord(
        (
            NormalizedCompositionAmountRecord("Cl", Fraction(1, 3), Fraction(1), None),
            NormalizedCompositionAmountRecord("Na", Fraction(2, 3), Fraction(2), None),
        ),
        True,
    )
    record = UnitcellStructureRecord(
        **values,
        sites=SitesRecord(**project_storage_record(SitesRecord, source.sites)),
        species_at_sites=source.species_at_sites,
        symmetry=None,
    )
    with pytest.raises(ValueError, match="normalized_composition contradicts"):
        validate_structure_record(record)

    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore

    with Database.sqlite() as database, pytest.raises(ValueError, match="normalized_composition contradicts"):
        SqlStore(database, entry_records={}).save(record)


def test_sql_fetch_of_a_root_record_does_not_reconstruct_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore

    import httk.atomistic.storage.records as structure_record_module

    source = _unitcell()
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        sid = store.save(source)

        def forbidden_reconstruction(*args: object, **kwargs: object) -> object:
            raise AssertionError("fetch must not reconstruct a root structure")

        monkeypatch.setattr(structure_record_module, "_structure_from_record", forbidden_reconstruction)
        fetched = store.fetch(UnitcellStructureRecord, sid)
        assert fetched.id == source.id


def test_asu_record_preserves_cross_orbit_deduplicated_composition() -> None:
    """The normalized relation must follow ASU expansion's exact global deduplication."""
    source = ASUStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        225,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("a", FracVector(()), "Na")),
        (Species("Na", ("Na",), (1,)),),
    )
    assert source.multiplicities() == (4, 0)
    record = _domain_record(source)

    assert isinstance(record, ASUStructureRecord)
    assert tuple((value.element, value.amount) for value in record.normalized_composition.amounts) == (
        ("Na", Fraction(4)),
    )
    assert UnitcellStructureView(record).composition.amounts == (("Na", Fraction(4)),)


def test_normalized_composition_record_requires_ratios_to_normalize_amounts() -> None:
    with pytest.raises(ValueError, match="normalize"):
        NormalizedCompositionRecord(
            (
                NormalizedCompositionAmountRecord("Cl", Fraction(1, 2), Fraction(2)),
                NormalizedCompositionAmountRecord("Na", Fraction(1, 2), Fraction(1)),
            ),
            True,
        )
    # Both a complete vacancy-only structure and an incomplete unknown-element
    # structure can have no stored real-element amounts.
    assert NormalizedCompositionRecord((), True).amounts == ()
    assert NormalizedCompositionRecord((), False).amounts == ()


@pytest.mark.parametrize(
    ("source", "record_type", "view_type"),
    (
        (_unitcell(immutable_id="unitcell"), UnitcellStructureRecord, UnitcellStructureView),
        (
            _domain(FundamentalDomainStructure, immutable_id="domain"),
            FundamentalDomainStructureRecord,
            UnitcellStructureView,
        ),
        (_domain(ASUStructure, immutable_id="asu"), ASUStructureRecord, ASUStructureView),
    ),
)
def test_sql_fetched_root_records_keep_identity_and_metadata(
    source: UnitcellStructure | FundamentalDomainStructure,
    record_type: type[UnitcellStructureRecord | FundamentalDomainStructureRecord | ASUStructureRecord],
    view_type: type[UnitcellStructureView] | type[ASUStructureView],
) -> None:
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore

    with Database.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={
                StructureEntry: (
                    UnitcellStructureRecord,
                    FundamentalDomainStructureRecord,
                    ASUStructureRecord,
                )
            },
        )
        sid = store.save(source)
        fetched = store.fetch(record_type, sid)

        assert fetched.id == source.id
        assert fetched.immutable_id == source.immutable_id
        view = view_type(fetched)
        assert view.unwrap() is fetched
        assert view.immutable_id == source.immutable_id


def test_unitcell_record_view_keeps_unread_cursor_fields_lazy() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, ExpiredCursorRowError, SqlStore

    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        store.save(_unitcell(optimization_type="local"))
        store.save(_unitcell(optimization_type="global"))
        searcher = store.searcher()
        record = searcher.variable(UnitcellStructureRecord)
        cursor = searcher.results(record=record).cursor()
        view = UnitcellStructureView(next(cursor).record)

        assert view.cell.basis == _unitcell().cell.basis
        next(cursor)
        with pytest.raises(ExpiredCursorRowError):
            _ = view.sites
