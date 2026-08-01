import datetime
from dataclasses import replace
from fractions import Fraction

import pytest
from httk.core import FracVector, SurdScalar, stored_entry_projection

from httk.atomistic import (
    Assembly,
    ASUSite,
    ASUStructure,
    Cell,
    CellParams,
    ChemicalComposition,
    FundamentalDomainStructure,
    Sites,
    Spacegroup,
    Species,
    SpeciesRecord,
    Structure,
    StructureBackend,
    StructureEntryRecord,
    StructureRecord,
    StructureRecordBackend,
    UnitcellStructureView,
)


def _species() -> tuple[Species, ...]:
    return (
        Species(
            name="CH3",
            chemical_symbols=("C",),
            concentration=(1.0,),
            mass=(12.011,),
            original_name="methyl",
            attached=("H",),
            nattached=(3,),
        ),
        Species(name="mixed", chemical_symbols=("Na", "K"), concentration=(0.25, 0.75)),
    )


def _structure(cell: Cell) -> Structure:
    return Structure(
        cell,
        Sites(
            [[Fraction(1, 3), 0, 0], [Fraction(1, 2), Fraction(1, 4), 0]],
            precision=Fraction(1, 1000),
        ),
        _species(),
        ["CH3", "mixed"],
    )


def _literal_surd(radicand: int, numerator: int, denominator: int = 1) -> SurdScalar:
    return SurdScalar({radicand: FracVector(numerator, denominator)}, ())


def _literal_record() -> StructureRecord:
    half_negative = _literal_surd(1, -1, 2)
    sqrt3_over_2 = _literal_surd(3, 1, 2)
    basis = (
        _literal_surd(1, 1),
        _literal_surd(1, 0),
        _literal_surd(1, 0),
        half_negative,
        sqrt3_over_2,
        _literal_surd(1, 0),
        _literal_surd(1, 0),
        _literal_surd(1, 0),
        _literal_surd(1, 3),
    )
    return StructureRecord(
        basis=basis,
        reduced_coords=FracVector.create([["1/3", "2/3", 0]]),
        species=(SpeciesRecord("C", ("C",), (1.0,)),),
        species_at_sites=("C",),
        periodicity=[True, True, True],
        basis_precision=Fraction(1, 1000),
        coordinate_precision=Fraction(1, 1000),
    )


def test_rational_record_roundtrip_preserves_precision_and_species() -> None:
    original = _structure(
        Cell(
            [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
            precision=Fraction(1, 10000),
            periodicity=(True, True, True),
        )
    )
    record = StructureRecord.from_structure(original)

    assert record.to_structure() == UnitcellStructureView(original)
    assert record.basis_precision == Fraction(1, 10000)
    assert record.coordinate_precision == Fraction(1, 1000)
    assert record.species[0].to_species() == original.species[0]
    assert record.species[1].to_species() == original.species[1]
    assert record.species[0].mass_present
    assert not record.species[1].mass_present
    assert not record.species[1].attached_present


def test_hexagonal_record_roundtrip_keeps_surd_basis() -> None:
    original = _structure(Cell(CellParams((1, 1, 3, 90, 90, 120)).basis))
    record = StructureRecord.from_structure(original)

    assert any(not value.is_rational for value in record.basis)
    assert record.to_structure() == original
    rebuilt_basis = record.to_structure().cell.basis
    assert any(radicand != 1 for radicand in rebuilt_basis.radicands)


def test_slab_record_roundtrip_preserves_periodicity() -> None:
    original = _structure(Cell([[4, 0, 0], [0, 5, 0], [0, 0, 20]], periodicity=(True, True, False)))
    record = StructureRecord.from_structure(original)

    assert record.periodicity == (True, True, False)
    assert record.to_structure().periodicity == (True, True, False)
    assert record.to_structure() == original


def test_record_is_hashable_and_immutable() -> None:
    record = _literal_record()
    equal = replace(record)
    assert isinstance(record.periodicity, tuple)
    assert hash(record) == hash(equal)
    assert record == equal
    with pytest.raises(TypeError):
        record.periodicity[1] = False


def test_literal_record_roundtrips_with_explicit_row_major_surds() -> None:
    record = _literal_record()
    assert record.basis == (
        _literal_surd(1, 1),
        _literal_surd(1, 0),
        _literal_surd(1, 0),
        _literal_surd(1, -1, 2),
        _literal_surd(3, 1, 2),
        _literal_surd(1, 0),
        _literal_surd(1, 0),
        _literal_surd(1, 0),
        _literal_surd(1, 3),
    )
    rebuilt = record.to_structure()
    assert StructureRecord.from_structure(rebuilt) == record
    assert rebuilt.cell.scale == SurdScalar({1: FracVector(1, 1)}, ())


def test_scaled_cell_basis_is_folded_into_the_record() -> None:
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], scale=Fraction(7, 3))
    original = Structure(cell, [[0, 0, 0]], ["C"], ["C"])
    record = StructureRecord.from_structure(original)
    rebuilt = record.to_structure()

    assert rebuilt.cell.scale == SurdScalar({1: FracVector(1, 1)}, ())
    assert rebuilt.cell.basis == original.cell.basis
    assert StructureRecord.from_structure(rebuilt).basis == record.basis


def test_species_record_presence_states_and_contradictions() -> None:
    absent = SpeciesRecord("C", ("C",), (1.0,))
    assert absent.mass is None
    encoded_absent = SpeciesRecord("C", ("C",), (1.0,), mass=(), mass_present=False)
    assert encoded_absent.mass is None
    present = SpeciesRecord("C", ("C",), (1.0,), mass=(12.0,), mass_present=True)
    assert present.to_species().mass == (12.0,)
    with pytest.raises(ValueError, match="mass_present=True"):
        SpeciesRecord("C", ("C",), (1.0,), mass_present=True)
    with pytest.raises(ValueError, match="mass_present=False"):
        SpeciesRecord("C", ("C",), (1.0,), mass=(12.0,))
    with pytest.raises(ValueError, match="valid Species"):
        SpeciesRecord("empty", (), (), mass=(), mass_present=True)

    attached_absent = SpeciesRecord("C", ("C",), (1.0,))
    encoded_attached_absent = SpeciesRecord("C", ("C",), (1.0,), attached=(), nattached=(), attached_present=False)
    assert encoded_attached_absent.attached is None
    attached_present = SpeciesRecord("C", ("C",), (1.0,), attached=("H",), nattached=(3,), attached_present=True)
    assert attached_present.to_species().attached == ("H",)
    with pytest.raises(ValueError, match="valid Species"):
        SpeciesRecord("C", ("C",), (1.0,), attached=(), nattached=(), attached_present=True)
    assert attached_absent == absent
    with pytest.raises(ValueError):
        SpeciesRecord("C", ("C",), (1.0,), attached=("H",), nattached=None)
    with pytest.raises(ValueError, match="attached_present=True"):
        SpeciesRecord("C", ("C",), (1.0,), attached_present=True)
    with pytest.raises(ValueError, match="attached_present=False"):
        SpeciesRecord("C", ("C",), (1.0,), attached=("H",), nattached=(3,))


def test_species_record_rejects_non_finite_floats() -> None:
    with pytest.raises(ValueError, match="finite"):
        SpeciesRecord("C", ("C",), (float("nan"),))
    with pytest.raises(ValueError, match="finite"):
        SpeciesRecord("C", ("C",), (1.0,), mass=(float("inf"),), mass_present=True)


def test_species_record_preserves_non_dyadic_concentration_and_mixed_precision() -> None:
    species = Species(
        "mixed",
        ("Ge", "Si"),
        (Fraction(1, 3), "0.6666"),
        concentration_precision=(None, Fraction(1, 10000)),
    )
    record = SpeciesRecord.from_species(species)
    assert record.concentration == (Fraction(1, 3), Fraction(3333, 5000))
    assert record.concentration_precision == (Fraction(), Fraction(1, 10000))
    assert record.concentration_precision_present
    assert record.to_species() == species


def test_structure_record_rejects_invalid_domain_combinations() -> None:
    record = _literal_record()
    duplicate = SpeciesRecord("C", ("C",), (1.0,))
    with pytest.raises(ValueError):
        replace(record, species=(record.species[0], duplicate))
    with pytest.raises(ValueError):
        replace(record, species_at_sites=("missing",))
    with pytest.raises(ValueError):
        replace(record, reduced_coords=FracVector.create([0, 0, 0]))
    with pytest.raises(ValueError):
        replace(record, basis_precision=Fraction(0))
    with pytest.raises(ValueError):
        replace(record, coordinate_precision=Fraction(-1))


def test_record_construction_does_not_build_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Structure construction is not part of StructureRecord validation")

    monkeypatch.setattr(Structure, "__init__", fail)
    _literal_record()


@pytest.mark.parametrize(
    ("species", "species_at_sites"),
    [(["C"], ["C"]), ([26], ["Fe"])],
)
def test_shorthand_species_record_roundtrip(species: list[object], species_at_sites: list[str]) -> None:
    original = Structure(
        cell=(1, 1, 3, 90, 90, 120),
        sites=[["1/3", "2/3", 0]],
        species=species,
        species_at_sites=species_at_sites,
    )
    for source in (original, UnitcellStructureView(original)):
        assert StructureRecord.from_structure(source).to_structure() == UnitcellStructureView(source)


def test_record_is_a_structure_like_backend() -> None:
    original = _structure(Cell([[4, 0, 0], [0, 5, 0], [0, 0, 6]]))
    record = StructureRecord.from_structure(original)

    backend = StructureBackend.create(record)
    assert isinstance(backend, StructureRecordBackend)
    assert backend.unwrap() is record
    assert StructureBackend.create(record, kind="record") is not None
    with pytest.raises(TypeError):
        StructureBackend.create(record, kind="unitcell")
    assert UnitcellStructureView(record) == record.to_structure()


def test_record_schema_and_sql_roundtrip() -> None:
    pytest.importorskip("httk.data")
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore
    from httk.data.db.schema import resolve_schema

    species_schema = resolve_schema(SpeciesRecord)
    record_schema = resolve_schema(StructureRecord)
    assert species_schema.field("mass").optional
    assert species_schema.field("attached").optional
    assert record_schema.field("basis").codec_name == "surdscalar"
    assert record_schema.field("periodicity").python_type == tuple[bool, ...]

    original = _structure(Cell(CellParams((1, 1, 3, 90, 90, 120)).basis, periodicity=(True, True, False)))
    record = StructureRecord.from_structure(original)
    database = Database.sqlite()
    sid = SqlStore(database).save(record)
    fetched = SqlStore(database).fetch(StructureRecord, sid)

    assert fetched is not record
    assert fetched.to_structure() == original
    assert fetched.basis == record.basis
    assert fetched.species == record.species
    assert fetched.species[1].to_species() == original.species[1]
    assert isinstance(fetched.periodicity, tuple)


def test_lazy_record_backend_decodes_components_independently() -> None:
    pytest.importorskip("httk.data")
    pytest.importorskip("sqlalchemy")
    import sqlalchemy
    from httk.data.db import Database, SqlStore

    record = _literal_record()
    with Database.sqlite() as database:
        store = SqlStore(database)
        store.save(record)
        searcher = store.searcher()
        variable = searcher.variable(StructureRecord)
        searcher.output(variable, "record")
        row = next(iter(searcher))[0][0]
        statements: list[str] = []

        def count_select(
            _connection: object,
            _cursor: object,
            statement: str,
            _parameters: object,
            _context: object,
            _executemany: bool,
        ) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            view = UnitcellStructureView(row)
            assert "basis" not in row.__dict__
            assert "reduced_coords" not in row.__dict__
            assert "species" not in row.__dict__

            _ = view.cell
            cell_statements = statements[:]
            assert any("atomistic_structure_v3_basis" in statement for statement in cell_statements)
            assert any("atomistic_structure_v3_periodicity" in statement for statement in cell_statements)
            assert not any("atomistic_structure_v3_reduced_coords" in statement for statement in cell_statements)
            assert not any("atomistic_structure_v3_species" in statement for statement in cell_statements)
            assert "basis" in row.__dict__
            assert "periodicity" in row.__dict__
            assert "reduced_coords" not in row.__dict__
            assert "species" not in row.__dict__

            _ = view.sites
            assert any("atomistic_structure_v3_reduced_coords" in statement for statement in statements)
            assert "reduced_coords" in row.__dict__
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)


def _domain(*, asu: bool, molecular: bool = False, representative: bool = True):
    cls = ASUStructure if asu else FundamentalDomainStructure
    return cls(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], precision=Fraction(1, 1000)),
        Spacegroup.standard(221),
        (
            ASUSite(
                "a",
                FracVector.create(()),
                "Cs",
                FracVector.create((0, 0, 0)) if representative else None,
            ),
        ),
        (Species("Cs", ("Cs",), (1,)),),
        coordinate_precision=Fraction(1, 10000),
        molecular=molecular,
        assemblies=(Assembly(((0,),), (1,)),),
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
        chemical_formula_descriptive="CsH2",
        optimization_type="experimental",
    )


@pytest.mark.parametrize(("asu", "molecular"), [(False, False), (True, False), (True, True)])
def test_tagged_domain_record_roundtrip_preserves_representation_and_semantics(asu: bool, molecular: bool) -> None:
    original = _domain(asu=asu, molecular=molecular)
    record = StructureRecord.from_structure(original)
    rebuilt = record.to_structure()

    assert record.representation == ("asymmetric_unit" if asu else "fundamental_domain")
    assert record.molecular is molecular
    assert record.domain_sites[0].representative_present
    assert record.domain_sites[0].representative == (Fraction(0), Fraction(0), Fraction(0))
    assert type(rebuilt) is type(original)
    assert rebuilt == original
    assert StructureRecord.from_structure(UnitcellStructureView(original)) == record


def test_domain_record_preserves_absent_representative_and_rejects_coordinate_contradiction() -> None:
    record = StructureRecord.from_structure(_domain(asu=False, representative=False))
    assert not record.domain_sites[0].representative_present
    assert record.domain_sites[0].representative == ()
    assert record.to_structure().domain_sites[0].representative is None

    with pytest.raises(ValueError, match="reduced_coords disagrees"):
        replace(record, reduced_coords=FracVector.create([[Fraction(1, 2), 0, 0]]))
    with pytest.raises(ValueError, match="representative values require"):
        replace(record.domain_sites[0], representative=(Fraction(0),) * 3)


def test_domain_record_accepts_periodically_equivalent_retained_representative() -> None:
    domain = _domain(asu=True)
    shifted = ASUStructure(
        domain.cell,
        domain.spacegroup,
        (ASUSite("a", (), "Cs", representative=(1, 0, 0)),),
        domain.species,
        coordinate_precision=domain.coordinate_precision,
    )
    record = StructureRecord.from_structure(shifted)
    assert record.domain_sites[0].representative == (Fraction(1), Fraction(0), Fraction(0))
    assert record.to_structure() == shifted


def test_nested_stored_entry_payloads_hide_precision_and_presence_internals() -> None:
    species = SpeciesRecord.from_species(
        Species("mixed", ("Ge", "Si"), (Fraction(1, 3), Fraction(2, 3)), concentration_precision=(None, "0.001"))
    )
    assembly = StructureRecord.from_structure(_domain(asu=False)).assemblies
    assert assembly is not None

    species_payload = species.to_stored_entry_value()
    assembly_payload = assembly[0].to_stored_entry_value()
    assert set(species_payload) == {"name", "chemical_symbols", "concentration"}
    assert set(assembly_payload) == {"sites_in_groups", "group_probabilities"}
    assert "concentration_precision" not in species_payload
    assert "groups" not in assembly_payload


def test_structure_entry_record_projection_and_authoritative_cross_checks() -> None:
    stamp = datetime.datetime(2026, 8, 1, 12, 30, tzinfo=datetime.UTC)
    entry = StructureEntryRecord.from_structure(
        _domain(asu=True), id="cs-domain", immutable_id="cs-domain-v1", last_modified=stamp
    )
    projection = stored_entry_projection(StructureEntryRecord)

    assert projection is not None
    assert projection.entry_type == "structures"
    assert projection.obsolete_storage_names == ("structure_record",)
    assert set(projection.property_fields) >= {
        "id",
        "chemical_formula_reduced",
        "lattice_vectors",
        "species",
        "assemblies",
        "site_coordinate_span",
    }
    assert {"id", "last_modified", "elements", "nsites", "site_coordinate_span"} <= projection.filterable
    assert entry.site_coordinate_span == "asymmetric_unit"
    assert entry.last_modified == stamp
    assert len(entry.lattice_vectors.to_stored_entry_value()) == 3
    assert len(entry.cartesian_site_positions.to_stored_entry_value()) == entry.nsites
    with pytest.raises(ValueError, match="denormalized fields disagree"):
        replace(entry, chemical_formula_reduced="Cs2")


def test_structure_entry_record_supports_implicit_only_zero_site_structure() -> None:
    structure = Structure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        Sites(()),
        (),
        (),
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
    )
    entry = StructureEntryRecord.from_structure(structure, id="implicit-only")
    assert entry.nsites == 0
    assert entry.cartesian_site_positions.to_stored_entry_value() == []
    assert entry.structure_features == ("implicit_atoms",)


def test_stored_entry_provider_serves_standard_nested_values() -> None:
    pytest.importorskip("httk.data")
    pytest.importorskip("sqlalchemy")
    from httk.data.db import Database, SqlStore, StoreEntryProvider

    with Database.sqlite() as database:
        store = SqlStore(database)
        store.save(StructureEntryRecord.from_structure(_domain(asu=True), id="stored-domain"))
        provider = StoreEntryProvider(store, {"structures": StructureEntryRecord})
        (record,) = tuple(provider.records("structures"))

    assert record["__id"] == "stored-domain"
    assert record["site_coordinate_span"] == "asymmetric_unit"
    assert record["lattice_vectors"] == [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    assert record["cartesian_site_positions"] == [[0.0, 0.0, 0.0]]
    assert record["fractional_site_positions"] == [[0.0, 0.0, 0.0]]
    assert record["species"] == [{"name": "Cs", "chemical_symbols": ["Cs"], "concentration": [1.0]}]
    assert record["assemblies"] == [{"sites_in_groups": [[0]], "group_probabilities": [1.0]}]
    assert "concentration_precision_present" not in str(record["species"])


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_exact_tagged_structure_and_entry_sql_roundtrip(dialect: str) -> None:
    pytest.importorskip("httk.data")
    pytest.importorskip("sqlalchemy")
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.data.db import Database, SqlStore
    from httk.data.db.schema import resolve_schema

    database = Database.sqlite() if dialect == "sqlite" else Database.duckdb()
    structure = StructureRecord.from_structure(_domain(asu=True))
    entry = StructureEntryRecord.from_structure(_domain(asu=True), id="stored-domain")
    assert resolve_schema(StructureRecord).table_name == "atomistic_structure_v3"
    assert resolve_schema(StructureEntryRecord).table_name == "atomistic_structure_entry_v3"
    assert resolve_schema(StructureEntryRecord).field("id").columns[0].unique
    with database:
        store = SqlStore(database)
        structure_sid = store.save(structure)
        entry_sid = store.save(entry)
        fetched_structure = store.fetch(StructureRecord, structure_sid)
        fetched_entry = store.fetch(StructureEntryRecord, entry_sid)

    assert fetched_structure == structure
    assert fetched_structure.to_structure() == structure.to_structure()
    assert fetched_entry == entry
    assert fetched_entry.structure == structure
