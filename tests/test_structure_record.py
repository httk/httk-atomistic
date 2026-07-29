from dataclasses import replace
from fractions import Fraction

import pytest
from httk.core import FracVector, SurdScalar

from httk.atomistic import (
    Cell,
    CellParams,
    Sites,
    Species,
    SpeciesRecord,
    Structure,
    StructureBackend,
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
    empty_mass_species = SpeciesRecord("empty", (), (), mass=(), mass_present=True)
    assert empty_mass_species.to_species().mass == ()

    attached_absent = SpeciesRecord("C", ("C",), (1.0,))
    encoded_attached_absent = SpeciesRecord("C", ("C",), (1.0,), attached=(), nattached=(), attached_present=False)
    assert encoded_attached_absent.attached is None
    attached_present = SpeciesRecord("C", ("C",), (1.0,), attached=("H",), nattached=(3,), attached_present=True)
    assert attached_present.to_species().attached == ("H",)
    empty_attached = SpeciesRecord("C", ("C",), (1.0,), attached=(), nattached=(), attached_present=True)
    assert empty_attached.to_species().attached == ()
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
            assert any("structure_record_basis" in statement for statement in cell_statements)
            assert any("structure_record_periodicity" in statement for statement in cell_statements)
            assert not any("structure_record_reduced_coords" in statement for statement in cell_statements)
            assert not any("structure_record_species" in statement for statement in cell_statements)
            assert "basis" in row.__dict__
            assert "periodicity" in row.__dict__
            assert "reduced_coords" not in row.__dict__
            assert "species" not in row.__dict__

            _ = view.sites
            assert any("structure_record_reduced_coords" in statement for statement in statements)
            assert "reduced_coords" in row.__dict__
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)
