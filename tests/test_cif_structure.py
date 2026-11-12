"""``CIFStructure``: lossless CIF blocks as structure sources, their record, saving and serving.

The store tests run on SQLite, and on MongoDB when ``HTTK_TEST_MONGODB_URI`` is set.
"""

import dataclasses
import datetime
import gzip
import io
import json
import logging
import os
import uuid
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import DecimalGap, DecimalScalar, DecimalVector, FracVector
from httk.core.storage import content_id

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    CIFStructure,
    CIFStructureRecord,
    ModulatedStructure,
    Species,
    StructureEntryProvider,
    UnitcellStructure,
    UnitcellStructureRecord,
    UnitcellStructureView,
    read_cif_structures,
)
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.io.cif import (
    CIFBlock,
    CIFComment,
    CIFDataItem,
    CIFLoop,
    CIFSaveFrame,
    CIFTable,
    CIFTextField,
    read_cif,
    read_mcif_asus,
)
from httk.atomistic.io.cif.cif_reader import _legacy_block
from httk.atomistic.io.cif.cif_writer import write_cif
from httk.atomistic.io.cif.mcif_parser import _mcif_payload
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.storage.cif_record import _cif_structure_from_record

FIXTURES = Path(__file__).parent / "fixtures"
MALFORMED = FIXTURES / "malformed_auxiliary_loop.cif"
# Structure fixtures: every fixture but the synthetic syntax probes of cif_lossless (which test
# the fast reader's documented faults) and the malformed loop that lossless reading rejects (D3).
STRUCTURES = sorted(
    path
    for path in FIXTURES.rglob("*")
    if path.suffix in (".cif", ".mcif") and path.parent.name != "cif_lossless" and path != MALFORMED
)
MULTIBLOCK = FIXTURES / "cif_lossless" / "multiblock.cif"
BONDED = FIXTURES / "cif_bonds" / "ordered.cif"
MAGNETIC = FIXTURES / "magnetic_kvector.mcif"


def _ids(path: Path) -> str:
    return str(path.relative_to(FIXTURES))


def _text(blocks: CIFBlock | list[CIFStructure]) -> str:
    out = io.StringIO()
    write_cif(out, blocks)
    return out.getvalue()


def _outcome(convert: Any) -> tuple[str, Any]:
    """Return ``("ok", structure)`` or ``("error", (type, message))``."""
    try:
        return "ok", convert()
    except (TypeError, ValueError) as error:
        return "error", (type(error), str(error))


def _same_mcif(left: Any, right: Any) -> bool:
    if isinstance(left, ModulatedStructure):
        return isinstance(right, ModulatedStructure) and left.payload == right.payload
    return UnitcellStructureView(left) == UnitcellStructureView(right)


@pytest.fixture(autouse=True)
def _quiet(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)


# ---------------------------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", STRUCTURES, ids=_ids)
def test_conversion_equals_loading_the_file(path: Path, tmp_path: Path) -> None:
    (cs,) = read_cif_structures(path)
    assert cs == CIFStructure.from_cif(path)
    assert cs.dialect == ("mcif" if path.suffix == ".mcif" else "cif")
    loaded = _outcome(lambda: httk.core.load(path))
    converted = _outcome(lambda: ASUStructureView(cs).resolve() if cs.dialect == "cif" else cs.resolve())
    saved = tmp_path / f"saved{path.suffix}"
    httk.core.save(cs, saved)
    reloaded = _outcome(lambda: httk.core.load(saved))
    if loaded[0] == "error":
        assert converted == loaded and reloaded == loaded
        return
    assert converted[0] == reloaded[0] == "ok"
    if cs.dialect == "cif":
        assert converted[1] == loaded[1] == reloaded[1]
        assert converted[1].bonds == loaded[1].bonds  # the bond network of the file
        assert cs.resolve() is cs.resolve()  # memoised
        return
    expected = dict(read_mcif_asus(path))
    payload = _mcif_payload([(cs.name.lower(), _legacy_block(cs, structural_only=False))], "")
    assert {**payload, "header": None} == {**expected, "header": None}
    assert _same_mcif(converted[1], loaded[1]) and _same_mcif(reloaded[1], loaded[1])


def test_the_bond_fixtures_carry_their_networks_through_conversion() -> None:
    bonded = [path for path in STRUCTURES if "_geom_bond_atom_site_label_1" in path.read_text(errors="replace")]
    assert len(bonded) >= 13
    for path in bonded:
        loaded = httk.core.load(path, repair=True)
        assert CIFStructure.from_cif(path).resolve(repair=True).bonds == loaded.bonds


def test_unit_cell_and_structure_members_convert() -> None:
    cs = CIFStructure.from_cif(BONDED)
    asu = httk.core.load(BONDED)
    assert UnitcellStructureView(cs) == UnitcellStructureView(asu)
    assert cs.species == asu.species and cs.species_at_sites == asu.species_at_sites
    assert cs.cell == asu.cell and cs.sites == asu.sites


BASE = """data_nacl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
x,y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0 0 0 1
Cl1 Cl 0.5 0.5 0.5 1
"""
# Valid CIF that is not a valid structure. load() accepts a plain repeated label and an unknown
# type symbol (an X pseudo-site); the labels repeated at one position and an element that cannot
# be inferred from a label are what it rejects.
INVALID = {
    "no cell": (BASE.replace("_cell_length_a 5.64\n", ""), "has no unit cell"),
    "occupancy above one": (BASE.replace("Na1 Na 0 0 0 1", "Na1 Na 0 0 0 1.5"), "must lie in"),
    "contradicting operations": (BASE.replace("IT_number 1", "IT_number 225"), "contradicts itself"),
    "duplicate atom labels": (BASE.replace("Cl1 Cl 0.5 0.5 0.5", "Na1 Cl 0 0 0"), "co-located sites 'Na1', 'Na1'"),
    "unknown element": (
        BASE.replace("_atom_site_type_symbol\n", "").replace("Na1 Na ", "Qq1 ").replace("Cl1 Cl ", "Cl1 "),
        "could not be inferred",
    ),
    "non-numeric coordinate": (BASE.replace("Cl1 Cl 0.5", "Cl1 Cl abc"), "Invalid CIF numeric token"),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_a_valid_cif_that_is_no_structure_loads_round_trips_and_raises_on_conversion(case: str, tmp_path: Path) -> None:
    text, message = INVALID[case]
    path = tmp_path / "x.cif"
    path.write_text(text)
    cs = CIFStructure.from_cif(path)
    assert read_cif_structures(io.StringIO(_text(cs))) == [cs]
    with pytest.raises(ValueError, match=message) as loaded:
        httk.core.load(path)
    with pytest.raises(ValueError) as converted:
        ASUStructureView(cs).resolve()
    assert str(converted.value) == str(loaded.value)


def test_a_repeated_data_name_raises_on_conversion_only_when_the_structure_reader_uses_it() -> None:
    repeated = BASE.replace("_cell_length_b 5.64\n", "_cell_length_b 5.64\n_Cell_Length_A 5.64\n")
    cs = CIFStructure.from_cif(io.StringIO(repeated))
    assert read_cif_structures(io.StringIO(_text(cs))) == [cs]
    with pytest.raises(ValueError, match="repeats data name '_cell_length_a'; a structure needs each data name once"):
        cs.resolve()
    in_frame = CIFStructure(
        "x",
        (*CIFStructure.from_cif(io.StringIO(BASE)).items, CIFSaveFrame("f", (CIFDataItem("_cell_angle_beta", "1"),))),
    )
    with pytest.raises(ValueError, match="repeats data name '_cell_angle_beta'"):
        in_frame.resolve()
    unused = BASE.replace("data_nacl\n", "data_nacl\n_journal_year 2026\n_journal_year 2027\n")
    assert CIFStructure.from_cif(io.StringIO(unused)).resolve() == CIFStructure.from_cif(io.StringIO(BASE)).resolve()


def test_a_block_without_atoms_is_a_cif_structure_that_does_not_convert() -> None:
    blocks = read_cif_structures(MULTIBLOCK)
    assert [block.name for block in blocks] == ["first", "notes", "last"]
    assert blocks[1].leading == (CIFComment(" trailing of first: moves to second.leading"),)
    with pytest.raises(ValueError, match="holds no structural data blocks"):
        blocks[1].resolve()
    with pytest.raises(ValueError, match="this CIF holds 3 data blocks; use read_cif_structures"):
        CIFStructure.from_cif(MULTIBLOCK)


def test_dialect_follows_the_suffix_and_mcif_rejects_the_large_uncertainty_override(tmp_path: Path) -> None:
    compressed = tmp_path / "m.mcif.gz"
    compressed.write_bytes(gzip.compress(MAGNETIC.read_bytes()))
    assert CIFStructure.from_cif(compressed) == CIFStructure.from_cif(MAGNETIC)
    assert CIFStructure.from_cif(MAGNETIC).dialect == "mcif"
    assert CIFStructure.from_cif(io.StringIO(MAGNETIC.read_text()), dialect="mcif").dialect == "mcif"
    with pytest.raises(ValueError, match="'cif' or 'mcif'"):
        read_cif_structures(MAGNETIC, dialect="cif2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="'cif' blocks"):
        CIFStructure.from_cif(MAGNETIC).resolve(allow_large_cif_uncertainty=True)


def test_numbers_are_exposed_as_decimal_vectors() -> None:
    cs = CIFStructure.from_cif(BONDED)
    lengths = cs.numbers("_cell_length_a", "_cell_length_b", "_cell_length_c")
    assert lengths == DecimalVector(["5.0", "6.0", "7.0"]) and FracVector(lengths) == FracVector([5, 6, 7])
    coordinates = cs.numbers("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    assert coordinates.dim == (3, 3) and coordinates[0] == DecimalVector(["0.1", "0.2", "0.3"])
    assert cs.numbers("_geom_bond_distance").dim[0] > 0
    assert cs.number("_cell_angle_beta") == DecimalScalar("100.0")


def test_repr_round_trips_through_the_constructor() -> None:
    cs = CIFStructure.from_cif(BONDED)
    namespace = {
        "CIFStructure": CIFStructure,
        "CIFComment": CIFComment,
        "CIFDataItem": CIFDataItem,
        "CIFLoop": CIFLoop,
        "CIFTextField": CIFTextField,
        "DecimalScalar": DecimalScalar,
        "DecimalGap": DecimalGap,
    }
    assert eval(repr(cs), namespace) == cs
    assert cs != CIFBlock(cs.name, cs.items, cs.header, cs.leading, cs.dialect)  # the class is content
    assert hash(cs) == hash(eval(repr(cs), namespace))


def test_backend_adoption_takes_cif_inputs_only() -> None:
    cs = CIFStructure.from_cif(BONDED)
    block = CIFBlock(cs.name, cs.items, cs.header, cs.leading, cs.dialect)
    assert CIFStructure._backend_adopt(cs) is cs
    assert CIFStructure._backend_adopt(block) == cs
    assert CIFStructure._backend_adopt(cs, kind="datastream") is None
    assert StructureBackend._select_backend(block) == cs
    for other in ("x.cif", BONDED, httk.core.load(BONDED), (1, 2, 3), None):
        assert CIFStructure._backend_adopt(other) is None
    invalid = CIFBlock("x", ())
    object.__setattr__(invalid, "name", "has space")  # a CIFBlock that bypassed its validation
    assert CIFStructure._backend_adopt(invalid) is None


# ---------------------------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------------------------
def test_save_writes_one_block_or_a_list_and_refuses_a_dialect_mismatch(tmp_path: Path) -> None:
    blocks = read_cif_structures(MULTIBLOCK)
    httk.core.save(blocks, tmp_path / "all.cif")
    assert read_cif_structures(tmp_path / "all.cif") == blocks
    assert (tmp_path / "all.cif").read_text() == _text(blocks)
    httk.core.save(blocks[0], tmp_path / "one.cif.gz", max_line_length=40)
    assert read_cif_structures(tmp_path / "one.cif.gz") == [blocks[0]]
    magnetic = CIFStructure.from_cif(MAGNETIC)
    httk.core.save(magnetic, tmp_path / "m.mcif")
    assert CIFStructure.from_cif(tmp_path / "m.mcif") == magnetic
    with pytest.raises(ValueError, match="has dialect 'mcif'; save it as .mcif"):
        httk.core.save(magnetic, tmp_path / "m.cif")
    with pytest.raises(ValueError, match="has dialect 'cif'; save it as .cif"):
        httk.core.save(blocks[0], tmp_path / "x.mcif")
    with pytest.raises(TypeError, match="carry their own header"):
        httk.core.save(blocks[0], tmp_path / "h.cif", header="# x")
    httk.core.save(blocks[0], tmp_path / "e.cif", approximate=False, exact_companions=True)  # no-ops
    assert read_cif_structures(tmp_path / "e.cif") == [blocks[0]]


def test_load_is_unchanged() -> None:
    assert isinstance(httk.core.load(BONDED), ASUStructure)


# ---------------------------------------------------------------------------------------------
# record and identity
# ---------------------------------------------------------------------------------------------
def _record(cs: CIFStructure) -> CIFStructureRecord:
    from httk.core.storage import project_storage_record

    return CIFStructureRecord(**project_storage_record(CIFStructureRecord, cs))  # type: ignore[arg-type]


def _reads(text: str) -> CIFStructure:
    return CIFStructure.from_cif(io.StringIO(text))


def test_identity_ignores_whitespace_and_quoting_but_not_comments() -> None:
    plain = "#h\ndata_a\n_x 'one two' # c\nloop_\n_y\n_z\n1 b\n2 'c'\n"
    respaced = "#h\ndata_a\n_x\n  \"one two\"   # c\nloop_ _y _z\n1 'b' 2\n c\n"
    moved = "#h\ndata_a\n_x 'one two'\n# c\nloop_\n_y\n_z\n1 b\n2 'c'\n"
    other = "#h\ndata_a\n_x 'one two' # d\nloop_\n_y\n_z\n1 b\n2 'c'\n"
    ids = [content_id(_reads(text)) for text in (plain, respaced, moved, other)]
    assert ids[0] == ids[1] and len(set(ids)) == 3
    assert content_id(_reads(plain)) == content_id(_record(_reads(plain)))
    assert _reads(plain) == _reads(respaced)
    assert content_id(_reads(plain.replace("data_a", "data_A"))) != ids[0]
    assert content_id(dataclasses.replace(_reads(plain), dialect="mcif")) != ids[0]


def test_every_value_kind_round_trips_through_the_record() -> None:
    cif2 = CIFStructure(
        "v",
        (
            CIFComment("first", inline=True),
            CIFDataItem("_s", "a b", ((0, CIFComment("before")), (1, CIFComment("after", inline=True)))),
            CIFDataItem("_t", CIFTextField("\nline one\n  line two")),
            CIFDataItem("_n", DecimalScalar("1.50(3)")),
            CIFDataItem("_g", DecimalGap.INAPPLICABLE),
            CIFDataItem("_l", ("x", (DecimalScalar("007"), DecimalGap.UNKNOWN))),
            CIFDataItem("_m", CIFTable((("k", "v"), ("j", ("1",))))),
            CIFLoop(("_p", "_q"), (("1", DecimalScalar("2")),), ((1, CIFComment("name", inline=True)),)),
            CIFSaveFrame("f", (CIFSaveFrame("g", (CIFDataItem("_e", "?"),)),)),
            CIFComment("last"),
        ),
        (CIFComment("\\#CIF_2.0"), CIFComment(" more")),
        (CIFComment("lead"),),
    )
    record = _record(cif2)
    CIFStructureRecord.__httk_validate__(record)
    assert _cif_structure_from_record(record) == cif2
    assert CIFStructure._backend_adopt(record) == cif2
    assert read_cif_structures(io.StringIO(_text(cif2))) == [cif2]


def test_the_record_rejects_non_canonical_and_malformed_content() -> None:
    record = _record(CIFStructure.from_cif(BONDED))
    spaced = dataclasses.replace(record, content_json=json.dumps(json.loads(record.content_json)))
    assert _cif_structure_from_record(spaced) == CIFStructure.from_cif(BONDED)
    with pytest.raises(ValueError, match="not canonical"):
        CIFStructureRecord.__httk_validate__(spaced)
    with pytest.raises(ValueError, match="canonical cif form 2, but this httk-atomistic uses 1.*rebuild the database"):
        CIFStructure._backend_adopt(dataclasses.replace(record, canonical_form=2))
    document = json.loads(record.content_json)
    for broken in (
        {**document, "extra": []},
        {**document, "items": [{"name": "_a", "value": {"number": "?"}, "comments": []}]},
        {**document, "items": [{"name": "_a", "value": {"float": 1.5}, "comments": []}]},
        {
            **document,
            "items": [{"name": "_a", "value": {"string": "x"}, "comments": [[True, {"comment": "c", "inline": True}]]}],
        },
        {**document, "header": [{"comment": "c", "inline": 1}]},
        {**document, "items": [{"save": "f", "items": [], "x": 1}]},
    ):
        with pytest.raises((TypeError, ValueError)):
            _cif_structure_from_record(dataclasses.replace(record, content_json=json.dumps(broken)))
    deep = (
        '{"header":[],"leading":[],"items":[{"name":"_a","value":' + '{"list":[' * 70 + "]}" * 70 + ',"comments":[]}]}'
    )
    with pytest.raises(ValueError, match="nested deeper than 64"):
        _cif_structure_from_record(dataclasses.replace(record, content_json=deep))
    with pytest.raises(ValueError, match="malformed document"):
        _cif_structure_from_record(dataclasses.replace(record, content_json="[" * 100_000))
    with pytest.raises(ValueError, match="dialect"):
        dataclasses.replace(record, dialect="cif2")
    with pytest.raises(TypeError):
        dataclasses.replace(record, canonical_form=True)


# ---------------------------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------------------------
IDS = ("httk.test", "1")


@pytest.fixture(params=["sqlite", "mongo"])
def store(request: pytest.FixtureRequest) -> Iterator[Any]:
    pytest.importorskip("httk.store")
    from httk.store import EntryIdScheme

    records = {StructureEntry: (CIFStructureRecord, UnitcellStructureRecord)}
    if request.param == "sqlite":
        pytest.importorskip("sqlalchemy")
        from httk.store import Backend, SqlStore

        with Backend.sqlite() as database:
            yield SqlStore(database, entry_ids=EntryIdScheme(*IDS), entry_records=records)
        return
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    pymongo = pytest.importorskip("pymongo")
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=1000)
    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        client.admin.command("ping")
    except Exception as error:
        client.close()
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        yield MongoStore(MongoDatabase(client, name), entry_ids=EntryIdScheme(*IDS), entry_records=records)
    finally:
        client.drop_database(name)
        client.close()


def test_a_store_round_trips_single_and_multi_block_files(store: Any, tmp_path: Path) -> None:
    for path in (BONDED, MAGNETIC, MULTIBLOCK, FIXTURES / "cif_lossless" / "comments.cif"):
        blocks = read_cif_structures(path)
        ids = [store.save(block) for block in blocks]  # one entry per block, in file order
        fetched = [store.fetch(CIFStructureRecord, entry_id, eager=True) for entry_id in ids]
        assert [_cif_structure_from_record(record) for record in fetched] == blocks
        assert all(content_id(record) == content_id(block) for record, block in zip(fetched, blocks, strict=True))
        written = tmp_path / f"back{path.suffix}"
        httk.core.save(fetched, written)  # records save as their blocks
        assert read_cif_structures(written) == read_cif_structures(path)
        assert [store.save(block) for block in blocks] == ids  # identical blocks deduplicate
    fetched_bonded = store.fetch(CIFStructureRecord, store.save(CIFStructure.from_cif(BONDED)), eager=True)
    assert ASUStructureView(fetched_bonded).resolve() == httk.core.load(BONDED)  # a record converts too


def test_stored_and_in_memory_rows_are_the_same_null_row(store: Any) -> None:
    from httk.store.backend.mongo import stored_property_mongo_plan
    from httk.store.backend.sql import SqlStore, stored_property_sql_plan

    timestamp = datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.UTC)
    no_cell = CIFStructure.from_cif(io.StringIO(INVALID["no cell"][0]))  # serving never converts
    cif_id = store.save(dataclasses.replace(_record(no_cell), last_modified=timestamp))
    unitcell = UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0]],
        (Species("Na", ("Na",), (Fraction(1, 2),)),),
        ("Na",),
    )
    store.save(unitcell)
    plan = (stored_property_sql_plan if isinstance(store, SqlStore) else stored_property_mongo_plan)(
        store, StructureEntry
    )
    rows = {row["id"]: row for row in plan.records()}
    fetched = store.fetch(CIFStructureRecord, cif_id, eager=True)
    (served,) = [row for row in rows.values() if row["immutable_id"] == fetched.immutable_id]
    expected = dict(next(iter(StructureEntryProvider({served["id"]: fetched}).records("structures"))))
    expected["id"] = expected.pop("__id")
    assert served == expected
    assert served["structure_features"] == [] and served["last_modified"] == timestamp.isoformat()
    assert all(served[name] is None for name in ("nsites", "elements", "lattice_vectors", "species"))

    def count(filter_string: str) -> int:
        return sum(searcher.count() for searcher in plan.filter_searchers(filter_string))

    assert count('structure_features HAS "disorder"') == 1  # the unit cell only
    assert count("structure_features LENGTH 0") == 1  # the CIF block only
    assert count("structure_features IS KNOWN") == 2
    assert count('last_modified = "2026-08-02T10:30:00Z"') == 1
    assert count("nsites = 1") == 1 and count("nsites IS UNKNOWN") == 1


def test_the_in_memory_provider_serves_without_converting() -> None:
    cs = CIFStructure.from_cif(io.StringIO(INVALID["no cell"][0]))
    (row,) = StructureEntryProvider({"x": cs}).records("structures")
    assert row["structure_features"] == [] and row["immutable_id"] is None and row["last_modified"] is None
    assert row["nsites"] is None and row["chemical_formula_reduced"] is None
    assert "_resolved_structure" not in vars(cs)
    (none_row,) = StructureEntryProvider({"x": None}).records("structures")
    assert {**none_row, "structure_features": []} == row


# ---------------------------------------------------------------------------------------------
# review follow-ups
# ---------------------------------------------------------------------------------------------
LOSSLESS = FIXTURES / "cif_lossless"


def test_the_fast_reader_divergences_are_as_documented() -> None:
    # FP1, fixed: comments in a loop header read alike in the fast reader and the lossless one.
    comments = CIFStructure.from_cif(LOSSLESS / "comments.cif")
    ((_, fast),), _ = read_cif(LOSSLESS / "comments.cif", structural_only=True)
    assert _legacy_block(comments, structural_only=True) == fast
    with pytest.raises(ValueError) as loaded:
        httk.core.load(LOSSLESS / "comments.cif")
    with pytest.raises(ValueError) as converted:
        comments.resolve()
    assert str(converted.value) == str(loaded.value)
    # D8: a quote before ']' in an mCIF list hides the atom sites from the fast reader only.
    with pytest.raises(ValueError, match="this mCIF holds no structure blocks$"):
        httk.core.load(LOSSLESS / "mcif_brackets.mcif")
    with pytest.raises(ValueError, match="block 'm': ValueError: CIF block has no unit cell"):
        CIFStructure.from_cif(LOSSLESS / "mcif_brackets.mcif").resolve()


CIF2_BASE = "#\\#CIF_2.0\n" + BONDED.read_text().split("\n", 1)[1]


@pytest.mark.parametrize(
    ("old", "new", "name"),
    (
        ("_cell_length_a 5.0", "_cell_length_a [5.0 5.0]", "_cell_length_a"),
        ("-x,-y,-z\n", "[-x,-y,-z]\n", "_space_group_symop_operation_xyz"),
        ("O2 O 0.12", "[O2] O 0.12", "_atom_site_label"),
        ("\nFe1 O1 ", "\n[Fe1] O1 ", "_geom_bond_atom_site_label_1"),
    ),
)
def test_a_cif2_list_in_a_name_the_structure_reader_reads_raises(old: str, new: str, name: str) -> None:
    assert old in CIF2_BASE
    cs = CIFStructure.from_cif(io.StringIO(CIF2_BASE.replace(old, new, 1)))
    with pytest.raises(ValueError, match=f"data name '{name}' holds a CIF2 list or table; a structure needs"):
        cs.resolve()
    table = CIFStructure.from_cif(io.StringIO(CIF2_BASE.replace(old, new.replace("[", "{'k':[").replace("]", "]}"), 1)))
    with pytest.raises(ValueError, match="holds a CIF2 list or table"):
        table.resolve()
    unread = CIFStructure.from_cif(io.StringIO(CIF2_BASE + "_journal_coden_astm [a b]\n"))
    assert unread.resolve() == CIFStructure.from_cif(BONDED).resolve()


def test_type_moments_charge_and_the_protonet_view() -> None:
    from httk.atomistic import ProtonetstructureView

    cs = CIFStructure.from_cif(BONDED)
    assert cs.type == "structures" and cs.charge == cs.resolve().charge
    assert ProtonetstructureView(cs).label == ProtonetstructureView(httk.core.load(BONDED)).label
    magnetic = CIFStructure.from_cif(FIXTURES / "magnetic_kvector.mcif")
    assert magnetic.site_moments is not None and magnetic.site_moments == magnetic.resolve().site_moments


def test_coerce_turns_a_record_into_its_block() -> None:
    from httk.atomistic.storage import CIFStructureRecord as Exported

    cs = CIFStructure.from_cif(BONDED)
    record = _record(cs)
    assert Exported is CIFStructureRecord
    assert httk.core.coerce(record, CIFStructure) == cs
    assert httk.core.coerce(CIFBlock(cs.name, cs.items, cs.header, cs.leading), CIFStructure) == cs
    with pytest.raises(TypeError):
        httk.core.coerce(httk.core.load(BONDED), CIFStructure)


def test_a_stream_name_sets_the_dialect_and_the_memo_is_not_pickled() -> None:
    import copy
    import pickle

    with MAGNETIC.open("rb") as stream:
        assert read_cif_structures(stream)[0].dialect == "mcif"
    cs = CIFStructure.from_cif(BONDED)
    cs.resolve()
    for copied in (pickle.loads(pickle.dumps(cs)), copy.deepcopy(cs)):
        assert copied == cs and "_resolved_structure" not in vars(copied)
    assert "_resolved_structure" in vars(cs)


def test_strings_utf8_cannot_encode_and_repeated_json_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="encodable as UTF-8"):
        CIFDataItem("_a", "x\ud800")
    with pytest.raises(ValueError, match="encodable as UTF-8"):
        CIFComment("\udfff")
    record = _record(CIFStructure("a", (CIFDataItem("_a", "x"),)))
    surrogate = dataclasses.replace(record, content_json=record.content_json.replace('"x"', '"\\ud800"'))
    with pytest.raises(ValueError, match="encodable as UTF-8"):
        _cif_structure_from_record(surrogate)
    repeated = dataclasses.replace(record, content_json=record.content_json[:-1] + ',"items":[]}')
    with pytest.raises(ValueError, match="malformed document"):
        _cif_structure_from_record(repeated)


def test_a_mixed_list_is_refused_and_a_bare_block_is_served_as_the_null_row(tmp_path: Path) -> None:
    cs = CIFStructure.from_cif(BONDED)
    with pytest.raises(TypeError, match="must hold only CIF structures"):
        httk.core.save([cs, httk.core.load(BONDED)], tmp_path / "x.cif")
    block = CIFBlock(cs.name, cs.items, cs.header, cs.leading)
    (row,) = StructureEntryProvider({"x": block}).records("structures")
    assert row == next(iter(StructureEntryProvider({"x": cs}).records("structures")))
