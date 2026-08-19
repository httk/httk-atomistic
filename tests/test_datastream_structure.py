"""Lazy file, stream, and URL structure sources."""

import io
import json
import pickle
import urllib.request
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import FracVector, TextstreamURLView
from httk.core.optimade import IncompleteOptimadeResourceError

from httk.atomistic import (
    Assembly,
    ASUStructure,
    ASUStructureView,
    Cell,
    CollinearSiteMoments,
    DatastreamStructure,
    NumericUnitcellStructureView,
    SettingTransform,
    Spacegroup,
    Species,
    StructureEntryProvider,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
)
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure


class _Response(io.BytesIO):
    headers = Message()


class _CountingResolver(StructureBackend):
    def __init__(self, native: Any) -> None:
        self.native = native
        self.resolve_calls = 0

    @property
    def cell(self) -> Any:
        return self.native.cell

    @property
    def sites(self) -> Any:
        return self.native.sites

    @property
    def species(self) -> Any:
        return self.native.species

    @property
    def species_at_sites(self) -> Any:
        return self.native.species_at_sites

    def resolve(self) -> Any:
        self.resolve_calls += 1
        return self.native

    def unwrap(self) -> Any:
        return self


class _GenericPickleResolver(StructureBackend):
    def __init__(self) -> None:
        self.source = "plain-he"
        self.resolve_calls = 0

    @property
    def cell(self) -> Any:
        return _plain_he().cell

    @property
    def sites(self) -> Any:
        return _plain_he().sites

    @property
    def species(self) -> Any:
        return _plain_he().species

    @property
    def species_at_sites(self) -> Any:
        return _plain_he().species_at_sites

    def resolve(self) -> Any:
        self.resolve_calls += 1
        return _plain_he()

    def unwrap(self) -> Any:
        return self.source


def _plain_he() -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        [[0, 0, 0]],
        [Species("He", ("He",), (1,))],
        ["He"],
    )


def _native_na_asu() -> ASUStructure:
    return ASUStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        225,
        [WyckoffSite("a", FracVector(()), "Na")],
        [Species("Na", ("Na",), (1,))],
    )


def _write_cif(path: Path) -> None:
    path.write_text(
        """data_x
_cell_length_a 2
_cell_length_b 3
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
He1 He 0 0 0 1
""",
        encoding="utf-8",
    )


def test_datastream_delegates_concrete_charge_and_site_moments() -> None:
    native = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [[0, 0, 0]],
        [Species("Si", ("Si",), (1,))],
        ["Si"],
        site_moments=CollinearSiteMoments([2]),
        charge="1/3",
    )
    lazy = object.__new__(DatastreamStructure)
    lazy._parsed = native

    assert lazy.charge == native.charge
    assert lazy.site_moments == native.site_moments
    assert lazy.site_moments is not None


def _mock_http(monkeypatch: pytest.MonkeyPatch, responses: dict[str, str]) -> None:
    def fake_urlopen(url: Any, *, timeout: float | None) -> _Response:
        key = url.full_url if isinstance(url, urllib.request.Request) else url
        if key in responses:
            return _Response(responses[key].encode())
        raise AssertionError(f"unexpected URL: {key!r}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _optimade_fixture(*, span: str | None = None) -> tuple[str, str]:
    schema = httk.core.load_entry_type_definition(
        "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
    )
    properties = {
        remote_name: {"$id": schema.properties[property_name].definition_id}
        for remote_name, property_name in {
            "remote_lattice": "lattice_vectors",
            "remote_fractional": "fractional_site_positions",
            "remote_species": "species",
            "remote_site_species": "species_at_sites",
            "remote_dimensions": "dimension_types",
            "remote_elements": "elements",
            "remote_ratios": "elements_ratios",
            "remote_immutable": "immutable_id",
            "remote_features": "structure_features",
        }.items()
    }
    if span is not None:
        properties["remote_span"] = {"$id": schema.properties["site_coordinate_span"].definition_id}
    entry = {
        "data": {
            "id": "fixture-1",
            "type": "structures",
            "attributes": {
                "remote_lattice": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
                "remote_fractional": [[0, 0, 0]],
                "remote_species": [{"name": "He", "chemical_symbols": ["He"], "concentration": [1]}],
                "remote_site_species": ["He"],
                "remote_dimensions": [1, 1, 1],
                "remote_elements": ["He"],
                "remote_ratios": [1],
                "remote_immutable": "fixture-1",
                "remote_features": [],
                **({"remote_span": span} if span is not None else {}),
            },
        }
    }
    return json.dumps(entry), json.dumps({"data": {"properties": properties}})


def test_local_source_is_lazy_and_memoized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    real_load = httk.core.load
    calls = 0

    def counted_load(filename: str) -> Any:
        nonlocal calls
        calls += 1
        return real_load(filename)

    monkeypatch.setattr(httk.core, "load", counted_load)
    view = UnitcellStructureView(str(path))
    assert calls == 0
    _ = view.cell
    assert calls == 1
    _ = view.cell
    _ = view.sites
    assert calls == 1


def test_backend_configuration_and_unwrap(tmp_path: Path) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    stream = io.StringIO(path.read_text(encoding="utf-8"))
    sources = (
        str(path),
        path.as_uri(),
        httk.core.DatastreamURL(path.as_uri()),
        urllib.request.Request(path.as_uri()),
        stream,
    )
    for source in sources:
        backend = DatastreamStructure(source, **({"name": "x.cif"} if source is stream else {}))
        assert backend.unwrap() is source


def test_local_configuration_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.cif"):
        UnitcellStructureView(str(tmp_path / "missing.cif"))
    with pytest.raises(TypeError, match="Cannot represent"):
        UnitcellStructureView(str(tmp_path / "x.unknown"))


def test_bare_network_string_explains_datastream_consent() -> None:
    with pytest.raises(PermissionError) as error:
        UnitcellStructureView("https://example.test/x.cif")
    message = str(error.value)
    assert "fetch(url)" in message
    assert "DatastreamURL" in message


def test_optimade_request_is_declined_to_preserve_headers() -> None:
    request = urllib.request.Request(
        "https://example.test/v1/structures/fixture-1",
        headers={"Authorization": "Bearer secret"},
    )
    assert DatastreamStructure._backend_adopt(request) is None


def test_optimade_request_with_loader_collision_is_actionable() -> None:
    request = urllib.request.Request("https://example.test/v1/structures/fixture-1.cif")
    with pytest.raises(ValueError, match=r"httk\.core\.fetch.*name="):
        DatastreamStructure(request)


def test_optimade_request_loader_collision_accepts_explicit_name() -> None:
    request = urllib.request.Request("https://example.test/v1/structures/fixture-1.cif")
    backend = DatastreamStructure(request, name="fixture-1.cif")
    assert backend is not None


def test_bytes_are_rejected_by_structure_backend() -> None:
    with pytest.raises(TypeError, match=r"Cannot represent <class 'bytes'> as StructureBackend"):
        list(StructureEntryProvider({"x": b""}).records("structures"))


def test_url_stream_view_without_loader_is_declined() -> None:
    source = TextstreamURLView("https://example.test/v1/structures/fixture-1")
    assert DatastreamStructure._backend_adopt(source) is None


def test_pathlike_colon_name_is_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("sample:1.cif")
    _write_cif(path)
    assert UnitcellStructureView(path).cell.basis.to_fractions_approx()[0][0] == 2


def test_names_cover_poscar_and_compressed_cif(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_text("", encoding="utf-8")
    compressed = tmp_path / "x.cif.gz"
    compressed.write_bytes(b"")
    assert DatastreamStructure(poscar)._name == str(poscar)
    assert DatastreamStructure(compressed)._name == str(compressed)


def test_named_text_stream_claims_loader() -> None:
    stream = io.StringIO("data_x\n")
    backend = DatastreamStructure(stream, name="x.cif")
    assert backend.unwrap() is stream


def test_views_adopt_loaded_cif_asu(tmp_path: Path) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    loaded = httk.core.load(str(path))
    view = ASUStructureView(str(path))
    assert view.spacegroup == loaded.spacegroup
    assert view.wyckoff_sites == loaded.wyckoff_sites
    assert UnitcellStructureView(str(path)).cell == UnitcellStructureView(loaded).cell


def test_asu_view_path_construction_and_unwrap_do_no_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    calls = 0
    real_load = httk.core.load

    def counted_load(filename: str) -> Any:
        nonlocal calls
        calls += 1
        return real_load(filename)

    monkeypatch.setattr(httk.core, "load", counted_load)
    view = ASUStructureView(str(path))

    assert calls == 0
    assert view.unwrap() == str(path)
    assert view._resolved_asu is None
    assert "_cell" not in view.__dict__


def test_asu_view_plain_unitcell_defers_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    module = __import__("httk.atomistic.models.structure.asu_view", fromlist=["recognize_asu"])
    real_recognize = module.recognize_asu
    recognition_calls = 0

    def counted_recognize(*args: Any, **kwargs: Any) -> ASUStructure:
        nonlocal recognition_calls
        recognition_calls += 1
        return real_recognize(*args, **kwargs)

    monkeypatch.setattr(module, "recognize_asu", counted_recognize)
    source = _plain_he()
    view = ASUStructureView(source, setting=Spacegroup.standard(1))

    assert recognition_calls == 0
    assert view.unwrap() is source
    assert view._resolved_asu is None
    assert view.immutable_id is None
    assert recognition_calls == 1
    assert view.spacegroup.it_number == 1


def test_asu_view_resolves_once_and_record_projection_uses_the_resolved_asu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _CountingResolver(_plain_he())
    view = ASUStructureView(source, setting=Spacegroup.standard(1))
    module = __import__("httk.atomistic.models.structure.asu_view", fromlist=["recognize_asu"])
    real_recognize = module.recognize_asu
    recognition_calls = 0

    def counted_recognize(*args: Any, **kwargs: Any) -> ASUStructure:
        nonlocal recognition_calls
        recognition_calls += 1
        return real_recognize(*args, **kwargs)

    monkeypatch.setattr(module, "recognize_asu", counted_recognize)
    assert view.wyckoff_sites == view.wyckoff_sites
    record = next(iter(StructureEntryProvider({"he": view}).records("structures")))

    assert source.resolve_calls == 1
    assert recognition_calls == 1
    assert record["nsites"] == 1
    assert record["species_at_sites"] == ["He"]


def test_asu_recognition_options_survive_unitcell_view_nesting() -> None:
    source = _CountingResolver(_plain_he())
    inner = ASUStructureView(source, setting=Spacegroup.standard(1))
    later = UnitcellStructureView(inner)
    outer = ASUStructureView(later)

    assert outer.spacegroup.it_number == 1
    assert source.resolve_calls == 1


def test_asu_nested_option_families_replace_each_other_without_work() -> None:
    standard = Spacegroup.standard(1)
    source = _CountingResolver(_plain_he())
    inner = ASUStructureView(source, standard=standard, transform=SettingTransform.identity())
    nested = UnitcellStructureView(inner)
    outer = ASUStructureView(nested, setting=standard)

    assert outer._setting is standard
    assert outer._standard is None
    assert outer._recognition_transform is None
    assert source.resolve_calls == 0
    assert outer.spacegroup.it_number == 1
    assert source.resolve_calls == 1

    source = _CountingResolver(_plain_he())
    inner = ASUStructureView(source, setting=standard)
    nested = UnitcellStructureView(inner)
    outer = ASUStructureView(nested, standard=standard, transform=SettingTransform.identity())

    assert outer._setting is None
    assert outer._standard is standard
    assert outer._recognition_transform == SettingTransform.identity()
    assert source.resolve_calls == 0
    assert outer.spacegroup.it_number == 1
    assert source.resolve_calls == 1


def test_asu_view_lazy_metadata_override_is_preserved() -> None:
    source = _CountingResolver(_plain_he())
    view = ASUStructureView(source, setting=Spacegroup.standard(1))
    resolved = view.unview()
    attached = ASUStructureView(view, immutable_id="attached")

    assert resolved.immutable_id is None
    assert attached.immutable_id == "attached"


def test_asu_view_failed_resolution_publishes_no_partial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _CountingResolver(_plain_he())
    view = ASUStructureView(source, setting=Spacegroup.standard(1))
    module = __import__("httk.atomistic.models.structure.asu_view", fromlist=["recognize_asu"])

    def fail(*args: Any, **kwargs: Any) -> ASUStructure:
        raise ValueError("recognition failed")

    monkeypatch.setattr(module, "recognize_asu", fail)
    with pytest.raises(ValueError, match="recognition failed"):
        _ = view.wyckoff_sites
    assert view._resolved_asu is None
    assert "_cell" not in view.__dict__


def test_asu_view_validates_expansion_before_publishing() -> None:
    invalid = ASUStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        225,
        [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("a", FracVector(()), "Na")],
        [Species("Na", ("Na",), (1,))],
    )
    view = ASUStructureView(_CountingResolver(invalid))

    with pytest.raises(ValueError, match="duplicates an earlier site's orbit"):
        _ = view.cell
    assert view._resolved_asu is None
    assert "_cell" not in view.__dict__


def test_asu_view_resolved_native_asu_skips_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    path_source = _CountingResolver(httk.core.load_source(io.StringIO(_cif_text()), "x.cif"))
    view = ASUStructureView(path_source)
    monkeypatch.setattr(
        "httk.atomistic.models.structure.asu_view.recognize_asu",
        lambda *args, **kwargs: pytest.fail("native ASU must not be recognized"),
    )

    assert view.spacegroup.it_number == 1
    assert path_source.resolve_calls == 1


def test_asu_view_native_asu_is_pending_until_first_access() -> None:
    invalid = ASUStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        225,
        [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("a", FracVector(()), "Na")],
        [Species("Na", ("Na",), (1,))],
    )

    view = ASUStructureView(invalid)

    assert view._resolved_asu is None
    assert "_expansion" not in invalid.__dict__
    with pytest.raises(ValueError, match="duplicates an earlier site's orbit"):
        _ = view.cell
    assert view._resolved_asu is None
    assert "_cell" not in view.__dict__


def test_asu_view_plain_unitcell_pickle_stays_unresolved() -> None:
    view = ASUStructureView(_plain_he(), setting=Spacegroup.standard(1))

    restored = pickle.loads(pickle.dumps(view))

    assert restored._resolved_asu is None
    assert "_cell" not in restored.__dict__
    assert restored.spacegroup.it_number == 1
    assert restored._resolved_asu is not None


def test_asu_view_native_asu_pickle_retains_source_and_stays_lazy() -> None:
    native = _native_na_asu()
    view = ASUStructureView(native)

    restored = pickle.loads(pickle.dumps(view))

    assert restored._resolved_asu is None
    assert isinstance(restored.unwrap(), ASUStructure)
    assert restored.unwrap() is restored._source_backend
    assert "_expansion" not in restored._source_backend.__dict__
    assert restored.spacegroup.it_number == 225
    assert restored.unview() is restored._source_backend


def test_asu_view_rejects_invalid_recognition_options_without_backend_work() -> None:
    source = _CountingResolver(_plain_he())
    with pytest.raises(TypeError, match="either 'setting' or 'standard'/'transform'"):
        ASUStructureView(source, setting=Spacegroup.standard(1), standard=Spacegroup.standard(1))
    assert source.resolve_calls == 0

    source = _CountingResolver(_plain_he())
    with pytest.raises(TypeError, match="needs both 'standard' and 'transform'"):
        ASUStructureView(source, standard=Spacegroup.standard(1))
    assert source.resolve_calls == 0

    source = _CountingResolver(_plain_he())
    with pytest.raises(TypeError, match="needs both 'standard' and 'transform'"):
        ASUStructureView(source, transform=SettingTransform.identity())
    assert source.resolve_calls == 0

    source = _CountingResolver(_plain_he())
    with pytest.raises(ValueError, match="'standard' must be an IT standard setting"):
        ASUStructureView(source, standard=Spacegroup.from_setting("15:c1"), transform=SettingTransform.identity())
    assert source.resolve_calls == 0


def test_asu_view_unresolved_replayable_pickle_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    view = ASUStructureView(str(path))
    restored = pickle.loads(pickle.dumps(view))

    assert restored._resolved_asu is None
    assert restored.unwrap() == str(path)
    assert restored.spacegroup.it_number == 1


def test_asu_view_unresolved_pickle_accepts_generic_resolver_backend() -> None:
    source = _GenericPickleResolver()
    view = ASUStructureView(source, setting=Spacegroup.standard(1))

    restored = pickle.loads(pickle.dumps(view))

    assert isinstance(restored._source_backend, _GenericPickleResolver)
    assert restored._resolved_asu is None
    assert source.resolve_calls == 0
    assert restored.unwrap() == "plain-he"
    assert restored.spacegroup.it_number == 1
    assert restored._source_backend.resolve_calls == 1


def test_asu_view_resolved_pickle_retains_source_backend(tmp_path: Path) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    view = ASUStructureView(str(path))
    source_backend = view._source_backend
    expected = view.unview()
    path.write_text("not a CIF", encoding="utf-8")
    restored = pickle.loads(pickle.dumps(view))

    assert type(restored._source_backend) is type(source_backend)
    assert restored.unwrap() == str(path)
    assert restored.unview() == expected
    assert restored.unview() is restored._resolved_asu


def test_asu_view_unresolved_stream_pickle_roundtrip() -> None:
    view = ASUStructureView(io.StringIO(_cif_text()), name="x.cif")
    restored = pickle.loads(pickle.dumps(view))

    assert restored._resolved_asu is None
    assert restored.unwrap().tell() == 0
    assert restored.spacegroup.it_number == 1


def test_asu_view_unview_materializes_the_standalone_asu(tmp_path: Path) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    view = ASUStructureView(str(path))

    result = view.unview()

    assert type(result) is ASUStructure
    assert result is view._resolved_asu


def test_supercell_uses_file_url_datastream(tmp_path: Path) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    result = build_supercell(httk.core.DatastreamURL(path.as_uri()), 2).structure
    assert len(result.sites) == 8


def test_supercell_uses_file_format_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://example.test/x.cif"
    _mock_http(monkeypatch, {url: _cif_text()})
    result = build_supercell(httk.core.DatastreamURL(url), 2).structure
    assert len(result.sites) == 8


def test_supercell_uses_optimade_entry_url(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture()
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    result = build_supercell(httk.core.DatastreamURL(entry_url), 2).structure
    assert len(result.sites) == 8


def test_entry_provider_resolves_datastream_url(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture()
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    (record,) = list(StructureEntryProvider({"x": httk.core.DatastreamURL(entry_url)}).records("structures"))
    assert record["__id"] == "x"
    assert record["lattice_vectors"] == [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]


def test_deferred_span_guard_runs_on_first_access(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture(span="asymmetric_unit")
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    view = UnitcellStructureView(httk.core.DatastreamURL(entry_url))
    with pytest.raises(IncompleteOptimadeResourceError, match="site_coordinate_span"):
        _ = view.cell


def test_numeric_deferred_species_runs_span_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture(span="asymmetric_unit")
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    view = NumericUnitcellStructureView(httk.core.DatastreamURL(entry_url))
    with pytest.raises(IncompleteOptimadeResourceError, match="site_coordinate_span"):
        _ = view.species


def test_deferred_metadata_conflict_precedes_span_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture(span="asymmetric_unit")
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    view = UnitcellStructureView(httk.core.DatastreamURL(entry_url), immutable_id="other")
    with pytest.raises(ValueError, match="immutable_id"):
        _ = view.cell


def test_deferred_view_of_view_inherits_metadata_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture()
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    inner = UnitcellStructureView(httk.core.DatastreamURL(entry_url), immutable_id="local")
    outer = UnitcellStructureView(inner, last_modified=datetime(2024, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="immutable_id"):
        _ = outer.immutable_id


def test_deferred_metadata_inherits_and_conflicts_at_first_access(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url = "https://example.test/v1/structures/fixture-1"
    info_url = "https://example.test/v1/info/structures"
    entry, info = _optimade_fixture()
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    view = UnitcellStructureView(httk.core.DatastreamURL(entry_url))
    assert view.immutable_id == "fixture-1"

    _mock_http(monkeypatch, {entry_url: entry, info_url: info})
    conflicting = UnitcellStructureView(httk.core.DatastreamURL(entry_url), immutable_id="other")
    with pytest.raises(ValueError, match="immutable_id"):
        _ = conflicting.cell


def test_deferred_cif_asu_assemblies_match_eager_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.cif"
    _write_cif(path)
    loaded = httk.core.load(str(path))
    native = type(loaded)(
        loaded.cell,
        loaded.spacegroup,
        loaded.wyckoff_sites,
        loaded.species,
        loaded.transform,
        loaded.coordinate_precision,
        assemblies=(Assembly(((0,),), (1,)),),
    )
    monkeypatch.setattr(httk.core, "load", lambda _: native)
    lazy = UnitcellStructureView(str(path))
    eager = UnitcellStructureView(native)
    assert lazy.assemblies == eager.assemblies
    assert (
        build_supercell(UnitcellStructureView(str(path)), 2).structure.assemblies
        == build_supercell(native, 2).structure.assemblies
    )


def test_file_url_optimade_datastream_uses_fixture_files(tmp_path: Path) -> None:
    entry, info = _optimade_fixture()
    entry_path = tmp_path / "v1" / "structures" / "fixture-1"
    info_path = tmp_path / "v1" / "info" / "structures"
    entry_path.parent.mkdir(parents=True)
    info_path.parent.mkdir(parents=True)
    entry_path.write_text(entry, encoding="utf-8")
    info_path.write_text(info, encoding="utf-8")
    view = UnitcellStructureView(httk.core.DatastreamURL(entry_path.as_uri()))
    assert view.immutable_id == "fixture-1"


def _cif_text() -> str:
    return """data_x
_cell_length_a 2
_cell_length_b 3
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
He1 He 0 0 0 1
"""
