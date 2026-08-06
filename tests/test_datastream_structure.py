"""Lazy file, stream, and URL structure sources."""

import io
import json
import urllib.request
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import TextstreamURLView
from httk.core.optimade import IncompleteOptimadeResourceError

from httk.atomistic import (
    Assembly,
    ASUStructureView,
    Cell,
    CollinearSiteMoments,
    DatastreamStructure,
    NumericUnitcellStructureView,
    Species,
    StructureEntryProvider,
    UnitcellStructureView,
    build_supercell,
)
from httk.atomistic.models.structure.unitcell import UnitcellStructure

pytest.importorskip("httk.io")


class _Response(io.BytesIO):
    headers = Message()


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
    assert DatastreamStructure(request) is None


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
    assert DatastreamStructure(source) is None


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
