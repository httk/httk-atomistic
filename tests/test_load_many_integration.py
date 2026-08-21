"""Cross-repository process-pool loading coverage for CIF structures."""

from pathlib import Path

from httk.core import load, load_many

_FIXTURES = Path(__file__).parent / "fixtures" / "structreading"
_SOURCES = tuple(str(_FIXTURES / name) for name in ("1.cif", "2.cif", "14.cif", "17.cif", "26.cif", "51.cif"))


def test_load_many_cif_results_match_serial_load() -> None:
    serial = [load(source) for source in _SOURCES]
    parallel = list(load_many(_SOURCES, processes=2, errors="raise"))

    assert [source for source, _result in parallel] == list(_SOURCES)
    assert [result for _source, result in parallel] == serial
