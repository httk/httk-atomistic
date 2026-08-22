"""Exact structure-reading regressions for the FINDSYM-processed COD tutorial corpus."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from httk.atomistic._structreading import structreading_golden

_ROOT = Path(__file__).parent
_FIXTURES = _ROOT / "fixtures" / "structreading"
_GOLDEN = _ROOT / "data" / "structreading_golden.json.gz"
_NORMAL_FIXTURES = (
    "1.cif",
    "2.cif",
    "14.cif",
    "16.cif",
    "26.cif",
    "51.cif",
    "70.cif",
    "75.cif",
    "102.cif",
    "110.cif",
    "120.cif",
    "130.cif",
    "142.cif",
    "149.cif",
    "160.cif",
    "168.cif",
    "175.cif",
    "190.cif",
    "194.cif",
    "200.cif",
    "207.cif",
    "214.cif",
    "221.cif",
    "225.cif",
    "228.cif",
)


def _golden() -> dict[str, dict[str, Any]]:
    """Read the committed compressed golden.

    :return: Golden entry by CIF filename.
    """
    with gzip.open(_GOLDEN, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_paths() -> list[Path]:
    """Return real corpus entries while ignoring editor lock files."""
    return sorted(path for path in _FIXTURES.glob("*.cif") if not path.name.startswith("."))


def _assert_golden(filename: str, expected: dict[str, Any], path: Path) -> None:
    """Compare each named regression field with a useful failure message.

    :param filename: CIF filename for the failure message.
    :param expected: Committed canonical interpretation.
    :param path: CIF to read now.
    """
    current = structreading_golden(path)
    for field, golden_value in expected.items():
        if current[field] != golden_value:
            pytest.fail(
                f"{filename}: {field} differs\ngolden={golden_value!r}\ncurrent={current[field]!r}",
                pytrace=False,
            )


def test_structreading_golden_is_substantial() -> None:
    """The full corpus golden must not silently become empty or partial."""
    assert len(_golden()) == 230


@pytest.mark.parametrize("path", [_FIXTURES / name for name in _NORMAL_FIXTURES], ids=lambda path: path.name)
def test_structreading_fixtures(path: Path) -> None:
    """Pin exact current interpretation for the always-available representative subset.

    :param path: In-repository CIF fixture.
    """
    _assert_golden(path.name, _golden()[path.name], path)


@pytest.mark.extended
def test_structreading_legacy_corpus_manifest() -> None:
    """The vendored corpus and committed golden must describe the same files."""
    golden = _golden()
    paths = _fixture_paths()
    assert {path.name for path in paths} == set(golden)


@pytest.mark.extended
@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda path: path.name)
def test_structreading_legacy_corpus(path: Path) -> None:
    """Pin one vendored full-corpus CIF per parallel test case.

    :param path: Vendored legacy tutorial CIF.
    """
    _assert_golden(path.name, _golden()[path.name], path)
