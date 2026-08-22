"""Exact structure-reading regressions for the disorder CIF corpus."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from httk.core import load, save

from httk.atomistic._structreading import structreading_golden

_ROOT = Path(__file__).parent
_FIXTURES = _ROOT / "fixtures" / "disorder"
_GOLDEN = _ROOT / "data" / "disorder_structreading_golden.json.gz"


def _golden() -> dict[str, dict[str, Any]]:
    """Read the committed compressed disorder golden.

    :return: Golden entry by CIF filename.
    """
    with gzip.open(_GOLDEN, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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


def test_disorder_structreading_golden_is_substantial() -> None:
    """The disorder corpus golden must cover the planned 33-file corpus."""
    assert len(_golden()) == 33


def test_disorder_structreading_corpus_manifest() -> None:
    """The vendored disorder CIFs and committed golden must describe the same files."""
    golden = _golden()
    paths = sorted(_FIXTURES.glob("*.cif"))
    assert {path.name for path in paths} == set(golden)


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.cif")), ids=lambda path: path.name)
def test_disorder_structreading_fixtures(path: Path) -> None:
    """Pin one disorder CIF's current interpretation per parallel test case.

    :param path: In-repository disorder CIF fixture.
    """
    _assert_golden(path.name, _golden()[path.name], path)


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.cif")), ids=lambda path: path.name)
def test_disorder_cif_roundtrip(path: Path, tmp_path: Path) -> None:
    """Preserve every disorder fixture's species and exact ASU orbit declaration through CIF."""
    source = load(path, repair=True)
    destination = tmp_path / path.name

    save(source, destination)
    restored = load(destination)

    assert restored.species == source.species
    assert restored.spacegroup == source.spacegroup
    assert restored.cell == source.cell
    assert [(site.wyckoff, site.species, site.free_params) for site in restored.wyckoff_sites] == [
        (site.wyckoff, site.species, site.free_params) for site in source.wyckoff_sites
    ]
