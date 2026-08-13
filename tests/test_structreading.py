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
_CORPUS = Path("/home/rar/Documents/containers/devel/agents/httk2/old/httk/Tutorial/tutorial_data/all_spacegroups/cifs")


def _golden() -> dict[str, dict[str, Any]]:
    """Read the committed compressed golden.

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


def test_structreading_golden_is_substantial() -> None:
    """The full corpus golden must not silently become empty or partial."""
    assert len(_golden()) > 200


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.cif")), ids=lambda path: path.name)
def test_structreading_fixtures(path: Path) -> None:
    """Pin exact current interpretation for the always-available representative subset.

    :param path: In-repository CIF fixture.
    """
    _assert_golden(path.name, _golden()[path.name], path)


@pytest.mark.extended
@pytest.mark.skipif(not _CORPUS.is_dir(), reason="legacy httk tutorial CIF corpus is unavailable")
def test_structreading_legacy_corpus_manifest() -> None:
    """The optional corpus and the committed golden must describe the same files."""
    golden = _golden()
    paths = sorted(_CORPUS.glob("*.cif"))
    assert {path.name for path in paths} == set(golden)


@pytest.mark.extended
@pytest.mark.skipif(not _CORPUS.is_dir(), reason="legacy httk tutorial CIF corpus is unavailable")
@pytest.mark.parametrize("path", sorted(_CORPUS.glob("*.cif")), ids=lambda path: path.name)
def test_structreading_legacy_corpus(path: Path) -> None:
    """Pin one optional full-corpus CIF per parallel test case.

    :param path: Legacy tutorial CIF outside the repository.
    """
    _assert_golden(path.name, _golden()[path.name], path)
