"""Exact native LAMMPS dump parser checks."""

import gzip
from fractions import Fraction
from pathlib import Path

import pytest

from httk.atomistic.integrations.lammps.io import LammpsDumpFile, read_lammps_dump


def _frame(
    *,
    step: int = 0,
    box: str = "ITEM: BOX BOUNDS pp ff ss\n-1 3\n2 6\n0 8",
    columns: str = "id type xs ys zs ix iy iz vx vy vz fx fy fz q",
    atoms: str = "2 2 1/2 1/4 3/4 -1 0 1 1 2 3 4 5 6 99\n1 1 0 0 0 0 0 0 -1 -2 -3 -4 -5 -6 98",
    prefix: str = "",
) -> str:
    return f"{prefix}ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n{box}\nITEM: ATOMS {columns}\n{atoms}\n"


def test_orthogonal_scaled_fields_are_exact_sorted_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "atoms.dump"
    source.write_text(_frame(prefix="ITEM: UNITS\nmetal\nITEM: TIME\n1.25\n"), encoding="utf-8")
    frame = next(LammpsDumpFile(source).frames())

    assert frame.step == 0
    assert frame.segment == 0
    assert frame.time == Fraction(5, 4)
    assert frame.units == "metal"
    assert frame.cell == (
        (Fraction(4), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(4), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(8)),
    )
    assert frame.origin == (Fraction(-1), Fraction(2), Fraction(0))
    assert frame.periodicity == (True, False, False)
    assert frame.atom_ids == (1, 2)
    assert frame.atom_types == (1, 2)
    assert frame.positions == ((Fraction(-1), Fraction(2), Fraction(0)), (Fraction(1), Fraction(3), Fraction(6)))
    assert frame.unwrapped_positions == (
        (Fraction(-1), Fraction(2), Fraction(0)),
        (Fraction(-3), Fraction(3), Fraction(14)),
    )
    assert frame.velocities == ((Fraction(-1), Fraction(-2), Fraction(-3)), (Fraction(1), Fraction(2), Fraction(3)))
    assert frame.forces == ((Fraction(-4), Fraction(-5), Fraction(-6)), (Fraction(4), Fraction(5), Fraction(6)))
    with pytest.raises(AttributeError):
        frame.step = 1  # type: ignore[misc]


def test_restricted_triclinic_recovers_bounds_with_negative_tilts(tmp_path: Path) -> None:
    source = tmp_path / "tilted.dump"
    source.write_text(
        _frame(
            box="ITEM: BOX BOUNDS xy xz yz pp pp ff\n-5 10 -2\n-4 7 3\n1 9 -1",
            columns="id type xs ys zs",
            atoms="1 1 0 0 0\n2 1 1 1 1",
        ),
        encoding="utf-8",
    )
    frame = next(LammpsDumpFile(source).frames())
    assert frame.origin == (Fraction(-3), Fraction(-3), Fraction(1))
    assert frame.cell == (
        (Fraction(10), Fraction(0), Fraction(0)),
        (Fraction(-2), Fraction(10), Fraction(0)),
        (Fraction(3), Fraction(-1), Fraction(8)),
    )
    assert frame.positions == ((Fraction(-3), Fraction(-3), Fraction(1)), (Fraction(8), Fraction(6), Fraction(9)))


def test_general_triclinic_origin_and_optional_flags(tmp_path: Path) -> None:
    source = tmp_path / "general.dump"
    source.write_text(
        _frame(
            box="ITEM: BOX BOUNDS abc origin\n2 0 0 -1\n1 3 0 4\n0 1 5 -2",
            columns="id type xsu ysu zsu",
            atoms="1 1 0 0 0\n2 1 3/2 -1 2",
        ),
        encoding="utf-8",
    )
    frame = next(LammpsDumpFile(source).frames())
    assert frame.periodicity is None
    assert frame.origin == (Fraction(-1), Fraction(4), Fraction(-2))
    assert frame.positions[1] == (Fraction(1), Fraction(3), Fraction(8))
    assert frame.unwrapped_positions == frame.positions

    source.write_text(
        _frame(
            box="ITEM: BOX BOUNDS abc origin pp ff ss\n2 0 0 -1\n1 3 0 4\n0 1 5 -2",
            columns="id type xu yu zu",
            atoms="1 1 0 0 0\n2 1 1 2 3",
        ),
        encoding="utf-8",
    )
    assert next(LammpsDumpFile(source).frames()).periodicity == (True, False, False)


def test_coordinate_selection_and_segment_identity(tmp_path: Path) -> None:
    source = tmp_path / "segments.dump"
    both = _frame(columns="id type x y z xu yu zu", atoms="1 1 1 2 3 11 12 13\n2 2 4 5 6 14 15 16")
    source.write_text(
        both + _frame(step=0, columns="id type x y z xu yu zu", atoms="3 1 0 0 0 10 11 12\n4 2 1 1 1 13 14 15"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple representations"):
        tuple(LammpsDumpFile(source).frames())
    frames = tuple(LammpsDumpFile(source, coordinates="xu").frames())
    assert [frame.segment for frame in frames] == [0, 1]
    assert frames[0].positions[0] == (Fraction(11), Fraction(12), Fraction(13))


def test_repeated_compressed_streaming_is_fresh(tmp_path: Path) -> None:
    source = tmp_path / "atoms.dump.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write(_frame() + _frame(step=1))
    dump = LammpsDumpFile(source)
    assert [frame.step for frame in dump.frames()] == [0, 1]
    assert [frame.step for frame in dump.frames()] == [0, 1]


def test_reader_returns_lazy_path_and_options_payload(tmp_path: Path) -> None:
    source = tmp_path / "lazy.dump"
    source.write_text("not scanned yet", encoding="utf-8")
    payload = read_lammps_dump(source, species={1: "Si"}, units="metal")
    assert payload == {
        "format": "lammps-dump",
        "source": source,
        "options": {"species": {1: "Si"}, "units": "metal"},
    }
    with pytest.raises(FileNotFoundError):
        read_lammps_dump(tmp_path / "missing.dump")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_frame(columns="id type x y", atoms="1 1 0 0\n2 1 0 0"), "incomplete atom column triplet"),
        (_frame(columns="id type x y z ix iy", atoms="1 1 0 0 0 0 0\n2 1 0 0 0 0 0"), "incomplete atom column triplet"),
        (_frame(columns="id type x y z vx vy", atoms="1 1 0 0 0 0 0\n2 1 0 0 0 0 0"), "incomplete atom column triplet"),
        (_frame(columns="id type x y z fx fy", atoms="1 1 0 0 0 0 0\n2 1 0 0 0 0 0"), "incomplete atom column triplet"),
        (_frame(columns="id type x y z", atoms="1 1 0 0 0\n1 1 0 0 0"), "IDs must be unique"),
        (_frame(columns="id type x y z", atoms="1 1 nan 0 0\n2 1 0 0 0"), "invalid finite numeric token"),
        (
            _frame(
                box="ITEM: BOX BOUNDS pp pp pp\n0 1\n0 0\n0 1", columns="id type x y z", atoms="1 1 0 0 0\n2 1 0 0 0"
            ),
            "box lengths must be positive",
        ),
        (_frame().rsplit("\n", 3)[0] + "\n", "truncated LAMMPS dump frame"),
    ],
)
def test_malformed_frames_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    source = tmp_path / "bad.dump"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        tuple(LammpsDumpFile(source).frames())


def test_identity_and_type_changes_are_segment_local(tmp_path: Path) -> None:
    source = tmp_path / "identity.dump"
    changed = _frame(step=1, columns="id type x y z", atoms="1 2 0 0 0\n2 2 0 0 0")
    source.write_text(_frame(columns="id type x y z", atoms="1 1 0 0 0\n2 2 0 0 0") + changed, encoding="utf-8")
    with pytest.raises(ValueError, match="IDs or types change"):
        tuple(LammpsDumpFile(source).frames())

    source.write_text(
        _frame(columns="id type x y z", atoms="1 1 0 0 0\n2 2 0 0 0")
        + _frame(step=0, columns="id type x y z", atoms="3 1 0 0 0\n4 2 0 0 0"),
        encoding="utf-8",
    )
    assert [frame.segment for frame in LammpsDumpFile(source).frames()] == [0, 1]


def test_coordinate_representation_is_stable_within_segment(tmp_path: Path) -> None:
    source = tmp_path / 'coordinates.dump'
    source.write_text(
        _frame(step=0, columns='id type x y z ix iy iz', atoms='1 1 1 0 0 1 0 0\n2 1 0 0 0 0 0 0')
        + _frame(step=1, columns='id type xu yu zu', atoms='1 1 5 0 0\n2 1 0 0 0'), encoding='utf-8'
    )
    frames = LammpsDumpFile(source).frames()
    next(frames)
    with pytest.raises(ValueError, match='coordinate representation'):
        next(frames)
