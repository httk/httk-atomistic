import io

import pytest
from httk.core import FracVector

from httk.atomistic import ASUStructure, WyckoffSite, band_path
from httk.atomistic.integrations.vasp.io import write_line_mode_kpoints

SPECIES = [
    {"name": "A", "chemical_symbols": ["Na"], "concentration": [1.0]},
    {"name": "B", "chemical_symbols": ["Cl"], "concentration": [1.0]},
]


def _structure(group, wa, wb):
    return ASUStructure(
        [["5.64", 0, 0], [0, "5.64", 0], [0, 0, "5.64"]],
        group,
        [WyckoffSite(wa, FracVector(()), "A"), WyckoffSite(wb, FracVector(()), "B")],
        SPECIES,
    )


def _write(result, **kw):
    out = io.StringIO()
    write_line_mode_kpoints(out, result.points, result.path, **kw)
    return out.getvalue()


def _row(x, label):
    return "".join(f"{v:16.12f}" for v in x) + f" ! {label}"


def test_rocksalt_golden_and_file(tmp_path):
    result = band_path(_structure(225, "a", "b"))
    text = _write(result, comment="c")
    segs = [
        (("GAMMA", (0, 0, 0)), ("X", (0.5, 0, 0.5))),
        (("X", (0.5, 0, 0.5)), ("U", (0.625, 0.25, 0.625))),
    ]
    head = "c\n40\nLine-mode\nReciprocal\n" + "\n\n".join("\n".join(_row(p, n) for n, p in seg) for seg in segs)
    assert text.startswith(head + "\n\n")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert text.count("\n\n") == len(result.path) - 1
    assert "  0.000000000000  0.000000000000  0.000000000000 ! GAMMA\n" in text
    target = tmp_path / "KPOINTS"
    write_line_mode_kpoints(target, result.points, result.path, comment="c")
    assert target.read_text() == text


def test_negative_and_primed_and_roundtrip():
    result = band_path(_structure(216, "a", "c"), time_reversal=False)
    text = _write(result)
    assert any("'" in line.split("! ")[1] for line in text.splitlines() if "!" in line)
    assert any(" -0." in line for line in text.splitlines())
    for line in text.splitlines():
        if "!" in line:
            coords, label = line.split(" ! ")
            got = [float(c) for c in coords.split()]
            want = [float(f) for f in result.points[label].to_fractions()]
            assert all(abs(g - w) < 1e-12 for g, w in zip(got, want, strict=True))


P = {"A": (0, 0, 0), "B": (0, 0, "1/2"), "C": (0, 0)}


@pytest.mark.parametrize(
    "path,kw",
    [
        ([], {}),
        ([("A",)], {}),
        (["AB"[:1]], {}),
        ([("A", "Z")], {}),
        ([("A", "C")], {}),
        ([("A", "B")], {"comment": "a\nb"}),
        ([("A", "B")], {"comment": "a\rb"}),
        ([("A", "B")], {"divisions": 1}),
        ([("A", "B")], {"divisions": True}),
        ([("A", "B")], {"divisions": 2.0}),
    ],
)
def test_invalid(path, kw):
    with pytest.raises(ValueError):
        write_line_mode_kpoints(io.StringIO(), P, path, **kw)


@pytest.mark.parametrize("label", ["", "a!b", "a\nb"])
def test_invalid_label(label):
    with pytest.raises(ValueError):
        write_line_mode_kpoints(io.StringIO(), {label: (0, 0, 0), "B": (0, 0, 0)}, [(label, "B")])


@pytest.mark.parametrize(
    "points,path",
    [
        ({"A": [[0], [0], [0]], "B": (0, 0, 0)}, [("A", "B")]),
        ({"A": (0, 0, 0), "B": (0, 0, 0)}, [5]),
    ],
)
def test_malformed_shapes_raise_value_error(points, path):
    with pytest.raises(ValueError):
        write_line_mode_kpoints(io.StringIO(), points, path)
