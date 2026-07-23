"""
Tests for the numpy-backed numeric presentation layer: NumericCell/NumericSites/NumericStructure,
the *NumericView classes, and the ``.numeric()`` convenience methods.
"""

import fractions

import pytest

numpy = pytest.importorskip("numpy")

import httk.core.vectors as vectors_pkg  # noqa: E402
from httk.core import unwrap  # noqa: E402

from httk.atomistic import (  # noqa: E402
    Cell,
    CellNumericView,
    CellParams,
    NumericCell,
    NumericSites,
    NumericStructure,
    Sites,
    SitesNumericView,
    Structure,
    StructureNumericView,
)

F = fractions.Fraction


def _hexagonal_structure() -> Structure:
    cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis)
    return Structure(
        cell=cell,
        sites=[[F(0), F(0), F(0)], [F(1, 3), F(1, 3), F(0)]],
        species=[{"name": "Mg", "chemical_symbols": ["Mg"], "concentration": [1.0]}],
        species_at_sites=["Mg", "Mg"],
    )


# ------------------------------------------------------------------ end-to-end numeric values


def test_numeric_structure_values_are_plain_numpy() -> None:
    structure = _hexagonal_structure()
    numeric = structure.numeric()
    assert isinstance(numeric, NumericStructure)

    exact_cell = structure.cell
    # Vectors are exactly-typed float64 ndarrays (not a subclass) matching the exact to_floats().
    for got, want in [
        (numeric.cell.basis, exact_cell.basis.to_floats()),
        (numeric.cell.unscaled_basis, exact_cell.unscaled_basis.to_floats()),
        (numeric.cell.metric(), exact_cell.metric().to_floats()),
        (numeric.cartesian_sites(), structure.cartesian_sites().to_floats()),
        (numeric.sites.reduced_coords, structure.sites.reduced_coords.to_floats()),
    ]:
        assert type(got) is numpy.ndarray
        assert got.dtype == numpy.float64
        assert got.tolist() == want

    # lengths are a (3,) float64 ndarray; angles a (3,) float64 ndarray in degrees.
    assert type(numeric.cell.lengths) is numpy.ndarray
    assert numeric.cell.lengths.tolist() == [length.to_float() for length in exact_cell.lengths]
    assert type(numeric.cell.angles) is numpy.ndarray
    assert numeric.cell.angles.tolist() == pytest.approx([90.0, 90.0, 120.0])

    # volume/scale are plain floats.
    assert type(numeric.cell.volume) is float
    assert numeric.cell.volume == pytest.approx(exact_cell.volume.to_float())
    assert type(numeric.cell.scale) is float

    # species pass through as the identical objects.
    assert numeric.species is structure.species
    assert numeric.species_at_sites == structure.species_at_sites


def test_numeric_sites_is_iterable_and_indexable() -> None:
    numeric = _hexagonal_structure().numeric().sites
    assert isinstance(numeric, NumericSites)
    assert len(numeric) == 2
    rows = list(numeric)
    assert all(type(row) is numpy.ndarray and row.shape == (3,) for row in rows)
    assert numeric[1].tolist() == rows[1].tolist()


# ------------------------------------------------------------------ views


def test_structure_numeric_view_of_structure_and_triple() -> None:
    structure = _hexagonal_structure()
    view = StructureNumericView(structure)
    assert isinstance(view, NumericStructure)
    assert unwrap(view) is structure  # unwrap returns the raw original
    assert StructureNumericView(view) is view  # rewrap idempotence

    triple = (
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        [11, 17],
    )
    from_triple = StructureNumericView(triple)
    assert isinstance(from_triple, NumericStructure)
    assert unwrap(from_triple) is triple
    assert from_triple.cell.basis.tolist() == [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def test_cell_numeric_view_of_params_and_cell() -> None:
    params = (3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
    from_params = CellNumericView(params)
    assert isinstance(from_params, NumericCell)
    assert from_params.basis.shape == (3, 3)
    assert unwrap(from_params) is params
    assert CellNumericView(from_params) is from_params  # rewrap idempotence

    cell = Cell([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    from_cell = CellNumericView(cell)
    assert unwrap(from_cell) is cell
    assert from_cell.basis.tolist() == cell.basis.to_floats()


def test_sites_numeric_view_rewrap_and_unwrap() -> None:
    raw = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    view = SitesNumericView(raw)
    assert isinstance(view, NumericSites)
    assert unwrap(view) is raw
    assert SitesNumericView(view) is view
    assert len(view) == 2


# ------------------------------------------------------------------ .numeric() round-trip


def test_numeric_convenience_methods_roundtrip_exact() -> None:
    cell = Cell([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert cell.numeric().exact is cell

    sites = Sites([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert sites.numeric().exact is sites

    structure = _hexagonal_structure()
    assert structure.numeric().exact is structure


# ------------------------------------------------------------------ numpy requirement


def test_construction_requires_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Report numpy unavailable via the core flag the helpers read; construction must fail fast.
    monkeypatch.setattr(vectors_pkg, "_numpy_available", False)
    structure = _hexagonal_structure()
    with pytest.raises(ImportError, match=r"numpy"):
        NumericStructure(structure)
    with pytest.raises(ImportError, match=r"numpy"):
        NumericCell(structure.cell)
    with pytest.raises(ImportError, match=r"numpy"):
        NumericSites(structure.sites)
    with pytest.raises(ImportError, match=r"numpy"):
        structure.numeric()
