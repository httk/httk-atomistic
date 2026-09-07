"""Prove the storage-record float accessors use the store's stored DOUBLE columns.

These tests exercise a real ``SqlStore``: a lazy fetch returns row proxies whose
``_httk_stored_floats`` companion columns the three record backends
(``RecordCell``, ``RecordSites``, ``RecordStructure``) should read instead of decoding the
exact text columns, with exact fallback for a materialized (eager) fetch.
"""

import dataclasses
from fractions import Fraction

import pytest

pytest.importorskip("httk.store.backend.sql")
numpy = pytest.importorskip("numpy")

from httk.core import FracVector, SurdVector
from httk.store import Backend, EntryIdScheme, SqlStore
from httk.store.backend.sql import rows as rows_module

from httk.atomistic import (
    ASUStructure,
    ASUStructureRecord,
    CartesianSiteMoments,
    Cell,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    FundamentalDomainStructureRecord,
    Sites,
    Species,
    StructureEntry,
    UnitcellStructure,
    UnitcellStructureRecord,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.models.structure.record import RecordStructure

_ENTRY_RECORDS = {StructureEntry: (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)}


@pytest.fixture
def store():
    with Backend.sqlite() as db:
        yield SqlStore(db, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=_ENTRY_RECORDS)


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))


def _hexagonal_cell() -> Cell:
    """A genuinely irrational (surd) hexagonal cell: a=b=4, c=12, gamma=120deg."""
    zero = SurdVector(0)._as_scalar()
    four = SurdVector(4)._as_scalar()
    minus_two = SurdVector(-2)._as_scalar()
    twelve = SurdVector(12)._as_scalar()
    root_three = SurdVector.sqrt_of(3)
    basis = SurdVector._from_scalar_grid(
        [
            [four, zero, zero],
            [minus_two, root_three * 2, zero],
            [zero, zero, twelve],
        ],
        (3, 3),
    )
    return Cell(basis)


def _sites() -> Sites:
    return Sites(
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)]]
    )


def _cartesian_moments() -> CartesianSiteMoments:
    return CartesianSiteMoments([[0.0, 0.0, 1.5], [0.3, -0.2, 0.0], [1.0, 1.0, 1.0]])


def _crystalaxis_moments(cell: Cell) -> CrystalAxisSiteMoments:
    return CrystalAxisSiteMoments([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]], cell)


def _collinear_moments() -> CollinearSiteMoments:
    return CollinearSiteMoments([0.5, -0.5, 1.0])


def _unitcell(site_moments=None) -> UnitcellStructure:
    return UnitcellStructure(
        _hexagonal_cell(),
        _sites(),
        _species(),
        ("Na", "Cl", "Na"),
        site_moments=site_moments,
    )


def _domain_no_moments() -> ASUStructure:
    return ASUStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        225,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")),
        _species(),
    )


def _lazy(store, source):
    sid = store.save(source)
    store._clear_identity_caches()
    return store.fetch(UnitcellStructureRecord, sid)


def _eager(store, source):
    sid = store.save(source)
    store._clear_identity_caches()
    return store.fetch(UnitcellStructureRecord, sid, eager=True)


def _exact_quartet(source: UnitcellStructure):
    return (
        source.lattice_vectors,
        source.fractional_site_positions,
        source.cartesian_site_positions,
        source.cartesian_site_moments_floats(),
    )


def _assert_quartet_matches(view: UnitcellStructureView, source: UnitcellStructure) -> None:
    expected_lattice, expected_fractional, expected_cartesian, expected_moments = _exact_quartet(source)
    assert view.lattice_vectors == pytest.approx(numpy.array(expected_lattice), rel=1e-12, abs=1e-12)
    assert view.fractional_site_positions == pytest.approx(numpy.array(expected_fractional), rel=1e-12, abs=1e-12)
    assert view.cartesian_site_positions == pytest.approx(numpy.array(expected_cartesian), rel=1e-12, abs=1e-12)
    actual_moments = view.cartesian_site_moments_floats()
    if expected_moments is None:
        assert actual_moments is None
    else:
        assert actual_moments == pytest.approx(numpy.array(expected_moments), rel=1e-12, abs=1e-12)


# --------------------------------------------------------------- bypass proof


def test_bypass_proof_lazy_accessors_never_decode_exact(store, monkeypatch):
    source = _unitcell(_cartesian_moments())
    sid = store.save(source)
    store._clear_identity_caches()

    def _boom_fracvector(*args, **kwargs):
        raise AssertionError("exact FracVector decoder invoked")

    original_codec_named = rows_module.codec_named

    def _boom_surdscalar(values):
        raise AssertionError("exact SurdScalar decoder invoked")

    def guarded_codec_named(name):
        codec = original_codec_named(name)
        if name == "surdscalar":
            return dataclasses.replace(codec, decode=_boom_surdscalar)
        return codec

    monkeypatch.setattr(rows_module, "decode_fracvector_exact", _boom_fracvector)
    monkeypatch.setattr(rows_module, "codec_named", guarded_codec_named)

    # Fetch AFTER patching, so no exact attribute has been touched yet.
    record = store.fetch(UnitcellStructureRecord, sid)
    view = UnitcellStructureView(record, kind="record")
    _assert_quartet_matches(view, source)

    # The patch is live: touching the exact sites field now raises through the same call.
    with pytest.raises(AssertionError):
        _ = record.sites.reduced_coords


# --------------------------------------------------------------- values (lazy, unpatched)


def test_lazy_record_values_match_exact_route(store):
    source = _unitcell(_cartesian_moments())
    record = _lazy(store, source)
    view = UnitcellStructureView(record, kind="record")
    _assert_quartet_matches(view, source)


# --------------------------------------------------------------- fallback (eager/materialized)


def test_eager_record_falls_back_to_exact_route(store):
    source = _unitcell(_cartesian_moments())
    record = _eager(store, source)
    assert getattr(record, "_httk_stored_floats", None) is None
    view = UnitcellStructureView(record, kind="record")
    _assert_quartet_matches(view, source)


# --------------------------------------------------------------- length


def test_len_uses_stored_row_count_without_filling_reduced_coords(store):
    source = _unitcell()
    record = _lazy(store, source)
    view = UnitcellStructureView(record, kind="record")

    assert len(view.sites) == len(source.sites)
    assert "_reduced_coords" not in view.sites.__dict__


# --------------------------------------------------------------- moment kinds


def test_crystalaxis_moments_fall_back_to_exact_route(store):
    cell = _hexagonal_cell()
    source = _unitcell(_crystalaxis_moments(cell))
    record = _lazy(store, source)
    view = UnitcellStructureView(record, kind="record")

    expected = source.cartesian_site_moments_floats()
    actual = view.cartesian_site_moments_floats()
    assert expected is not None
    assert actual == pytest.approx(numpy.array(expected), rel=1e-12, abs=1e-12)


def test_collinear_moments_are_none(store):
    source = _unitcell(_collinear_moments())
    record = _lazy(store, source)
    view = UnitcellStructureView(record, kind="record")

    assert source.cartesian_site_moments_floats() is None
    assert view.cartesian_site_moments_floats() is None


def test_no_moments_are_none(store):
    source = _unitcell()
    record = _lazy(store, source)
    view = UnitcellStructureView(record, kind="record")

    assert view.cartesian_site_moments_floats() is None


# --------------------------------------------------------------- domain record


def test_domain_record_does_not_raise_and_uses_default_route(store):
    source = _domain_no_moments()
    sid = store.save(source)
    store._clear_identity_caches()
    record = store.fetch(ASUStructureRecord, sid)

    assert RecordStructure(record).cartesian_site_moments_floats() is None
