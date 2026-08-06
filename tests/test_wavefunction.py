import shutil
import subprocess
import sys
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")

from httk.core import FracVector, load, save

from httk.atomistic import PlaneWaveFunctions, save_vesta, wavefunction_overlap
from httk.atomistic.wavefunction import _generate_gvectors, _generate_kgrid, _wavecar_payload_from_planewaves


def _counts(kpoints, encut=100.0):
    cell = numpy.diag([4.0, 4.0, 4.0])
    grid = numpy.array([13, 13, 13])
    return [len(_generate_gvectors(_generate_kgrid(grid, False), point, cell, encut)) for point in kpoints]


def _standard(dtype=numpy.complex64):
    kpoints = numpy.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    counts = _counts(kpoints)
    coefficients = {(0, kpt, 0): numpy.ones(counts[kpt], dtype=dtype) for kpt in range(2)}
    return PlaneWaveFunctions(
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        encut=100,
        kpoints=kpoints,
        eigenvalues=numpy.array([[[1.0], [2.0]]]),
        occupations=numpy.ones((1, 2, 1)),
        coefficients=coefficients,
    )


def _gamma_standard(dtype=numpy.complex128):
    grid = numpy.array([13, 13, 13])
    gvecs = _generate_gvectors(_generate_kgrid(grid, False), numpy.zeros(3), numpy.diag([4.0, 4.0, 4.0]), 100)
    values = numpy.zeros(len(gvecs), dtype=dtype)
    values[numpy.all(gvecs == 0, axis=1)] = 1
    return PlaneWaveFunctions(
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        encut=100,
        kpoints=[[0, 0, 0]],
        eigenvalues=[[[1.0]]],
        occupations=[[[1.0]]],
        coefficients={(0, 0, 0): values},
    )


def test_in_memory_properties_and_validation():
    wave = _standard()
    assert (wave.nspins, wave.nkpts, wave.nbands) == (1, 2, 1)
    assert wave.double_precision is False
    assert wave.coefficients(0, 1, 0).dtype == numpy.complex64
    with pytest.raises(ValueError, match="shape|inconsistent"):
        PlaneWaveFunctions(
            cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            encut=100,
            kpoints=[[0, 0, 0]],
            eigenvalues=[[[1], [2]]],
            occupations=[[[1]]],
            coefficients={(0, 0, 0): numpy.ones(1, complex)},
        )
    with pytest.raises(ValueError, match="cover"):
        PlaneWaveFunctions(
            cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            encut=100,
            kpoints=[[0, 0, 0]],
            eigenvalues=[[[1]]],
            occupations=[[[1]]],
            coefficients={},
        )


def test_vector_adoption_and_exact_input():
    raw = numpy.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    wave = PlaneWaveFunctions(
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        encut=100,
        kpoints=raw,
        eigenvalues=[[[1], [2]]],
        occupations=[[[1], [1]]],
        coefficients={(0, 0, 0): numpy.ones(147, complex), (0, 1, 0): numpy.ones(155, complex)},
    )
    assert numpy.shares_memory(wave.kpoints, raw)
    assert (
        PlaneWaveFunctions(
            cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            encut=100,
            kpoints=FracVector.create([[0, 0, 0], ["1/4", 0, 0]]),
            eigenvalues=[[[1], [2]]],
            occupations=[[[1], [1]]],
            coefficients={(0, 0, 0): numpy.ones(147, complex), (0, 1, 0): numpy.ones(155, complex)},
        ).nkpts
        == 2
    )


@pytest.mark.parametrize("dtype", [numpy.complex64, numpy.complex128])
def test_wavecar_roundtrip(tmp_path, dtype):
    pytest.importorskip("httk.io")
    original = _standard(dtype)
    path = tmp_path / "WAVECAR"
    save(original, path)
    with load(str(path)) as loaded:
        assert isinstance(loaded, PlaneWaveFunctions)
        numpy.testing.assert_array_equal(loaded.kpoints, original.kpoints)
        numpy.testing.assert_array_equal(loaded.nplanewaves, original.nplanewaves)
        numpy.testing.assert_array_equal(loaded.coefficients(0, 1, 0), original.coefficients(0, 1, 0))


class _CountingSource:
    def __init__(self, source):
        self._source = source
        self.reads = 0
        for name in (
            "nspins",
            "nkpts",
            "nbands",
            "encut",
            "cell",
            "kpoints",
            "eigenvalues",
            "occupations",
            "nplanewaves",
            "double_precision",
        ):
            setattr(self, name, getattr(source, name))

    @property
    def closed(self):
        return False

    def coefficients(self, spin, kpt, band):
        self.reads += 1
        return self._source.coefficients(spin, kpt, band)


def test_coefficients_cache_false_reads_once_without_caching():
    source = _CountingSource(_wavecar_payload_from_planewaves(_standard())["wavecar"])
    wave = PlaneWaveFunctions(source)
    assert wave.coefficients(0, 0, 0, cache=False).shape == (147,)
    assert source.reads == 1
    assert wave._coeff_cache == {}
    wave.coefficients(0, 0, 0, cache=True)
    assert source.reads == 2
    assert (0, 0, 0) in wave._coeff_cache
    wave.coefficients(0, 0, 0, cache=False)
    assert source.reads == 2


def test_wavecar_context_manager_and_lifecycle(tmp_path):
    pytest.importorskip("httk.io")
    path = tmp_path / "WAVECAR"
    save(_standard(), path)
    with load(str(path)) as loaded:
        cached = loaded.coefficients(0, 1, 0)
        assert loaded.closed is False
        assert loaded.nkpts == 2
    assert loaded.closed is True
    numpy.testing.assert_array_equal(loaded.coefficients(0, 1, 0), cached)
    assert loaded.nkpts == 2
    with pytest.raises(ValueError, match="closed"):
        loaded.coefficients(0, 0, 0)
    with pytest.raises(ValueError, match="closed"):
        loaded.__enter__()
    loaded.close()
    assert loaded.closed is True


def test_in_memory_close_is_noop():
    wave = _standard()
    wave.close()
    assert wave.closed is False
    with wave as entered:
        assert entered is wave
    assert wave.closed is False


def test_gamma_conversion_and_gvectors():
    standard = _gamma_standard()
    assert standard.is_gamma is False
    assert standard.gvectors().shape[0] == 2 * len(standard.select(format="gamma").gvectors()) - 1
    for half in ("x", "z"):
        gamma = standard.select(format="gamma", gamma_half=half)
        assert gamma.is_gamma and gamma.gamma_half == half
        restored = gamma.select(format="std")
        assert abs(
            wavefunction_overlap(standard.realspace_wave(0, 0, 0), restored.realspace_wave(0, 0, 0))
        ) == pytest.approx(1)


@pytest.mark.parametrize("half", ["x", "z"])
def test_gamma_realspace_known_answer(half):
    cell = numpy.diag([4.0, 4.0, 4.0])
    encut = 100.0
    grid_size = numpy.array([13, 13, 13])
    gamma_gvecs = _generate_gvectors(_generate_kgrid(grid_size, True, half), numpy.zeros(3), cell, encut)
    axis = 0 if half == "x" else 2
    origin = numpy.flatnonzero(numpy.all(gamma_gvecs == 0, axis=1))[0]
    boundary = numpy.flatnonzero((gamma_gvecs[:, axis] == 0) & numpy.any(gamma_gvecs != 0, axis=1))[0]
    exclusive = (1, 0, -1) if half == "x" else (-1, 0, 1)
    opposite = tuple(-value for value in exclusive)
    gamma_set = {tuple(int(value) for value in gvec) for gvec in gamma_gvecs}
    assert exclusive in gamma_set
    assert opposite not in gamma_set
    exclusive_index = numpy.flatnonzero(numpy.all(gamma_gvecs == exclusive, axis=1))[0]
    indices = numpy.arange(len(gamma_gvecs))
    extra_index = numpy.flatnonzero((indices != origin) & (indices != boundary) & (indices != exclusive_index))[0]
    selected = [origin, boundary, exclusive_index, extra_index]
    values = numpy.zeros(len(gamma_gvecs), dtype=numpy.complex128)
    values[selected] = [1.25 + 0j, -0.75 + 1.1j, 0.4 - 0.8j, -1.3 - 0.2j]
    wave = PlaneWaveFunctions(
        cell=cell,
        encut=encut,
        kpoints=[[0, 0, 0]],
        eigenvalues=[[[1.0]]],
        occupations=[[[1.0]]],
        coefficients={(0, 0, 0): values},
        gamma_half=half,
    )

    full_coefficients = {}
    for gvec, coefficient in zip(gamma_gvecs, values):
        if coefficient == 0:
            continue
        key = tuple(int(value) for value in gvec)
        scale = coefficient if key == (0, 0, 0) else coefficient / numpy.sqrt(2)
        full_coefficients[key] = scale
        if key != (0, 0, 0):
            full_coefficients[tuple(-value for value in key)] = scale.conjugate()
    grid = wave.kgrid_size * 2
    mesh = numpy.indices(tuple(int(value) for value in grid), dtype=float)
    coordinates = tuple(mesh[index] / grid[index] for index in range(3))
    reference = numpy.zeros(tuple(int(value) for value in grid), dtype=complex)
    for gvec, coefficient in full_coefficients.items():
        phase = sum(gvec[index] * coordinates[index] for index in range(3))
        reference += coefficient * numpy.exp(2j * numpy.pi * phase)
    reference /= numpy.sqrt(numpy.prod(grid))
    reference /= numpy.linalg.norm(reference)
    numpy.testing.assert_allclose(wave.realspace_wave(0, 0, 0), reference, rtol=1e-10, atol=1e-12)


def test_gamma_save_load(tmp_path):
    pytest.importorskip("httk.io")
    gamma = _gamma_standard().select(format="gamma")
    path = tmp_path / "WAVECAR"
    save(gamma, path)
    with load(str(path)) as loaded:
        assert loaded.is_gamma and loaded.gamma_half == "x"
        numpy.testing.assert_array_equal(loaded.nplanewaves, gamma.nplanewaves)


def test_gamma_half_load_hint(tmp_path):
    pytest.importorskip("httk.io")
    from httk.io.vasp import read_wavecar

    path = tmp_path / "WAVECAR"
    save(_gamma_standard().select(format="gamma"), path)
    with load(str(path), gamma_half="z") as loaded:
        assert loaded.is_gamma is True
        assert loaded.gamma_half == "z"
    with PlaneWaveFunctions(read_wavecar(path, gamma_half="z")) as direct:
        assert direct.gamma_half == "z"
    with PlaneWaveFunctions(read_wavecar(path, gamma_half="z"), gamma_half="x") as overridden:
        assert overridden.gamma_half == "x"


def test_select_and_validation():
    wave = _standard()
    selected = wave.select(kpts=[1], bands=[0], spins=[0])
    assert selected.kpoints.tolist() == [[0.25, 0.0, 0.0]]
    numpy.testing.assert_array_equal(selected.coefficients(0, 0, 0), wave.coefficients(0, 1, 0))
    with pytest.raises(ValueError, match="duplicates"):
        wave.select(kpts=[0, 0])
    with pytest.raises(ValueError, match="range"):
        wave.select(bands=[2])
    with pytest.raises(ValueError, match="gamma"):
        wave.select(format="gamma")


def test_save_vesta(tmp_path):
    pytest.importorskip("httk.io")
    from httk.atomistic import Cell, Sites, Species, UnitcellStructure

    structure = UnitcellStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), Sites([[0, 0, 0]]), (Species("C", ("C",), (1,)),), ("C",)
    )
    wave = numpy.arange(24, dtype=float).reshape(2, 3, 4).astype(complex)
    save_vesta(str(tmp_path / "wave"), structure, wave)
    text = (tmp_path / "wave_r.vasp").read_text()
    assert "2 3 4\n" in text
    assert text.count("0.00000000E+00") >= 1


FIXTURES = Path(__file__).resolve().parents[2] / "old/httk/Examples/example_resources/wavefunctions"


@pytest.mark.skipif(not FIXTURES.exists(), reason="workspace-only real WAVECAR fixtures")
def test_real_fixture_gvectors(tmp_path):
    pytest.importorskip("httk.io")
    destinations = {}
    for suffix in ("std", "gam"):
        destination = tmp_path / f"{suffix}.wavecar"
        shutil.copyfile(FIXTURES / f"WAVECAR.{suffix}", destination)
        destinations[suffix] = destination
    with load(str(destinations["std"])) as standard, load(str(destinations["gam"])) as gamma:
        for suffix, wave in (("std", standard), ("gam", gamma)):
            assert wave.is_gamma is (suffix == "gam")
            counts = [wave.gvectors(kpt).shape[0] for kpt in range(wave.nkpts)]
            assert counts == wave.nplanewaves.tolist()
            print(f"WAVECAR.{suffix}: G-vector counts {counts}")
        restored = gamma.select(format="std")
        assert abs(
            wavefunction_overlap(gamma.realspace_wave(0, 0, 0), restored.realspace_wave(0, 0, 0))
        ) == pytest.approx(1, rel=1e-6)


def test_numpy_absent_probe():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['numpy'] = None; from httk.atomistic import PlaneWaveFunctions; "
                "PlaneWaveFunctions()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "httk-atomistic[numpy]" in result.stderr
