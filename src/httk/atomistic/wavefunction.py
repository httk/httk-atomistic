"""Provide NumPy-native plane-wave wavefunctions and VASP WAVECAR adapters."""

import math
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

from httk.core import VectorLike, unview

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.view import CellView

# Constants used by VASP and retained from httk v1.
RYTOEV = 13.605826
AUTOA = 0.529177249
PI = 3.141592653589793238


def _as_float_array(value: VectorLike, dtype: Any = None) -> Any:
    """Render a vector-like value as a base numpy array, adopting raw arrays."""
    import numpy
    from httk.core.vectors import VectorNumpyView

    return numpy.asarray(unview(VectorNumpyView(value)), dtype=numpy.float64 if dtype is None else dtype)


def _generate_kgrid(grid_size: Any, gamma: bool, gamma_half: str = "x") -> Any:
    import numpy

    grid_size = numpy.asarray(grid_size, dtype=numpy.int64)
    axes = [(numpy.arange(size) + size // 2) % size - size // 2 for size in grid_size]
    if gamma:
        if gamma_half == "x":
            axes[0] = axes[0][axes[0] >= 0]

            def keep(values: Any) -> Any:
                return (
                    (values[:, 0] > 0)
                    | ((values[:, 0] == 0) & (values[:, 1] > 0))
                    | ((values[:, 0] == 0) & (values[:, 1] == 0) & (values[:, 2] >= 0))
                )
        elif gamma_half == "z":
            axes[2] = axes[2][axes[2] >= 0]

            def keep(values: Any) -> Any:
                return (
                    (values[:, 2] > 0)
                    | ((values[:, 2] == 0) & (values[:, 1] > 0))
                    | ((values[:, 2] == 0) & (values[:, 1] == 0) & (values[:, 0] >= 0))
                )
        else:
            raise ValueError(f"Unknown gamma-halving scheme provided {gamma_half!r}")
    else:
        keep = lambda values: numpy.ones(values.shape[0], dtype=bool)

    grid = numpy.array(numpy.meshgrid(axes[2], axes[1], axes[0], indexing="ij")).reshape(3, -1).T[:, [2, 1, 0]]
    return grid[keep(grid)]


def _generate_gvectors(kgrid: Any, kvec: Any, cell_basis_float: Any, encut: float) -> Any:
    import numpy

    reciprocal = 2 * PI * numpy.linalg.inv(numpy.asarray(cell_basis_float, dtype=numpy.float64)).T
    kinetic = RYTOEV * AUTOA**2 * numpy.linalg.norm(numpy.matmul(numpy.asarray(kgrid) + kvec, reciprocal), axis=1) ** 2
    return numpy.asarray(kgrid)[kinetic < float(encut)]


def _expand_gamma_wav(buffer: Any, xyz: Any) -> Any:
    import numpy

    buffer[-xyz[:, 0], -xyz[:, 1], -xyz[:, 2]] = buffer[xyz[:, 0], xyz[:, 1], xyz[:, 2]].conjugate()
    buffer /= numpy.sqrt(2)
    buffer[0, 0, 0] *= numpy.sqrt(2)
    return buffer


def _expand_gamma_coeffs(coeffs: Any, std_gvecs: Any, gam_gvecs: Any, buffer: Any = None) -> Any:
    import numpy

    if buffer is None:
        sizes = [int(numpy.max(std_gvecs[:, i]) - numpy.min(std_gvecs[:, i]) + 1) for i in range(3)]
        buffer = numpy.zeros(sizes, dtype=numpy.complex128)
    buffer[gam_gvecs[:, 0], gam_gvecs[:, 1], gam_gvecs[:, 2]] = coeffs
    return _expand_gamma_wav(buffer, gam_gvecs)[std_gvecs[:, 0], std_gvecs[:, 1], std_gvecs[:, 2]]


def _to_real_wave(
    coeffs: Any, grid_size: Any, gvecs: Any, gamma: bool = False, gamma_half: str = "x", norm: bool = True
) -> Any:
    import numpy

    grid = numpy.asarray(grid_size, dtype=numpy.int64) * 2
    if gamma:
        if gamma_half == "x":
            phi = numpy.zeros((grid[0] // 2 + 1, grid[1], grid[2]), dtype=complex)
        elif gamma_half == "z":
            phi = numpy.zeros((grid[0], grid[1], grid[2] // 2 + 1), dtype=complex)
        else:
            raise ValueError('Unrecognized gamma-half argument. "z" or "x" is supported')
    else:
        phi = numpy.zeros(grid, dtype=complex)

    phi[gvecs[:, 0], gvecs[:, 1], gvecs[:, 2]] = coeffs
    if gamma_half == "x" and gamma:
        _expand_gamma_wav(phi, gvecs[gvecs[:, 0] == 0])
        tmp = numpy.swapaxes(phi, 0, 2)
        tmp = numpy.fft.irfftn(tmp, s=tuple(int(value) for value in grid[[2, 1, 0]]), axes=(0, 1, 2), norm="ortho")
        phi = numpy.swapaxes(tmp, 0, 2)
    elif gamma_half == "z" and gamma:
        _expand_gamma_wav(phi, gvecs[gvecs[:, 2] == 0])
        phi = numpy.fft.irfftn(phi, s=tuple(int(value) for value in grid), axes=(0, 1, 2), norm="ortho")
    else:
        phi = numpy.fft.ifftn(phi, norm="ortho")
    if norm:
        phi /= numpy.linalg.norm(phi)
    return numpy.asarray(phi, dtype=complex)


def _reduce_std_coeffs(coeffs: Any, grid_size: Any, std_gvecs: Any, gam_gvecs: Any, gamma_half: str = "x") -> Any:
    import numpy

    if gamma_half not in ("x", "z"):
        raise ValueError("Unrecognized gamma-half argument. z or x is supported")
    phi = _to_real_wave(coeffs, grid_size, std_gvecs, False, gamma_half, norm=False)
    phi = numpy.sqrt(phi.real**2 + phi.imag**2) * numpy.sign(phi.real)
    grid = numpy.asarray(grid_size, dtype=numpy.int64) * 2
    if gamma_half == "x":
        tmp = numpy.swapaxes(phi, 2, 0)
        phi = numpy.swapaxes(
            numpy.fft.rfftn(tmp, s=tuple(int(value) for value in grid[[2, 1, 0]]), axes=(0, 1, 2), norm="ortho"),
            2,
            0,
        )
    else:
        phi = numpy.fft.rfftn(phi, s=tuple(int(value) for value in grid), axes=(0, 1, 2), norm="ortho")
    phi *= numpy.sqrt(2)
    phi[0, 0, 0] /= numpy.sqrt(2)
    return phi[gam_gvecs[:, 0], gam_gvecs[:, 1], gam_gvecs[:, 2]]


def _validate_indices(indices: Any, size: int, name: str) -> Any:
    import numpy

    if isinstance(indices, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of zero-based indices")
    try:
        result = numpy.asarray(list(indices), dtype=object)
    except TypeError:
        raise ValueError(f"{name} must be a non-empty sequence of zero-based indices") from None
    if result.size == 0 or any(
        isinstance(value, bool) or not isinstance(value, (int, numpy.integer)) for value in result
    ):
        raise ValueError(f"{name} must be a non-empty sequence of zero-based indices")
    result = result.astype(numpy.int64)
    if numpy.any(result < 0) or numpy.any(result >= size):
        raise ValueError(f"{name} indices are out of range [0, {size})")
    if len(set(result.tolist())) != len(result):
        raise ValueError(f"{name} indices must not contain duplicates")
    return result


class PlaneWaveFunctions:
    """Provide a zero-based, NumPy-native collection of plane-wave coefficients.

    This is an eager NumPy representation, not a backend or view family. NumPy is required
    at construction time; install the ``httk-atomistic[numpy]`` extra when it is absent.

    A WAVECAR does not store whether gamma compression used the ``x`` or ``z``
    half-space. The default interpretation is ``x``; pass ``gamma_half="z"``
    to :func:`~httk.core.load` when loading a z-half gamma WAVECAR. Gamma compression is
    detected from the k-point and plane-wave count during construction.

    :param source: A WAVECAR source or neutral WAVECAR payload, or ``None`` for in-memory data.
    :param cell: The real-space cell used by the in-memory coefficients.
    :param encut: The plane-wave energy cutoff used to generate reciprocal vectors.
    :param kpoints: The k-points used by the in-memory coefficients.
    :param eigenvalues: The band eigenvalues.
    :param occupations: The band occupations.
    :param coefficients: The coefficient vectors keyed by zero-based spin, k-point, and band.
    :param nplanewaves: The number of plane waves for each k-point, or ``None`` to infer it.
    :param double_precision: Whether to retain double-precision complex coefficients.
    :param gamma_half: The gamma-compression half-space, ``"x"`` or ``"z"``, if applicable.
    :raises ImportError: If NumPy is unavailable.
    """

    def __init__(
        self,
        source: Any = None,
        *,
        cell: Any = None,
        encut: Any = None,
        kpoints: Any = None,
        eigenvalues: Any = None,
        occupations: Any = None,
        coefficients: Any = None,
        nplanewaves: Any = None,
        double_precision: Any = None,
        gamma_half: Any = None,
    ) -> None:
        require_numpy()
        import numpy

        self._coeff_cache: dict[tuple[int, int, int], Any] = {}
        self._gvec_cache: dict[int, Any] = {}
        self._source: Any = None
        if source is not None:
            if any(
                value is not None
                for value in (cell, encut, kpoints, eigenvalues, occupations, coefficients, nplanewaves)
            ):
                raise ValueError("source cannot be combined with metadata or coefficients")
            if isinstance(source, Mapping):
                if source.get("format") != "vasp-wavecar" or "wavecar" not in source:
                    raise ValueError("source mapping must be a vasp-wavecar payload")
                if gamma_half is None:
                    gamma_half = source.get("gamma_half")
                source = source["wavecar"]
            required = (
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
                "coefficients",
            )
            if any(not hasattr(source, name) for name in required):
                raise ValueError("source must provide the WavecarSource contract")
            self._source = source
            nspins, nkpts, nbands = int(source.nspins), int(source.nkpts), int(source.nbands)
            encut = source.encut
            cell = source.cell
            kpoints = source.kpoints
            eigenvalues = source.eigenvalues
            occupations = source.occupations
            nplanewaves = source.nplanewaves
            if double_precision is None:
                double_precision = source.double_precision
        else:
            if coefficients is None or any(value is None for value in (cell, encut, kpoints, eigenvalues, occupations)):
                raise ValueError("cell, encut, kpoints, eigenvalues, occupations, and coefficients are required")
            if not isinstance(coefficients, Mapping):
                raise ValueError("coefficients must be a mapping keyed by (spin, k-point, band)")
            kpoints_array = _as_float_array(kpoints)
            eigenvalues_array = _as_float_array(eigenvalues)
            occupations_array = _as_float_array(occupations)
            if kpoints_array.ndim != 2 or kpoints_array.shape[1] != 3:
                raise ValueError("kpoints must have shape (nkpts, 3)")
            if eigenvalues_array.ndim != 3:
                raise ValueError("eigenvalues must have shape (nspins, nkpts, nbands)")
            nspins, nkpts, nbands = eigenvalues_array.shape
            if kpoints_array.shape[0] != nkpts:
                raise ValueError("kpoints and eigenvalues have inconsistent nkpts")
            if occupations_array.shape != eigenvalues_array.shape:
                raise ValueError("occupations and eigenvalues must have the same shape")
            expected = {(spin, kpt, band) for spin in range(nspins) for kpt in range(nkpts) for band in range(nbands)}
            if set(coefficients) != expected:
                raise ValueError("coefficients must cover every zero-based (spin, k-point, band) triple")
            lengths: dict[int, int] = {}
            dtype: Any = None
            for key, value in coefficients.items():
                array = numpy.asarray(value)
                if array.ndim != 1 or array.dtype not in (numpy.dtype(numpy.complex64), numpy.dtype(numpy.complex128)):
                    raise ValueError("each coefficient array must be one-dimensional complex64 or complex128")
                if dtype is None:
                    dtype = array.dtype
                elif array.dtype != dtype:
                    raise ValueError("coefficient arrays must all have the same dtype")
                previous = lengths.setdefault(key[1], len(array))
                if previous != len(array):
                    raise ValueError("coefficient lengths must be consistent for each k-point")
                self._coeff_cache[key] = array
            derived_nplanewaves = numpy.asarray(list(lengths.values()), dtype=numpy.int64)
            if nplanewaves is None:
                nplanewaves = derived_nplanewaves
            else:
                supplied_nplanewaves = numpy.asarray(nplanewaves, dtype=numpy.int64)
                if supplied_nplanewaves.shape != derived_nplanewaves.shape or not numpy.array_equal(
                    supplied_nplanewaves, derived_nplanewaves
                ):
                    raise ValueError("explicit nplanewaves does not match coefficient lengths")
            if double_precision is None:
                double_precision = dtype == numpy.dtype(numpy.complex128)
            kpoints, eigenvalues, occupations = kpoints_array, eigenvalues_array, occupations_array

        self._nspins, self._nkpts, self._nbands = nspins, nkpts, nbands
        if isinstance(double_precision, numpy.bool_):
            double_precision = bool(double_precision)
        if not isinstance(double_precision, bool):
            raise ValueError("double_precision must be a boolean")
        self._cell = cell if isinstance(cell, Cell) else CellView(cell)
        self._encut = float(encut)
        if not math.isfinite(self._encut) or self._encut <= 0:
            raise ValueError("encut must be a positive finite number")
        self._kpoints = _as_float_array(kpoints)
        self._eigenvalues = _as_float_array(eigenvalues)
        self._occupations = _as_float_array(occupations)
        if self._kpoints.shape != (nkpts, 3):
            raise ValueError("kpoints must have shape (nkpts, 3)")
        if self._eigenvalues.shape != (nspins, nkpts, nbands) or self._occupations.shape != self._eigenvalues.shape:
            raise ValueError("eigenvalues and occupations must have shape (nspins, nkpts, nbands)")
        self._nplanewaves = numpy.asarray(nplanewaves, dtype=numpy.int64)
        if self._nplanewaves.shape != (nkpts,) or numpy.any(self._nplanewaves <= 0):
            raise ValueError("nplanewaves must have one positive value per k-point")
        if source is None and any(
            len(self._coeff_cache[(0, kpt, 0)]) != self._nplanewaves[kpt] for kpt in range(nkpts)
        ):
            raise ValueError("explicit nplanewaves does not match coefficient lengths")
        self._double_precision = double_precision
        cell_lengths = numpy.asarray([float(value) for value in self._cell.lengths], dtype=numpy.float64)
        cutoff = numpy.sqrt(self._encut / RYTOEV) * cell_lengths / (AUTOA * 2 * PI)
        self._kgrid_size = numpy.asarray([math.ceil(float(value)) * 2 + 1 for value in cutoff], dtype=numpy.int64)
        self._cell_basis_float = numpy.asarray(self._cell.basis.to_floats(), dtype=numpy.float64)
        self._gamma_half = None
        self._is_gamma = False
        if nkpts == 1 and numpy.array_equal(self._kpoints[0], numpy.zeros(3)):
            selected_half = "x" if gamma_half is None else gamma_half
            if selected_half not in ("x", "z"):
                raise ValueError("gamma_half must be 'x' or 'z'")
            standard_count = len(
                _generate_gvectors(
                    _generate_kgrid(self._kgrid_size, False), self._kpoints[0], self._cell_basis_float, self._encut
                )
            )
            gamma_count = len(
                _generate_gvectors(
                    _generate_kgrid(self._kgrid_size, True, selected_half),
                    self._kpoints[0],
                    self._cell_basis_float,
                    self._encut,
                )
            )
            if self._nplanewaves[0] == standard_count:
                self._is_gamma = False
            elif self._nplanewaves[0] == gamma_count:
                self._is_gamma = True
                self._gamma_half = selected_half
            else:
                raise ValueError(
                    f"No. of planewaves inconsistent: standard={standard_count}, gamma={gamma_count}, supplied={self._nplanewaves[0]}"
                )

    @property
    def nspins(self) -> int:
        """Return the number of spin channels."""
        return self._nspins

    @property
    def nkpts(self) -> int:
        """Return the number of k-points."""
        return self._nkpts

    @property
    def nbands(self) -> int:
        """Return the number of bands."""
        return self._nbands

    @property
    def encut(self) -> float:
        """Return the plane-wave energy cutoff."""
        return self._encut

    @property
    def cell(self) -> Cell:
        """Return the real-space cell."""
        return self._cell

    @property
    def kpoints(self) -> Any:
        """Return the k-point coordinates."""
        return self._kpoints

    @property
    def eigenvalues(self) -> Any:
        """Return the band eigenvalues."""
        return self._eigenvalues

    @property
    def occupations(self) -> Any:
        """Return the band occupations."""
        return self._occupations

    @property
    def nplanewaves(self) -> Any:
        """Return the plane-wave count for each k-point."""
        return self._nplanewaves

    @property
    def double_precision(self) -> bool:
        """Return whether coefficients use double precision."""
        return self._double_precision

    @property
    def is_gamma(self) -> bool:
        """Return whether the coefficients use gamma compression."""
        return self._is_gamma

    @property
    def gamma_half(self) -> str | None:
        """Return the detected gamma-compression half-space, if applicable."""
        return self._gamma_half

    @property
    def kgrid_size(self) -> Any:
        """Return the reciprocal-grid dimensions used for transforms."""
        return self._kgrid_size

    def close(self) -> None:
        """Close a file-backed source while retaining cached coefficients and metadata."""
        if self._source is not None:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()

    @property
    def closed(self) -> bool:
        """Report whether the file-backed source is closed."""
        if self._source is None:
            return False
        return bool(getattr(self._source, "closed", False))

    def __enter__(self) -> Self:
        """Enter an open wavefunction source for context-managed use.

        :return: This wavefunction collection.
        :raises ValueError: If the source is already closed.
        """
        if self.closed:
            raise ValueError("Cannot enter a closed PlaneWaveFunctions.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the source while retaining cached coefficients and metadata."""
        self.close()

    def _check_index(self, value: Any, size: int, name: str) -> int:
        import numpy

        if isinstance(value, bool) or not isinstance(value, (int, numpy.integer)) or not 0 <= value < size:
            raise ValueError(f"{name} index {value!r} is out of range [0, {size})")
        return int(value)

    def coefficients(self, spin: int, kpt: int, band: int, *, cache: bool = True) -> Any:
        """Return one coefficient vector, using an existing cache even when ``cache=False``.

        An uncached source read occurs once and is not stored; cached coefficients remain
        available after a file-backed source is closed.

        :param spin: The zero-based spin index.
        :param kpt: The zero-based k-point index.
        :param band: The zero-based band index.
        :param cache: Whether to cache a coefficient vector read from the source.
        :return: The selected coefficient vector.
        :raises ValueError: If an index is out of range or source coefficients have the wrong length.
        """
        spin = self._check_index(spin, self._nspins, "spin")
        kpt = self._check_index(kpt, self._nkpts, "k-point")
        band = self._check_index(band, self._nbands, "band")
        key = (spin, kpt, band)
        if key in self._coeff_cache:
            return self._coeff_cache[key]
        import numpy

        values = numpy.asarray(self._source.coefficients(*key))
        if values.ndim != 1 or len(values) != self._nplanewaves[kpt]:
            raise ValueError(f"coefficients{key} has the wrong length")
        if cache:
            self._coeff_cache[key] = values
        return values

    def gvectors(self, kpt: int = 0, *, gamma: bool | None = None, gamma_half: str | None = None) -> Any:
        """Return the reciprocal grid vectors for a k-point.

        :param kpt: The zero-based k-point index.
        :param gamma: Whether to use gamma compression, or the construction default when ``None``.
        :param gamma_half: The gamma-compression half-space, if gamma compression is requested.
        :return: The reciprocal grid vectors selected by the cutoff.
        :raises ValueError: If the k-point, gamma flag, or half-space is invalid.
        """
        kpt = self._check_index(kpt, self._nkpts, "k-point")
        requested_gamma = self._is_gamma if gamma is None else gamma
        if not isinstance(requested_gamma, bool):
            raise ValueError("gamma must be a boolean or None")
        requested_half = self._gamma_half if gamma_half is None else gamma_half
        if requested_gamma and requested_half not in ("x", "z"):
            raise ValueError("gamma_half must be 'x' or 'z' when gamma is true")
        default = requested_gamma == self._is_gamma and (not requested_gamma or requested_half == self._gamma_half)
        if default and kpt in self._gvec_cache:
            return self._gvec_cache[kpt]
        grid = _generate_kgrid(self._kgrid_size, requested_gamma, requested_half or "x")
        values = _generate_gvectors(grid, self._kpoints[kpt], self._cell_basis_float, self._encut)
        if default:
            self._gvec_cache[kpt] = values
        return values

    def realspace_wave(self, spin: int, kpt: int, band: int, *, norm: bool = True) -> Any:
        """Transform coefficients to a real-space wave using NumPy FFTs.

        The transform uses ``numpy.fft`` with ``norm="ortho"``. Gamma-compressed coefficients
        are expanded according to the detected half-space before the transform.

        :param spin: The zero-based spin index.
        :param kpt: The zero-based k-point index.
        :param band: The zero-based band index.
        :param norm: Whether to normalize the resulting wave to unit norm.
        :return: The real-space wave on the reciprocal grid.
        :raises ValueError: If an index is out of range or the stored gamma metadata is invalid.
        """
        return _to_real_wave(
            self.coefficients(spin, kpt, band),
            self._kgrid_size,
            self.gvectors(kpt),
            self._is_gamma,
            self._gamma_half or "x",
            norm,
        )

    def select(
        self,
        spins: Sequence[int] | None = None,
        kpts: Sequence[int] | None = None,
        bands: Sequence[int] | None = None,
        *,
        format: str | None = None,
        gamma_half: str = "x",
    ) -> "PlaneWaveFunctions":
        """Select spins, k-points, and bands, optionally converting coefficient format.

        Indices are zero-based and must be unique. Converting standard coefficients to gamma
        format derives a signed real wave from the standard complex wave and therefore destroys
        phase information; converting gamma coefficients to standard format expands the stored
        half-space. A gamma selection must contain exactly one gamma-point k-point.

        :param spins: The zero-based spin indices to retain, or all spins when ``None``.
        :param kpts: The zero-based k-point indices to retain, or all k-points when ``None``.
        :param bands: The zero-based band indices to retain, or all bands when ``None``.
        :param format: The requested coefficient format, ``"std"``, ``"gamma"``, or ``None``.
        :param gamma_half: The target gamma-compression half-space.
        :return: A new in-memory collection containing the selected data.
        :raises ValueError: If indices, format, gamma selection, or half-space conversion is invalid.
        """
        import numpy

        spin_indices = numpy.arange(self._nspins) if spins is None else _validate_indices(spins, self._nspins, "spins")
        kpt_indices = numpy.arange(self._nkpts) if kpts is None else _validate_indices(kpts, self._nkpts, "kpts")
        band_indices = numpy.arange(self._nbands) if bands is None else _validate_indices(bands, self._nbands, "bands")
        if format not in (None, "std", "gamma"):
            raise ValueError("format must be None, 'std', or 'gamma'")
        target_gamma = self._is_gamma if format is None else format == "gamma"
        target_half = self._gamma_half if target_gamma and format is None else gamma_half
        if target_gamma and target_half not in ("x", "z"):
            raise ValueError("gamma_half must be 'x' or 'z'")
        if target_gamma and (
            len(kpt_indices) != 1 or not numpy.array_equal(self._kpoints[kpt_indices[0]], numpy.zeros(3))
        ):
            raise ValueError("gamma format requires one gamma-point k-point")
        converting = target_gamma != self._is_gamma
        if target_gamma and self._is_gamma and target_half != self._gamma_half:
            raise ValueError("changing the gamma-half scheme requires a standard wavefunction")
        new_coeffs: dict[tuple[int, int, int], Any] = {}
        target_counts = []
        for new_k, old_k in enumerate(kpt_indices):
            old_g = self.gvectors(int(old_k))
            if converting:
                target_g = self.gvectors(int(old_k), gamma=target_gamma, gamma_half=target_half)
            else:
                target_g = old_g
            target_counts.append(len(target_g))
            for new_s, old_s in enumerate(spin_indices):
                for new_b, old_b in enumerate(band_indices):
                    values = self.coefficients(int(old_s), int(old_k), int(old_b), cache=False)
                    if converting and target_gamma:
                        values = _reduce_std_coeffs(values, self._kgrid_size, old_g, target_g, str(target_half))
                    elif converting:
                        values = _expand_gamma_coeffs(values, self.gvectors(int(old_k), gamma=False), old_g)
                    dtype = numpy.complex128 if self._double_precision else numpy.complex64
                    new_coeffs[(new_s, new_k, new_b)] = numpy.asarray(values, dtype=dtype)
        selected = type(self)(
            cell=self._cell,
            encut=self._encut,
            kpoints=self._kpoints[kpt_indices],
            eigenvalues=self._eigenvalues[numpy.ix_(spin_indices, kpt_indices, band_indices)],
            occupations=self._occupations[numpy.ix_(spin_indices, kpt_indices, band_indices)],
            coefficients=new_coeffs,
            nplanewaves=numpy.asarray(target_counts, dtype=numpy.int64),
            double_precision=self._double_precision,
            gamma_half=target_half if target_gamma else None,
        )
        return selected


def wavefunction_overlap(phi1: Any, phi2: Any) -> complex:
    """Return the complex overlap of two wavefunctions.

    :param phi1: The first wavefunction.
    :param phi2: The second wavefunction.
    :return: The conjugate-inner-product overlap.
    :raises ValueError: If the wavefunctions do not have matching shapes.
    """
    import numpy

    first, second = numpy.asarray(phi1), numpy.asarray(phi2)
    if first.shape != second.shape:
        raise ValueError("wavefunctions must have the same shape")
    return complex(numpy.sum(numpy.conjugate(first) * second))


class _PlaneWaveSource:
    def __init__(self, wave: PlaneWaveFunctions) -> None:
        import numpy

        self.wave = wave
        self.nspins, self.nkpts, self.nbands = wave.nspins, wave.nkpts, wave.nbands
        self.encut = wave.encut
        self.cell = numpy.asarray(wave.cell.basis.to_floats(), dtype=numpy.float64)
        self.kpoints = wave.kpoints
        self.eigenvalues = wave.eigenvalues
        self.occupations = wave.occupations
        self.nplanewaves = wave.nplanewaves
        self.double_precision = wave.double_precision

    def coefficients(self, spin: int, kpt: int, band: int) -> Any:
        return self._wave.coefficients(spin, kpt, band)

    @property
    def _wave(self) -> PlaneWaveFunctions:
        return self.wave


def _wavecar_payload_from_planewaves(obj: Any) -> Mapping[str, Any]:
    if not isinstance(obj, PlaneWaveFunctions):
        raise TypeError("vasp-wavecar serializer expects PlaneWaveFunctions")
    return {"format": "vasp-wavecar", "wavecar": _PlaneWaveSource(obj)}


def _planewaves_from_payload(payload: Mapping[str, Any]) -> PlaneWaveFunctions:
    return PlaneWaveFunctions(payload, gamma_half=payload.get("gamma_half"))


def save_vesta(basename: str, structure: Any, wave: Any, *, cols: int = 10) -> None:
    """Save real and imaginary wave components as VASP volumetric files.

    The files are written as ``<basename>_r.vasp`` and ``<basename>_i.vasp``.

    :param basename: The output filename prefix.
    :param structure: The structure supplying the volumetric-file cell and species metadata.
    :param wave: The three-dimensional complex wave to write.
    :param cols: The number of values written per output line.
    :raises ImportError: If NumPy is not installed.
    :raises ValueError: If ``wave`` is not a three-dimensional complex array.
    """
    require_numpy()
    import numpy
    from httk.core.register import format_serializers

    from httk.atomistic.integrations.vasp.io.volumetric import write_vasp_volumetric
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    values = numpy.asarray(wave)
    if values.ndim != 3 or not numpy.iscomplexobj(values):
        raise ValueError("wave must be a three-dimensional complex array")
    payload = format_serializers.dispatch("vasp-poscar", UnitcellStructureView(structure))
    write_vasp_volumetric(f"{basename}_r.vasp", payload, numpy.real(values), cols=cols)
    write_vasp_volumetric(f"{basename}_i.vasp", payload, numpy.imag(values), cols=cols)
