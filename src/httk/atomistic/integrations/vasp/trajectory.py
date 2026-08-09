"""Lazy VASP trajectory access through neutral httk-io payloads.

This backend is explicit-only: a generic trajectory source must not claim an
OUTCAR/XDATCAR path unless the optional httk-io readers are installed.
"""

import os
import re
from collections.abc import Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

import httk.core
from httk.core import SurdVector
from httk.core.datastream import compression

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.backend import TrajectoryBackend

_ENERGY = "_httk_frame_total_energies"
_TEMPERATURE = "_httk_frame_temperatures"
_STRESS = "_httk_frame_stresses"


def _compression_suffixes() -> tuple[str, ...]:
    return tuple(ext for codec in compression._registry.values() for ext in codec.extensions)


def _payload_object(payload: Any, fmt: str, key: str) -> Any:
    if isinstance(payload, Mapping) and payload.get("format") == fmt:
        return payload[key]
    return payload


def _base_name(path: Path) -> str:
    name = path.name
    for suffix in _compression_suffixes():
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)]
            break
    return name.upper()


def _find(directory: Path, name: str) -> Path | None:
    for suffix in ("", *_compression_suffixes()):
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _potcar_symbol(title: str) -> str:
    """Extract the element from the second TITEL token, before any ``_`` suffix."""
    tokens = title.split()
    if len(tokens) < 2:
        raise ValueError(f"cannot extract an element symbol from POTCAR title {title!r}")
    symbol = tokens[1].split("_", 1)[0]
    if re.fullmatch(r"[A-Z][a-z]?", symbol) is None:
        raise ValueError(f"cannot extract an element symbol from POTCAR title {title!r}")
    return symbol


class VASPTrajectory(TrajectoryBackend):
    r"""Read VASP OUTCAR and/or XDATCAR data lazily.

    XDATCAR supplies geometry when present.  OUTCAR observables use the
    per-frame ``energy_sigma0``, parsed as a float, plus temperature and
    ``stress_gpa_voigt()``.  One bounded pass caches those three scalar/6-tuple
    sequences; frame geometry is never cached.

    XDATCAR geometry is preferred when both files are available. Cartesian coordinates
    are reduced exactly against the frame cell. A mismatch between OUTCAR and XDATCAR
    frame counts raises an error.

    :param source: A VASP trajectory path, directory, payload, or VASP-outputs-like object.
    :param \**hints: Backend-selection hints.
    """

    kind: ClassVar[str] = "vasp"

    def __new__(cls, source: Any, **hints: Any) -> Any:
        if hints.get("kind", cls.kind) != cls.kind:
            return None
        if isinstance(source, cls):
            return source
        return super().__new__(cls)

    def __init__(self, source: Any, **hints: Any) -> None:
        if getattr(self, "_vasp_trajectory_initialized", False):
            return
        self._source = source
        self._outcar: Any | None = None
        self._xdatcar: Any | None = None
        self._poscar: Mapping[str, Any] | None = None
        self._sources_ready = False
        self._nframes: int | None = None
        self._header: UnitcellStructure | None = None
        self._observable_cache: dict[str, tuple[Any, ...]] | None = None
        self._observable_names: tuple[str, ...] = ()

        if isinstance(source, Mapping):
            fmt = source.get("format")
            if fmt not in ("vasp-outcar", "vasp-xdatcar"):
                raise ValueError("VASPTrajectory payload must have format 'vasp-outcar' or 'vasp-xdatcar'.")
        elif isinstance(source, (str, os.PathLike)):
            path = Path(os.fsdecode(os.fspath(source)))
            if not path.exists():
                raise FileNotFoundError(f"VASP trajectory source does not exist: {path!s}")
            if path.is_dir():
                found = tuple(
                    candidate for name in ("OUTCAR", "XDATCAR") if (candidate := _find(path, name)) is not None
                )
                if not found:
                    raise FileNotFoundError(
                        "VASP trajectory directory has no OUTCAR or XDATCAR; "
                        "install httk-io to provide the VASP readers."
                    )
                if not any(httk.core.has_reader_for(os.fsdecode(os.fspath(candidate))) for candidate in found):
                    raise ImportError(
                        "VASPTrajectory requires an OUTCAR/XDATCAR reader provided by httk-io; "
                        "install httk-io to load VASP trajectory files."
                    )
            elif _base_name(path) not in {"OUTCAR", "XDATCAR"}:
                raise ValueError("VASP trajectory file must be named OUTCAR or XDATCAR.")
            elif not httk.core.has_reader_for(os.fsdecode(os.fspath(source))):
                raise ImportError(
                    "VASPTrajectory requires the OUTCAR/XDATCAR readers provided by httk-io; "
                    "install httk-io to load VASP trajectory files."
                )
        elif not any(name in dir(source) for name in ("outcar", "xdatcar", "poscar")):
            raise TypeError(
                "VASPTrajectory expects a directory, VASPOutputs-shaped object, or vasp-outcar/vasp-xdatcar payload."
            )
        self._vasp_trajectory_initialized = True

    def _ensure_sources(self) -> None:
        if self._sources_ready:
            return
        source = self._source
        if isinstance(source, Mapping):
            if source.get("format") == "vasp-outcar":
                self._outcar = source.get("outcar")
            else:
                self._xdatcar = source.get("xdatcar")
        elif isinstance(source, (str, os.PathLike)):
            path = Path(os.fsdecode(os.fspath(source)))
            paths = {
                "OUTCAR": _find(path, "OUTCAR") if path.is_dir() else path,
                "XDATCAR": _find(path, "XDATCAR") if path.is_dir() else path,
            }
            if not path.is_dir():
                paths = {_base_name(path): path}
            for name, candidate in paths.items():
                if candidate is None:
                    continue
                if not httk.core.has_reader_for(os.fsdecode(os.fspath(candidate))):
                    continue
                payload = httk.core.load(os.fsdecode(os.fspath(candidate)), raw=True)
                if name == "OUTCAR":
                    self._outcar = _payload_object(payload, "vasp-outcar", "outcar")
                else:
                    self._xdatcar = _payload_object(payload, "vasp-xdatcar", "xdatcar")
            if path.is_dir():
                poscar = _find(path, "POSCAR")
                if poscar is not None and httk.core.has_reader_for(os.fsdecode(os.fspath(poscar))):
                    self._poscar = httk.core.load(os.fsdecode(os.fspath(poscar)), raw=True)
        else:
            self._outcar = _payload_object(getattr(source, "outcar", None), "vasp-outcar", "outcar")
            self._xdatcar = _payload_object(getattr(source, "xdatcar", None), "vasp-xdatcar", "xdatcar")
            poscar = getattr(source, "poscar", None)
            if isinstance(poscar, Mapping):
                self._poscar = poscar

        if self._outcar is None and self._xdatcar is None:
            raise ValueError("VASPTrajectory source has neither an OUTCAR nor an XDATCAR.")
        self._sources_ready = True

    def _ensure_counts(self) -> None:
        self._ensure_sources()
        if self._nframes is not None:
            return
        xdat_count = None if self._xdatcar is None else self._xdatcar.nframes
        outcar_count = None if self._outcar is None else self._outcar.nframes
        if xdat_count is not None and outcar_count is not None and xdat_count != outcar_count:
            raise ValueError(f"VASP trajectory frame-count mismatch: XDATCAR={xdat_count}, OUTCAR={outcar_count}.")
        count = xdat_count if xdat_count is not None else outcar_count
        if count is None or count < 1:
            raise ValueError("VASP trajectory contains no complete frames.")
        self._nframes = count

    @property
    def nframes(self) -> int:
        """Return the validated number of frames."""
        self._ensure_counts()
        assert self._nframes is not None
        return self._nframes

    def _header_structure(self) -> UnitcellStructure:
        if self._header is not None:
            return self._header
        self._ensure_sources()
        if self._poscar is None:
            if self._xdatcar is not None:
                symbols = self._xdatcar.symbols
                if symbols is None:
                    raise ValueError(
                        "VASPTrajectory needs a VASP-5 POSCAR or XDATCAR species line; "
                        "VASP-4 data has no species symbols."
                    )
                cell = self._xdatcar.cell
                scale = self._xdatcar.scale
                volume = None
                if float(scale) < 0:
                    volume = scale.removeprefix("-")
                    scale = None
                counts = self._xdatcar.counts
            elif self._outcar is not None:
                ions_per_type = getattr(self._outcar, "ions_per_type", None)
                titles = tuple(getattr(self._outcar, "potcar_titles", ()))
                if ions_per_type is None:
                    raise ValueError("standalone OUTCAR composition requires httk-io OutcarFile.ions_per_type")
                if len(ions_per_type) != len(titles):
                    raise ValueError(
                        "standalone OUTCAR ions_per_type and potcar_titles disagree in length: "
                        f"{len(ions_per_type)} != {len(titles)}"
                    )
                symbols = tuple(_potcar_symbol(title) for title in titles)
                counts = tuple(ions_per_type)
                first = next(self._outcar.frames(), None)
                if first is None or first.cell is None:
                    raise ValueError("standalone OUTCAR has no frame cell for its composition")
                cell = first.cell
                scale = "1"
                volume = None
            else:
                raise ValueError(
                    "VASPTrajectory needs a VASP-5 POSCAR or XDATCAR species line; VASP-4 data has no species symbols."
                )
            self._poscar = {
                "format": "vasp-poscar",
                "cell": cell,
                "scale": scale,
                "volume": volume,
                "cartesian": False,
                "coords": tuple(("0", "0", "0") for _ in range(sum(counts))),
                "symbols": symbols,
                "counts": counts,
            }
        from httk.atomistic._loading import _structure_from_poscar

        self._header = _structure_from_poscar(self._poscar)
        return self._header

    def _xdatcar_cell(self, rows: Any, frame_scale: Any = None) -> Cell:
        self._ensure_sources()
        assert self._xdatcar is not None
        scale = self._xdatcar.scale if frame_scale is None else frame_scale
        if float(scale) >= 0:
            return Cell(rows, scale)
        header = self._header_structure()
        from httk.atomistic._loading import _structure_from_poscar

        symbols = tuple(species.chemical_symbols[0] for species in header.species)
        counts = tuple(header.species_at_sites.count(species.name) for species in header.species)
        return _structure_from_poscar(
            {
                "format": "vasp-poscar",
                "cell": rows,
                "scale": None,
                "volume": scale.removeprefix("-"),
                "cartesian": False,
                "coords": tuple(("0", "0", "0") for _ in header.species_at_sites),
                "symbols": symbols,
                "counts": counts,
            }
        ).cell

    @property
    def species(self) -> tuple[Any, ...]:
        """Return the composition inferred from POSCAR, XDATCAR, or OUTCAR."""
        return self._header_structure().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the species name at each site."""
        return self._header_structure().species_at_sites

    @property
    def reference_frames(self) -> None:
        """Return ``None`` because VASP frames are not bounded references."""
        return None

    def _from_xdatcar(self, frame: Mapping[str, Any]) -> UnitcellStructure:
        self._ensure_sources()
        assert self._xdatcar is not None
        header = self._header_structure()
        raw_rows = frame["cell"] or self._xdatcar.cell
        cell = self._xdatcar_cell(raw_rows, frame.get("scale"))
        coords = frame["coords"]
        if frame.get("cartesian", False):
            coords = SurdVector(coords) * SurdVector(raw_rows).inv()
        return UnitcellStructure(cell, Sites(coords), header.species, header.species_at_sites)

    def _from_outcar(self, frame: Any) -> UnitcellStructure:
        header = self._header_structure()
        if frame.positions is None:
            raise ValueError(f"OUTCAR frame {frame.index} has no Cartesian positions.")
        cell = header.cell if frame.cell is None else Cell(frame.cell)
        reduced = SurdVector(frame.positions) * cell.basis.inv()
        return UnitcellStructure(cell, Sites(reduced), header.species, header.species_at_sites)

    def frame(self, i: int) -> UnitcellStructure:
        """Read one VASP frame by index.

        :param i: Frame index; negative indexes count from the end.
        :return: The requested unit-cell structure.
        :raises IndexError: If the frame index is out of range.
        :raises ValueError: If the source has no complete frame geometry.
        """
        count = self.nframes
        index = i if i >= 0 else count + i
        if index < 0 or index >= count:
            raise IndexError(f"trajectory frame index out of range: {i}")
        if self._xdatcar is not None:
            self._ensure_sources()
            item = next(islice(self._xdatcar.frames(), index, index + 1), None)
            if item is None:
                raise IndexError(f"trajectory frame index out of range: {i}")
            return self._from_xdatcar(item)
        assert self._outcar is not None
        item = next(islice(self._outcar.frames(), index, index + 1), None)
        if item is None:
            raise IndexError(f"trajectory frame index out of range: {i}")
        return self._from_outcar(item)

    def frames(self) -> Iterator[UnitcellStructure]:
        """Stream VASP frame geometry without caching full frames.

        :yields: Unit-cell structures in source order.
        """
        self._ensure_counts()
        if self._xdatcar is not None:
            yield from (self._from_xdatcar(frame) for frame in self._xdatcar.frames())
        else:
            assert self._outcar is not None
            yield from (self._from_outcar(frame) for frame in self._outcar.frames())

    def _ensure_observables(self) -> None:
        if self._observable_cache is not None:
            return
        self._ensure_counts()
        if self._outcar is None:
            self._observable_cache = {}
            return
        energies: list[float | None] = []
        temperatures: list[float | None] = []
        stresses: list[tuple[float, ...] | None] = []
        for frame in self._outcar.frames():
            energies.append(None if frame.energy_sigma0 is None else float(frame.energy_sigma0))
            temperatures.append(None if frame.temperature is None else float(frame.temperature))
            stresses.append(frame.stress_gpa_voigt())
        self._observable_cache = {
            _ENERGY: tuple(energies),
            _TEMPERATURE: tuple(temperatures),
            _STRESS: tuple(stresses),
        }
        self._observable_names = tuple(
            name for name, values in self._observable_cache.items() if any(value is not None for value in values)
        )

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return available OUTCAR observable names."""
        self._ensure_observables()
        return self._observable_names

    def observable(self, name: str) -> tuple[Any, ...]:
        """Return one OUTCAR observable in frame order.

        :param name: Observable name.
        :return: The observable values.
        :raises KeyError: If the observable is unavailable.
        """
        self._ensure_observables()
        if name not in self._observable_names:
            raise KeyError(name)
        assert self._observable_cache is not None
        return self._observable_cache[name]

    def unwrap(self) -> Any:
        """Return the original VASP trajectory source."""
        return self._source

    @property
    def source_locator(self) -> str | None:
        """Return the source path, if one is available."""
        if isinstance(self._source, os.PathLike | str):
            return os.fsdecode(os.fspath(self._source))
        self._ensure_sources()
        for payload in (self._outcar, self._xdatcar):
            path = getattr(payload, "path", None)
            if isinstance(path, str):
                return path
        return None
