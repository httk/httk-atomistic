"""Lazy VASP POSCAR structures backed by neutral httk payloads."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import httk.core
from httk.core.register import format_serializers

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.view import StructureView


class VASPStructure(StructureBackend):
    """A lazy VASP POSCAR structure.

    This backend is explicitly constructed because a generic structure source should not
    silently claim every POSCAR path without the optional httk-io reader.
    """

    kind: ClassVar[str] = "vasp"
    _source: Any
    _payload: Mapping[str, Any] | None
    _resolved: UnitcellStructure | None
    _vasp_initialized: bool

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints.get("kind", cls.kind) != cls.kind:
            return None
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, StructureView):
            unwrapped = httk.core.unwrap(obj)
            if isinstance(unwrapped, cls):
                return unwrapped
            backend = getattr(obj, "_backend", None)
            if isinstance(backend, cls):
                return backend
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        if getattr(self, "_vasp_initialized", False):
            return

        self._source = obj
        self._payload = None
        self._resolved = None

        if isinstance(obj, (str, os.PathLike)):
            name = os.fsdecode(os.fspath(obj))
            if not Path(name).exists():
                raise FileNotFoundError(f"VASP structure source does not exist: {name!r}")
            if not httk.core.has_reader_for(name):
                raise ImportError(
                    "VASPStructure requires the POSCAR reader provided by httk-io; "
                    "install httk-io to load POSCAR files."
                )
        elif isinstance(obj, Mapping):
            if obj.get("format") != "vasp-poscar":
                raise ValueError("VASPStructure payload must have format 'vasp-poscar'.")
            self._payload = obj

        self._vasp_initialized = True

    @property
    def payload(self) -> Mapping[str, Any]:
        """Return the original, loaded, or synthesized neutral POSCAR payload."""
        if self._payload is None:
            if isinstance(self._source, (str, os.PathLike)):
                self._payload = httk.core.load(os.fsdecode(os.fspath(self._source)), raw=True)
            else:
                payload = format_serializers.dispatch("vasp-poscar", self._source)
                if payload is None:
                    raise TypeError("VASP POSCAR serializer returned no payload")
                self._payload = payload
        assert self._payload is not None
        return self._payload

    @property
    def comment(self) -> Any:
        return self.payload.get("comment")

    @property
    def selective_dynamics(self) -> Any:
        return self.payload.get("selective_dynamics")

    def resolve(self) -> UnitcellStructure:
        """Build and memoize the canonical structure from the POSCAR payload."""
        if self._resolved is None:
            from httk.atomistic._loading import _structure_from_poscar

            self._resolved = _structure_from_poscar(self.payload)
        return self._resolved

    @property
    def cell(self) -> Cell:
        return self.resolve().cell

    @property
    def sites(self) -> Sites:
        return self.resolve().sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self.resolve().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self.resolve().species_at_sites

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        return self.resolve().site_moments

    @property
    def charge(self) -> Any:
        return self.resolve().charge

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def unwrap(self) -> Any:
        return self._source
