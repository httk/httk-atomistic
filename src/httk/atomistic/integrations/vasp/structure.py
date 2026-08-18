"""Lazy VASP POSCAR structures backed by neutral httk payloads."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

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
    r"""Load a VASP POSCAR structure lazily.

    This backend is explicitly constructed because a generic structure source should not
    silently claim every POSCAR path.

    It is not registered in ``backend_classes``. Constructing it from a view whose
    unwrapped value is already a ``VASPStructure`` returns that backend by identity.
    The payload's ``raw`` channel preserves the source representation for byte-exact
    saving.

    :param obj: A POSCAR path, neutral payload, or serializer-supported source.
    :param \**hints: Backend-selection hints.
    """

    kind: ClassVar[str] = "vasp"
    _source: Any
    _payload: Mapping[str, Any] | None
    _resolved: UnitcellStructure | None
    _vasp_initialized: bool

    def __new__(cls, obj: Any, **hints: Any) -> Self:
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

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a VASP structure source.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
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
        return cls(obj, **hints)

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
        """Return the POSCAR comment, if present."""
        return self.payload.get("comment")

    @property
    def selective_dynamics(self) -> Any:
        """Return selective-dynamics flags, if present."""
        return self.payload.get("selective_dynamics")

    def resolve(self) -> UnitcellStructure:
        """Build and memoize the canonical structure from the POSCAR payload.

        :return: The resolved unit-cell structure.
        """
        if self._resolved is None:
            from httk.atomistic._loading import _structure_from_poscar

            self._resolved = _structure_from_poscar(self.payload)
        return self._resolved

    @property
    def cell(self) -> Cell:
        """Return the resolved cell."""
        return self.resolve().cell

    @property
    def sites(self) -> Sites:
        """Return the resolved reduced coordinates."""
        return self.resolve().sites

    @property
    def species(self) -> tuple[Species, ...]:
        """Return the resolved distinct species."""
        return self.resolve().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the resolved species name at each site."""
        return self.resolve().species_at_sites

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        """Return resolved site moments, or ``None``."""
        return self.resolve().site_moments

    @property
    def charge(self) -> Any:
        """Return the resolved structure charge, if present."""
        return self.resolve().charge

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def unwrap(self) -> Any:
        """Return the original POSCAR source."""
        return self._source
