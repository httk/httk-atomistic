"""The structure-to-crystallotype recognition adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.chromastructure.backend import ChromastructureBackend
from httk.atomistic.models.chromastructure.view_base import ChromastructureViewBase
from httk.atomistic.models.crystallotype.backend import CrystallotypeBackend
from httk.atomistic.models.crystallotype.crystallotype import Crystallotype
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView


class RecognizedCrystallotype(CrystallotypeBackend):
    r"""Recognize an ordinary structure lazily as a crystallotype.

    The source is used as the geometrical-class anchor: it is standardized to its IT
    standard-setting fundamental domain (through
    :func:`~httk.atomistic.symmetry.standardization.conventional_cell`, which the chromastructure
    view also routes through — an exact fundamental domain without spglib and a raw
    structure with it) and kept as the real-species representative. The protostructure derives
    from that representative, so the resulting crystallotype is representative-carrying and has
    no discriminator.

    :param obj: The structure-like source to recognize.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    kind = "structure"
    _source: Any
    _tolerance: float | None
    _limit_denominator: int | None

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a structure-like class anchor.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "structure") != "structure":
            return None
        from httk.atomistic.models.protochroma.backend import ProtochromaBackend
        from httk.atomistic.models.protochroma.view_base import ProtochromaViewBase
        from httk.atomistic.models.protostructure.backend import ProtostructureBackend
        from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase

        # A dummy-species chromastructure names no real species; a bare protochroma or protostructure has
        # no geometry to anchor a class; the chemical-formula family is not a structure at all.
        if isinstance(obj, (ChromastructureBackend, ChromastructureViewBase)):
            return None
        if isinstance(obj, (ProtochromaBackend, ProtochromaViewBase)):
            return None
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return None
        if isinstance(obj, (ChemicalFormulaBackend, ChemicalFormulaViewBase)):
            return None
        if isinstance(obj, (StructureView, StructureBackend)):
            return cls(obj, **hints)
        source_hints = {
            name: value for name, value in hints.items() if name not in ("kind", "tolerance", "limit_denominator")
        }
        try:
            StructureBackend._select_backend(obj, **source_hints)
        except TypeError as exc:
            if str(exc) == f"Cannot represent {type(obj)} as StructureBackend":
                return None
            raise
        return cls(obj, **hints)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._source = obj
        self._tolerance = hints.get("tolerance")
        self._limit_denominator = hints.get("limit_denominator")

    @cached_property
    def _derived(self) -> Crystallotype:
        from httk.atomistic.symmetry.standardization import conventional_cell

        representative = conventional_cell(
            self._source, tolerance=self._tolerance, limit_denominator=self._limit_denominator
        ).asu
        return Crystallotype(representative=representative)

    def resolve(self) -> Crystallotype:
        """Return the complete recognized crystallotype."""
        return self._derived

    @property
    def protostructure(self):
        """Return the recognized geometry-free protostructure."""
        return self._derived.protostructure

    @property
    def representative(self):
        """Return the recognized canonical class representative."""
        return self._derived.representative

    @property
    def discriminator(self) -> str | None:
        """Return ``None``; a recognized crystallotype carries no discriminator."""
        return self._derived.discriminator

    def unwrap(self) -> Any:
        """Return the original recognition source."""
        return unwrap(self._source)
