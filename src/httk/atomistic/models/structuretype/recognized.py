"""The structure-to-structuretype recognition adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.crystalpattern.backend import CrystalPatternBackend
from httk.atomistic.models.crystalpattern.view_base import CrystalPatternViewBase
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.models.structuretype.backend import StructuretypeBackend
from httk.atomistic.models.structuretype.structuretype import Structuretype


class RecognizedStructuretype(StructuretypeBackend):
    r"""Recognize an ordinary structure lazily as a structuretype.

    The source is used as the geometrical-class anchor: it is standardized to its IT
    standard-setting fundamental domain (through
    :func:`~httk.atomistic.symmetry.standardization.conventional_cell`, which the crystalpattern
    pattern-view also routes through — an exact fundamental domain without spglib and a raw
    structure with it) and kept as the real-species representative. The protostructure derives
    from that representative, so the resulting structuretype is representative-carrying and has
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
        from httk.atomistic.models.protopattern.backend import ProtopatternBackend
        from httk.atomistic.models.protopattern.view_base import ProtopatternViewBase
        from httk.atomistic.models.protostructure.backend import ProtostructureBackend
        from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase

        # A dummy-species pattern names no real species; a bare pattern or protostructure has
        # no geometry to anchor a class; the chemical-formula family is not a structure at all.
        if isinstance(obj, (CrystalPatternBackend, CrystalPatternViewBase)):
            return None
        if isinstance(obj, (ProtopatternBackend, ProtopatternViewBase)):
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
    def _derived(self) -> Structuretype:
        from httk.atomistic.symmetry.standardization import conventional_cell

        representative = conventional_cell(
            self._source, tolerance=self._tolerance, limit_denominator=self._limit_denominator
        ).asu
        return Structuretype(representative=representative)

    def resolve(self) -> Structuretype:
        """Return the complete recognized structuretype."""
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
        """Return ``None``; a recognized structuretype carries no discriminator."""
        return self._derived.discriminator

    def unwrap(self) -> Any:
        """Return the original recognition source."""
        return unwrap(self._source)
