"""The structure-to-prototype recognition adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.crystaltemplate.backend import CrystalTemplateBackend
from httk.atomistic.models.crystaltemplate.view_base import CrystalTemplateViewBase
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView


class RecognizedPrototype(PrototypeBackend):
    r"""Recognize an ordinary structure or template lazily as a prototype.

    The source is used as the geometrical-class anchor: it is recognized (through
    :class:`~httk.atomistic.models.crystaltemplate.fundamental_view.FundamentalDomainTemplateView`,
    which handles an exact fundamental domain without spglib and a raw structure with it) to a
    standard-setting dummy-species representative, and the anonymous prototemplate folds from
    that representative. The resulting prototype is representative-carrying and has no
    discriminator.

    :param obj: The anonymous-template-like or structure-like source to recognize.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    kind = "structure"
    _source: Any
    _tolerance: float | None
    _limit_denominator: int | None

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a template-like or structure-like class anchor.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "structure") != "structure":
            return None
        from httk.atomistic.models.protostructure.backend import ProtostructureBackend
        from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
        from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
        from httk.atomistic.models.prototemplate.view_base import PrototemplateViewBase

        # A bare prototemplate or protostructure carries no geometry, so it cannot anchor a
        # geometrical class; the chemical-formula family is not a structure source at all.
        if isinstance(obj, (PrototemplateBackend, PrototemplateViewBase)):
            return None
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return None
        if isinstance(obj, (ChemicalFormulaBackend, ChemicalFormulaViewBase)):
            return None
        # A structuretype erases lazily to its anonymous prototype (see _derived).
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        if isinstance(obj, (StructuretypeBackend, StructuretypeViewBase)):
            return cls(obj, **hints)
        if isinstance(obj, (CrystalTemplateBackend, CrystalTemplateViewBase, StructureView, StructureBackend)):
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
    def _derived(self) -> Prototype:
        from httk.atomistic.models.crystaltemplate.fundamental_view import FundamentalDomainTemplateView
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        source = self._source
        if isinstance(source, (StructuretypeBackend, StructuretypeViewBase)):
            # Erase a structuretype: prototemplate from its protostructure, representative
            # anonymized, discriminator carried over (it names the species-independent class).
            from httk.atomistic.models.prototemplate.view import PrototemplateView

            structuretype = source._backend if isinstance(source, StructuretypeViewBase) else source
            prototemplate = PrototemplateView(structuretype.protostructure).unview()
            representative_structure = structuretype.representative
            representative = (
                None
                if representative_structure is None
                else FundamentalDomainTemplateView(representative_structure).unview()
            )
            return Prototype(prototemplate, representative=representative, discriminator=structuretype.discriminator)
        representative = FundamentalDomainTemplateView(
            source, tolerance=self._tolerance, limit_denominator=self._limit_denominator
        ).unview()
        return Prototype(representative.prototemplate, representative=representative)

    def resolve(self) -> Prototype:
        """Return the complete recognized prototype."""
        return self._derived

    @property
    def prototemplate(self):
        """Return the recognized anonymous prototemplate."""
        return self._derived.prototemplate

    @property
    def representative(self):
        """Return the recognized canonical class representative."""
        return self._derived.representative

    @property
    def discriminator(self) -> str | None:
        """Return ``None``; a recognized prototype carries no discriminator."""
        return self._derived.discriminator

    def unwrap(self) -> Any:
        """Return the original recognition source."""
        return unwrap(self._source)
