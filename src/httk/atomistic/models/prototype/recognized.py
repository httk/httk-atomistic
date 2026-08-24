"""The structure-to-prototype recognition adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.models.structuretype.backend import StructuretypeBackend
from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase


class RecognizedPrototype(PrototypeBackend):
    r"""Recognize an ordinary structure or template lazily as a prototype.

    The source is used as the geometrical-class anchor: it is recognized (through
    :class:`~httk.atomistic.models.structuretype.fundamental_view.FundamentalDomainTemplateView`,
    which handles an exact fundamental domain without spglib and a raw structure with it) to a
    standard-setting dummy-species representative, and the anonymous prototype folds from
    that representative. The resulting prototype is representative-carrying and has no
    discriminator.

    :param obj: The structure-like source to recognize.
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

        # A bare protostructure carries no geometry, so it cannot anchor a
        # geometrical class; the chemical-formula family is not a structure source at all.
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return None
        if isinstance(obj, (ChemicalFormulaBackend, ChemicalFormulaViewBase)):
            return None
        if isinstance(obj, (StructuretypeBackend, StructuretypeViewBase, StructureView, StructureBackend)):
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
        from httk.atomistic.models.structuretype.fundamental_view import FundamentalDomainTemplateView

        source = self._source
        # Recognition yields a base prototype. A representative/discriminator is attached only
        # when the user constructs a Prototype with one explicitly, so recognized values compare
        # equal to hand-built and label-parsed ones of the same class.
        template = FundamentalDomainTemplateView(
            source, tolerance=self._tolerance, limit_denominator=self._limit_denominator
        ).unview()
        return Prototype(template.spacegroup, [(site.wyckoff, site.species) for site in template.wyckoff_sites])

    def resolve(self) -> Prototype:
        """Return the complete recognized prototype."""
        return self._derived

    @property
    def spacegroup(self):
        return self._derived.spacegroup

    @property
    def occupations(self):
        return self._derived.occupations

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
