"""The structure-to-prototype recognition adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.bareprotostructure.recognized import RecognizedBareProtostructure
from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.models.structuretype.backend import StructuretypeBackend
from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase


class _AnonymousRecognition(RecognizedBareProtostructure):
    """Recognize a structure's species classes for their anonymous erasure, molecular metadata dropped."""

    _molecular_ok = True
    _view_name = "BarePrototypeView"


class RecognizedBarePrototype(BarePrototypeBackend):
    r"""Recognize an ordinary structure or template lazily as a prototype.

    A structure is recognized as a bare protostructure and erased: the prototype is the
    partition of its occupied Wyckoff positions into species classes, so mixed and partially
    occupied sites are accepted. A structuretype resolves its exact fundamental-domain template
    and keeps its standard-setting Wyckoff occupations and anonymous class partition. The
    original source remains available through ``unwrap()``.

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

        # Assigned classifications use the dedicated erasure adapter.
        # Chemical formulas are not structure sources.
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
    def _derived(self) -> BarePrototype:
        source = self._source
        if not isinstance(source, (StructuretypeBackend, StructuretypeViewBase)):
            # a structure: erase its recognized species classes, so disorder is accepted
            from httk.atomistic.models.prototype.derived import _anonymous_occupations

            hints = {"tolerance": self._tolerance, "limit_denominator": self._limit_denominator}
            assigned = _AnonymousRecognition._backend_adopt(source, **{k: v for k, v in hints.items() if v is not None})
            if assigned is None:
                raise TypeError(f"cannot recognize a bare prototype of {type(source).__name__}")
            return BarePrototype(assigned.spacegroup, _anonymous_occupations(assigned.spacegroup, assigned.occupations))
        from httk.atomistic.models.structuretype.fundamental_view import FundamentalDomainTemplateView

        template = FundamentalDomainTemplateView(
            source, tolerance=self._tolerance, limit_denominator=self._limit_denominator
        ).unview()
        return BarePrototype(template.spacegroup, [(site.wyckoff, site.species) for site in template.wyckoff_sites])

    def resolve(self) -> BarePrototype:
        """Return the complete recognized prototype."""
        return self._derived

    @property
    def spacegroup(self):
        return self._derived.spacegroup

    @property
    def occupations(self):
        return self._derived.occupations

    def unwrap(self) -> Any:
        """Return the original recognition source."""
        return unwrap(self._source)
