"""The structure/protostructure-to-prototemplate erasure adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
from httk.atomistic.models.prototemplate.notation import canonical_label_map
from httk.atomistic.models.prototemplate.occupation import PrototemplateOccupation
from httk.atomistic.models.prototemplate.prototemplate import Prototemplate
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView


class DerivedPrototemplate(PrototemplateBackend):
    r"""Erase a protostructure or structure lazily to an anonymous prototemplate.

    A protostructure-family source has its real species erased to anonymous classes by the
    pinned group-ordering rule. A crystaltemplate-family (fundamental-domain) or
    structure-family source is routed through
    :class:`~httk.atomistic.models.crystaltemplate.fundamental_view.FundamentalDomainTemplateView`
    recognition and then discretized to its Wyckoff letters and anonymous classes. An exact
    fundamental-domain source needs no spglib; a raw structure source does.

    :param obj: The protostructure-like or structure-like source to erase.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "prototemplate"
    _source: Any

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a protostructure-like or structure-like erasure source.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "prototemplate") != "prototemplate":
            return None
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return cls(obj, **hints)
        # A prototype exposes its prototemplate directly; a structuretype exposes its erased one.
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        if isinstance(obj, (PrototypeBackend, PrototypeViewBase, StructuretypeBackend, StructuretypeViewBase)):
            return cls(obj, **hints)
        from httk.atomistic.models.crystaltemplate.backend import CrystalTemplateBackend
        from httk.atomistic.models.crystaltemplate.view_base import CrystalTemplateViewBase

        if isinstance(obj, (CrystalTemplateBackend, CrystalTemplateViewBase, StructureView, StructureBackend)):
            return cls(obj, **hints)
        source_hints = {name: value for name, value in hints.items() if name != "kind"}
        try:
            StructureBackend._select_backend(obj, **source_hints)
        except TypeError as exc:
            if str(exc) == f"Cannot represent {type(obj)} as StructureBackend":
                return None
            raise
        return cls(obj, **hints)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._source = obj

    @cached_property
    def _derived(self) -> Prototemplate:
        source = self._source
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        if isinstance(source, (PrototypeBackend, PrototypeViewBase, StructuretypeBackend, StructuretypeViewBase)):
            api = source._backend if isinstance(source, (PrototypeViewBase, StructuretypeViewBase)) else source
            folded = api.prototemplate
            return folded if type(folded) is Prototemplate else Prototemplate(folded.spacegroup, folded.occupations)
        if isinstance(source, (ProtostructureBackend, ProtostructureViewBase)):
            proto: ProtostructureBackend = source._backend if isinstance(source, ProtostructureViewBase) else source
            letters_by_name: dict[str, list[str]] = {}
            for occupation in proto.occupations:
                letters_by_name.setdefault(occupation.species.name, []).append(occupation.wyckoff)
            relabel = canonical_label_map({name: tuple(sorted(letters)) for name, letters in letters_by_name.items()})
            occupations = [
                PrototemplateOccupation(occupation.wyckoff, relabel[occupation.species.name])
                for occupation in proto.occupations
            ]
            return Prototemplate(proto.spacegroup, occupations)

        from httk.atomistic.models.crystaltemplate.fundamental_view import FundamentalDomainTemplateView

        template = (
            source if isinstance(source, FundamentalDomainTemplateView) else FundamentalDomainTemplateView(source)
        )
        occupations = [PrototemplateOccupation(site.wyckoff, site.species) for site in template.wyckoff_sites]
        return Prototemplate(template.spacegroup, occupations)

    def resolve(self) -> Prototemplate:
        """Return the complete erased prototemplate."""
        return self._derived

    @property
    def spacegroup(self):
        """Return the erased standard-setting space group."""
        return self._derived.spacegroup

    @property
    def occupations(self) -> tuple[PrototemplateOccupation, ...]:
        """Return the erased class-partitioned occupations."""
        return self._derived.occupations

    def unwrap(self) -> Any:
        """Return the original erasure source."""
        return unwrap(self._source)
