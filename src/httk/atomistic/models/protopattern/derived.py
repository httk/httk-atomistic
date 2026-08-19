"""The structure/protostructure-to-protopattern erasure adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protopattern.backend import ProtopatternBackend
from httk.atomistic.models.protopattern.notation import canonical_label_map
from httk.atomistic.models.protopattern.occupation import ProtopatternOccupation
from httk.atomistic.models.protopattern.protopattern import Protopattern
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView


class DerivedProtopattern(ProtopatternBackend):
    r"""Erase a protostructure or structure lazily to an anonymous protopattern.

    A protostructure-family source has its real species erased to anonymous classes by the
    pinned group-ordering rule. A crystalpattern-family (fundamental-domain) or
    structure-family source is routed through
    :class:`~httk.atomistic.models.crystalpattern.fundamental_view.FundamentalDomainPatternView`
    recognition and then discretized to its Wyckoff letters and anonymous classes. An exact
    fundamental-domain source needs no spglib; a raw structure source does.

    :param obj: The protostructure-like or structure-like source to erase.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "protopattern"
    _source: Any

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a protostructure-like or structure-like erasure source.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "protopattern") != "protopattern":
            return None
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return cls(obj, **hints)
        # A prototype exposes its protopattern directly; a structuretype exposes its erased one.
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        if isinstance(obj, (PrototypeBackend, PrototypeViewBase, StructuretypeBackend, StructuretypeViewBase)):
            return cls(obj, **hints)
        from httk.atomistic.models.crystalpattern.backend import CrystalPatternBackend
        from httk.atomistic.models.crystalpattern.view_base import CrystalPatternViewBase

        if isinstance(obj, (CrystalPatternBackend, CrystalPatternViewBase, StructureView, StructureBackend)):
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
    def _derived(self) -> Protopattern:
        source = self._source
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        if isinstance(source, (PrototypeBackend, PrototypeViewBase, StructuretypeBackend, StructuretypeViewBase)):
            api = source._backend if isinstance(source, (PrototypeViewBase, StructuretypeViewBase)) else source
            folded = api.protopattern
            return folded if type(folded) is Protopattern else Protopattern(folded.spacegroup, folded.occupations)
        if isinstance(source, (ProtostructureBackend, ProtostructureViewBase)):
            proto: ProtostructureBackend = source._backend if isinstance(source, ProtostructureViewBase) else source
            letters_by_name: dict[str, list[str]] = {}
            for occupation in proto.occupations:
                letters_by_name.setdefault(occupation.species.name, []).append(occupation.wyckoff)
            relabel = canonical_label_map({name: tuple(sorted(letters)) for name, letters in letters_by_name.items()})
            occupations = [
                ProtopatternOccupation(occupation.wyckoff, relabel[occupation.species.name])
                for occupation in proto.occupations
            ]
            return Protopattern(proto.spacegroup, occupations)

        from httk.atomistic.models.crystalpattern.fundamental_view import FundamentalDomainPatternView

        pattern = source if isinstance(source, FundamentalDomainPatternView) else FundamentalDomainPatternView(source)
        occupations = [ProtopatternOccupation(site.wyckoff, site.species) for site in pattern.wyckoff_sites]
        return Protopattern(pattern.spacegroup, occupations)

    def resolve(self) -> Protopattern:
        """Return the complete erased protopattern."""
        return self._derived

    @property
    def spacegroup(self):
        """Return the erased standard-setting space group."""
        return self._derived.spacegroup

    @property
    def occupations(self) -> tuple[ProtopatternOccupation, ...]:
        """Return the erased class-partitioned occupations."""
        return self._derived.occupations

    def unwrap(self) -> Any:
        """Return the original erasure source."""
        return unwrap(self._source)
