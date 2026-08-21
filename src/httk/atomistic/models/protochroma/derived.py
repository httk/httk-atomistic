"""The structure/protostructure-to-protochroma erasure adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protochroma.backend import ProtochromaBackend
from httk.atomistic.models.protochroma.notation import canonical_label_map
from httk.atomistic.models.protochroma.occupation import ProtochromaOccupation
from httk.atomistic.models.protochroma.protochroma import Protochroma
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView


class DerivedProtochroma(ProtochromaBackend):
    r"""Erase a protostructure or structure lazily to an anonymous protochroma.

    A protostructure-family source has its real species erased to anonymous classes by the
    pinned group-ordering rule. A chromastructure-family (fundamental-domain) or
    structure-family source is routed through
    :class:`~httk.atomistic.models.chromastructure.fundamental_view.FundamentalDomainPatternView`
    recognition and then discretized to its Wyckoff letters and anonymous classes. An exact
    fundamental-domain source needs no spglib; a raw structure source does.

    :param obj: The protostructure-like or structure-like source to erase.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "protochroma"
    _source: Any

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a protostructure-like or structure-like erasure source.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "protochroma") != "protochroma":
            return None
        if isinstance(obj, (ProtostructureBackend, ProtostructureViewBase)):
            return cls(obj, **hints)
        # A prototype exposes its protochroma directly; a crystallotype exposes its erased one.
        from httk.atomistic.models.crystallotype.backend import CrystallotypeBackend
        from httk.atomistic.models.crystallotype.view_base import CrystallotypeViewBase
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase

        if isinstance(obj, (PrototypeBackend, PrototypeViewBase, CrystallotypeBackend, CrystallotypeViewBase)):
            return cls(obj, **hints)
        from httk.atomistic.models.chromastructure.backend import ChromastructureBackend
        from httk.atomistic.models.chromastructure.view_base import ChromastructureViewBase

        if isinstance(obj, (ChromastructureBackend, ChromastructureViewBase, StructureView, StructureBackend)):
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
    def _derived(self) -> Protochroma:
        source = self._source
        from httk.atomistic.models.crystallotype.backend import CrystallotypeBackend
        from httk.atomistic.models.crystallotype.view_base import CrystallotypeViewBase
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase

        if isinstance(source, (PrototypeBackend, PrototypeViewBase, CrystallotypeBackend, CrystallotypeViewBase)):
            api = source._backend if isinstance(source, (PrototypeViewBase, CrystallotypeViewBase)) else source
            folded = api.protochroma
            return folded if type(folded) is Protochroma else Protochroma(folded.spacegroup, folded.occupations)
        if isinstance(source, (ProtostructureBackend, ProtostructureViewBase)):
            proto: ProtostructureBackend = source._backend if isinstance(source, ProtostructureViewBase) else source
            letters_by_name: dict[str, list[str]] = {}
            for occupation in proto.occupations:
                letters_by_name.setdefault(occupation.species.name, []).append(occupation.wyckoff)
            relabel = canonical_label_map({name: tuple(sorted(letters)) for name, letters in letters_by_name.items()})
            occupations = [
                ProtochromaOccupation(occupation.wyckoff, relabel[occupation.species.name])
                for occupation in proto.occupations
            ]
            return Protochroma(proto.spacegroup, occupations)

        from httk.atomistic.models.chromastructure.fundamental_view import FundamentalDomainPatternView

        pattern = source if isinstance(source, FundamentalDomainPatternView) else FundamentalDomainPatternView(source)
        occupations = [ProtochromaOccupation(site.wyckoff, site.species) for site in pattern.wyckoff_sites]
        return Protochroma(pattern.spacegroup, occupations)

    def resolve(self) -> Protochroma:
        """Return the complete erased protochroma."""
        return self._derived

    @property
    def spacegroup(self):
        """Return the erased standard-setting space group."""
        return self._derived.spacegroup

    @property
    def occupations(self) -> tuple[ProtochromaOccupation, ...]:
        """Return the erased class-partitioned occupations."""
        return self._derived.occupations

    def unwrap(self) -> Any:
        """Return the original erasure source."""
        return unwrap(self._source)
