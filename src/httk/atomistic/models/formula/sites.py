"""Formula backend counting the sites each species class occupies."""

from fractions import Fraction
from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

import httk.atomistic.models.bareprotostructure.backend
import httk.atomistic.models.bareprotostructure.view_base
import httk.atomistic.models.bareprototype.backend
import httk.atomistic.models.bareprototype.view_base
import httk.atomistic.models.protonetstructure.backend
import httk.atomistic.models.protonetstructure.view_base
import httk.atomistic.models.protonettype.backend
import httk.atomistic.models.protonettype.view_base
import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.view_base
import httk.atomistic.models.prototype.backend
import httk.atomistic.models.prototype.view_base
from httk.atomistic.composition import _site_data
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.formula.structure import StructureComposition
from httk.atomistic.models.formula.wyckoff import WyckoffComposition
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.prototype.notation import _classes
from httk.atomistic.models.species.classes import _species_class
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.models.structuretype.backend import StructuretypeBackend
from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

_WYCKOFF_BACKENDS = (
    httk.atomistic.models.prototype.backend.PrototypeBackend,
    httk.atomistic.models.bareprototype.backend.BarePrototypeBackend,
    httk.atomistic.models.protostructure.backend.ProtostructureBackend,
    httk.atomistic.models.bareprotostructure.backend.BareProtostructureBackend,
    httk.atomistic.models.protonettype.backend.ProtonettypeBackend,
    httk.atomistic.models.protonetstructure.backend.ProtonetstructureBackend,
)
_VIEWS = (
    StructureView,
    StructuretypeViewBase,
    httk.atomistic.models.prototype.view_base.PrototypeViewBase,
    httk.atomistic.models.bareprototype.view_base.BarePrototypeViewBase,
    httk.atomistic.models.protostructure.view_base.ProtostructureViewBase,
    httk.atomistic.models.bareprotostructure.view_base.BareProtostructureViewBase,
    httk.atomistic.models.protonettype.view_base.ProtonettypeViewBase,
    httk.atomistic.models.protonetstructure.view_base.ProtonetstructureViewBase,
)


class SiteCountFormula(ChemicalFormulaBackend):
    r"""Count the sites each species class of a structural object occupies.

    A Wyckoff-family object's classes are its label classes (occupation labels, or
    species class names once assigned), counted by summed conventional Wyckoff multiplicity
    exactly as its label prefix counts them. A structure's or structuretype's classes are
    its species classes (equal chemistry, whatever the names; a species with ``X`` is its own
    class), counted over the sites of the represented cell. Disorder and
    partial occupancy do not affect the counts. This backend is not part of formula
    dispatch; :class:`~httk.atomistic.FormulatypeView` selects it for structural sources.

    :param obj: The structure, structuretype, prototype or protostructure to count.
    :param \*\*hints: Backend-selection hints.
    """

    _source: Any

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a structural source.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "sites") != "sites":
            return None
        if isinstance(obj, _VIEWS):
            return cls(obj, **hints) if getattr(obj, "_backend", None) is not None else None
        if isinstance(obj, (StructureBackend, StructuretypeBackend, *_WYCKOFF_BACKENDS)):
            return cls(obj, **hints)
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._source = obj._backend if isinstance(obj, _VIEWS) else obj

    @cached_property
    def _amounts(self) -> tuple[tuple[str, Fraction], ...]:
        source = self._source
        if isinstance(source, _WYCKOFF_BACKENDS):
            occupations = [
                (o.wyckoff, o.species.name if isinstance(o, WyckoffOccupation) else o.label) for o in source.occupations
            ]
            counts = [Fraction(multiplicity) for _, multiplicity in _classes(source.spacegroup, occupations).values()]
        else:
            names, factors, species = _site_data(source)
            by_name = {value.name: value for value in species}
            classes = {name: _species_class(by_name[name]) for name in set(names) if name in by_name}
            by_class: dict[Any, Fraction] = {}
            for name, factor in zip(names, factors):
                key = classes.get(name, name)
                by_class[key] = by_class.get(key, Fraction()) + factor
            counts = list(by_class.values())
        return tuple((anonymous_symbol(index), count) for index, count in enumerate(sorted(counts, reverse=True)))

    @property
    def is_anonymous(self) -> bool:
        """Return whether the counts use anonymous labels."""
        return True

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the per-cell class site counts, largest first, under anonymous labels, not gcd-reduced."""
        return self._amounts

    def unwrap(self) -> Any:
        """Return the raw structural object behind the counts.

        :return: The unwrapped source object.
        """
        return unwrap(self._source)


def composition_source(backend: ChemicalFormulaBackend) -> Any | None:
    """Return the structural object a composition backend projects, if any.

    :param backend: The chemical-formula backend to inspect.
    :return: The structure or Wyckoff-family backend behind a structure or Wyckoff
        composition, or ``None`` for any other backend.
    """
    if isinstance(backend, StructureComposition):
        return backend._structure
    if isinstance(backend, WyckoffComposition):
        return backend._prototype
    return None
