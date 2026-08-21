"""Formula bridge for Wyckoff-multiplicity-based backends."""

from collections import Counter
from fractions import Fraction
from functools import cached_property
from types import SimpleNamespace
from typing import Any, Self

from httk.core import unwrap

import httk.atomistic.models.protochroma.backend
import httk.atomistic.models.protochroma.view_base
import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.view_base
from httk.atomistic.composition import project_composition
from httk.atomistic.models.chromastructure.backend import ChromastructureBackend
from httk.atomistic.models.chromastructure.view_base import ChromastructureViewBase
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.notation import anonymous_symbol


class WyckoffComposition(ChemicalFormulaBackend):
    r"""Represent the canonical composition of a Wyckoff-multiplicity-based backend.

    Chromastructure and protochroma inputs use anonymous labels; protostructure
    inputs retain their real elemental composition at the standard conventional-cell scale.

    :param obj: The chromastructure or protostructure to present.
    :param \*\*hints: Backend-selection hints.
    """

    _prototype: Any
    kind = "wyckoff"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a Wyckoff-multiplicity-based backend.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "wyckoff") != "wyckoff":
            return None
        if isinstance(
            obj,
            (
                ChromastructureBackend,
                ChromastructureViewBase,
                httk.atomistic.models.protochroma.backend.ProtochromaBackend,
                httk.atomistic.models.protochroma.view_base.ProtochromaViewBase,
                httk.atomistic.models.protostructure.backend.ProtostructureBackend,
                httk.atomistic.models.protostructure.view_base.ProtostructureViewBase,
            ),
        ):
            return cls(obj, **hints)
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        if isinstance(
            obj,
            (
                ChromastructureViewBase,
                httk.atomistic.models.protochroma.view_base.ProtochromaViewBase,
                httk.atomistic.models.protostructure.view_base.ProtostructureViewBase,
            ),
        ):
            self._prototype = obj._backend
        else:
            self._prototype = obj

    @property
    def _is_protostructure(self) -> bool:
        return isinstance(self._prototype, httk.atomistic.models.protostructure.backend.ProtostructureBackend)

    @property
    def _is_protochroma(self) -> bool:
        return isinstance(self._prototype, httk.atomistic.models.protochroma.backend.ProtochromaBackend)

    @cached_property
    def _projected(self) -> Composition:
        source = self._prototype
        species_by_name: dict[str, Any] = {}
        for occupation in source.occupations:
            species_by_name.setdefault(occupation.species.name, occupation.species)
        proxy = SimpleNamespace(
            species=tuple(species_by_name.values()),
            wyckoff_sites=tuple(SimpleNamespace(species=occupation.species.name) for occupation in source.occupations),
            multiplicities=source.multiplicities,
            assemblies=None,
            chemical_composition=None,
        )
        return project_composition(proxy)

    @property
    def is_anonymous(self) -> bool:
        """Return whether this composition uses anonymous labels."""
        return not self._is_protostructure

    @cached_property
    def _amounts(self) -> tuple[tuple[str, Fraction], ...]:
        if self._is_protostructure:
            return self._projected.amounts
        if self._is_protochroma:
            counts: Counter[str] = Counter()
            for occupation, multiplicity in zip(self._prototype.occupations, self._prototype.multiplicities()):
                counts[occupation.label] += multiplicity
        else:
            counts = Counter(self._prototype.species_at_sites)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return tuple((anonymous_symbol(index), Fraction(count)) for index, (_, count) in enumerate(ordered))

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the canonical composition amounts."""
        return self._amounts

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        """Return the composition amount precisions."""
        if self._is_protostructure:
            return self._projected.uncertainties
        return super().uncertainties

    @property
    def complete(self) -> bool:
        """Return whether all represented elemental material is known."""
        return self._projected.complete if self._is_protostructure else True

    @property
    def exact(self) -> bool:
        """Return whether all composition amounts are exact."""
        return self._projected.exact if self._is_protostructure else super().exact

    @property
    def normalized(self) -> bool:
        """Return whether the composition is normalized within precision."""
        return self._projected.normalized if self._is_protostructure else super().normalized

    @property
    def normalization_status(self) -> str:
        """Return the composition's normalization status."""
        return self._projected.normalization_status if self._is_protostructure else super().normalization_status

    @property
    def diagnostics(self):
        """Return non-fatal diagnostics associated with the composition."""
        return self._projected.diagnostics if self._is_protostructure else super().diagnostics

    def unwrap(self) -> Any:
        """Return the raw object behind the Wyckoff-composition backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._prototype)
