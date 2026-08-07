"""Eager protostructure recognition and presentation view."""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.recognized import RecognizedProtostructure
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.symmetry.recognition import recognize_asu


def _occupations(asu: FundamentalDomainStructure) -> tuple[WyckoffOccupation, ...]:
    species_by_name = {species.name: species for species in asu.species}
    return tuple(WyckoffOccupation(site.wyckoff, species_by_name[site.species]) for site in asu.wyckoff_sites)


class ProtostructureView(ProtostructureViewBase, Protostructure):
    r"""Recognize an eager standard-setting protostructure view.

    Recognition accepts optional ``setting``, ``standard``, ``transform``, ``tolerance``,
    and ``limit_denominator`` values through the recognition hints.

    :param obj: The structure-like or protostructure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: ProtostructureBackend

    def __new__(
        cls,
        obj: Any,
        *,
        setting: Any = None,
        standard: Any = None,
        transform: Any = None,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
        **hints: Any,
    ) -> Self:
        if isinstance(obj, cls):
            if (
                any(value is not None for value in (setting, standard, transform, tolerance, limit_denominator))
                or hints
            ):
                raise ValueError("ProtostructureView rewrapping does not accept recognition arguments")
            return obj

        # Prototype-family inputs have dummy species; report the domain mismatch before backend probing.
        from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
        from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase

        if isinstance(obj, (AnonymousStructureBackend, AnonymousStructureViewBase)):
            raise TypeError(
                "a prototype/anonymous structure carries dummy species; a protostructure needs the real ones"
            )

        backend = cls._prepare_backend(obj, hints)
        recognition_values = (setting, standard, transform, tolerance, limit_denominator)
        if isinstance(backend, RecognizedProtostructure):
            structure = backend._structure
            asu = getattr(structure, "asu", None)
            if isinstance(asu, FundamentalDomainStructure):
                if any(value is not None for value in recognition_values):
                    raise ValueError("ProtostructureView recognition arguments cannot be used with an existing ASU")
            else:
                asu = recognize_asu(
                    structure,
                    setting=setting,
                    standard=standard,
                    transform=transform,
                    tolerance=tolerance,
                    limit_denominator=limit_denominator,
                )
        else:
            presented = getattr(backend, "protostructure", None)
            if presented is None:
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a protostructure source")
            if any(value is not None for value in recognition_values) or hints:
                raise ValueError("ProtostructureView recognition arguments cannot be used with a protostructure")
            instance = super().__new__(cls)
            Protostructure.__init__(instance, presented.spacegroup, presented.occupations)
            instance._backend = backend
            return instance

        instance = super().__new__(cls)
        Protostructure.__init__(instance, asu.spacegroup, _occupations(asu))
        instance._backend = backend
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Protostructure:
        """Return the recognized protostructure as a standalone value.

        :return: The protostructure value.
        """
        if type(self._backend) is Protostructure:
            return self._backend
        return Protostructure(self.spacegroup, self.occupations)
