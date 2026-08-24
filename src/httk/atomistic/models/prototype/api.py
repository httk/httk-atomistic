"""The anonymous geometrical-class prototype interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formulatype_view import FormulatypeView
from httk.atomistic.models.prototype.notation import pearson_symbol, render_prototype_label

if TYPE_CHECKING:
    from httk.atomistic.models.prototype.backend import PrototypeBackend
    from httk.atomistic.models.prototype.label import PrototypeLabel
    from httk.atomistic.models.prototype.like import PrototypeLike
    from httk.atomistic.models.prototype.occupation import PrototypeOccupation
    from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class PrototypeAPI(ABC):
    """Common interface for anonymous standard-setting Wyckoff prototypes."""

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["PrototypeOccupation", ...]:
        raise NotImplementedError

    @property
    def representative(self) -> "FundamentalDomainTemplate | None":
        """Return an optional retained exact representative."""
        return None

    @property
    def discriminator(self) -> str | None:
        """Return an optional geometrical-class discriminator."""
        return None

    def multiplicities(self) -> tuple[int, ...]:
        return tuple(self.spacegroup.wyckoff_position(value.wyckoff).multiplicity for value in self.occupations)

    @property
    def nsites_conventional(self) -> int:
        return sum(self.multiplicities())

    @property
    def pearson_symbol(self) -> str:
        return pearson_symbol(self.spacegroup, self.nsites_conventional)

    @property
    def anonymous_formula(self) -> FormulatypeView:
        return FormulatypeView(cast("PrototypeBackend", self))

    @property
    def label(self) -> "PrototypeLabel":
        from httk.atomistic.models.prototype.label import PrototypeLabel

        return PrototypeLabel(cast("PrototypeBackend", self))

    @property
    def prototype(self) -> Self:
        return self

    def _prototype_label_text(self) -> str:
        return render_prototype_label(self.spacegroup, [(value.wyckoff, value.label) for value in self.occupations])

    def similar(self, other: "PrototypeLike", delta: float) -> bool:
        """Return whether two prototypes have compatible geometry within ``delta``.

        The base identity (space group, anonymous occupations, and any discriminators
        present on both) must agree; when both sides retain a geometrical representative,
        their total Cartesian atom travel must not exceed ``delta``. A prototype-like or
        protostructure-like ``other`` is erased through :class:`~httk.atomistic.PrototypeView` first, so a
        bare protostructure compares against its anonymous prototype.

        :param other: The prototype-like or protostructure-like value to compare against.
        :param delta: The non-negative finite Cartesian travel budget.
        :return: Whether the two values are compatible within ``delta``.
        :raises TypeError: If ``delta`` is not a real number.
        :raises ValueError: If ``delta`` is negative or non-finite.
        """
        import math
        from numbers import Real

        if not isinstance(delta, Real) or isinstance(delta, bool):
            raise TypeError("delta must be a finite non-negative real")
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("delta must be a finite non-negative real")
        from httk.atomistic.models.prototype.backend import PrototypeBackend
        from httk.atomistic.models.prototype.prototype import Prototype
        from httk.atomistic.models.prototype.view import PrototypeView
        from httk.atomistic.models.prototype.view_base import PrototypeViewBase

        resolved: PrototypeBackend
        if isinstance(other, PrototypeViewBase):
            resolved = other.unview()
        elif isinstance(other, PrototypeBackend):
            resolved = other
        else:
            try:
                resolved = PrototypeView(other).unview()
            except (TypeError, ValueError):
                return False
        left: PrototypeBackend = self if isinstance(self, Prototype) else PrototypeView(self).unview()
        if left.spacegroup != resolved.spacegroup or left.occupations != resolved.occupations:
            return False
        if (
            left.discriminator is not None
            and resolved.discriminator is not None
            and left.discriminator != resolved.discriminator
        ):
            return False
        if left.representative is None or resolved.representative is None:
            return True
        from httk.atomistic.models.prototype.derived import _prototype_to_structure
        from httk.atomistic.symmetry.paths import NoCommonRepresentation, structure_delta

        first = _prototype_to_structure(left.representative)
        second = _prototype_to_structure(resolved.representative)
        try:
            return structure_delta(first, second) <= delta
        except NoCommonRepresentation:
            return False
