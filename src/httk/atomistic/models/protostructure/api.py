"""The assigned-species geometrical-classification interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formula_view import ChemicalFormulaView
from httk.atomistic.models.formula.formulatype_view import FormulatypeView
from httk.atomistic.models.prototype.notation import render_aflow_label

if TYPE_CHECKING:
    from httk.atomistic.models.protostructure.backend import ProtostructureBackend
    from httk.atomistic.models.protostructure.label import ProtostructureLabel
    from httk.atomistic.models.protostructure.like import ProtostructureLike
    from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class ProtostructureAPI(ABC):
    """The common interface for standard-setting occupied Wyckoff positions.

    Composition and formula derivations from this interface use the standard-setting
    conventional-cell scale, independently of the setting or transform of a source
    structure from which a protostructure was recognized.
    """

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group."""
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["WyckoffOccupation", ...]:
        """Return the occupied Wyckoff positions in canonical order."""
        raise NotImplementedError

    @property
    def representative(self) -> "FundamentalDomainStructure | None":
        """Return an optional retained exact representative."""
        return None

    @property
    def discriminator(self) -> str | None:
        """Return an optional geometrical-class discriminator."""
        return None

    def similar(self, other: "ProtostructureLike", delta: float) -> bool:
        """Return whether two protostructures have compatible geometry within ``delta``.

        The base identity (space group, occupations, and any discriminators present on
        both) must agree; when both sides retain a geometrical representative, their total
        Cartesian atom travel must not exceed ``delta``. A protostructure-like ``other``
        (a view, a backend, or a label string) is erased to a value first.

        :param other: The protostructure-like value to compare against.
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
        from httk.atomistic.models.protostructure.backend import ProtostructureBackend
        from httk.atomistic.models.protostructure.protostructure import Protostructure
        from httk.atomistic.models.protostructure.view import ProtostructureView
        from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase

        resolved: ProtostructureBackend
        if isinstance(other, ProtostructureViewBase):
            resolved = other.unview()
        elif isinstance(other, ProtostructureBackend):
            resolved = other
        else:
            try:
                resolved = ProtostructureView(other).unview()
            except (TypeError, ValueError):
                return False
        left: ProtostructureBackend = self if isinstance(self, Protostructure) else ProtostructureView(self).unview()
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
        from httk.atomistic.symmetry.paths import NoCommonRepresentation, structure_delta

        try:
            return structure_delta(left.representative, resolved.representative) <= delta
        except NoCommonRepresentation:
            return False

    def multiplicities(self) -> tuple[int, ...]:
        """Return tabulated standard-setting multiplicities for each occupation.

        These are deliberately not multiplicities from a source structure's transform;
        they define the provenance-independent conventional-cell scale of this value.

        :return: The standard-setting multiplicity for each occupation.
        """
        return tuple(
            self.spacegroup.wyckoff_position(occupation.wyckoff).multiplicity for occupation in self.occupations
        )

    @property
    def nsites_conventional(self) -> int:
        """Return the total number of sites in the standard conventional cell.

        :return: The conventional-cell site count.
        """
        return sum(self.multiplicities())

    @property
    def formula(self) -> ChemicalFormulaView:
        """Return a reduced formula at the standard conventional-cell scale.

        :return: The conventional-cell reduced formula view.
        """
        return ChemicalFormulaView(cast("ProtostructureBackend", self))

    @property
    def anonymous_formula(self) -> FormulatypeView:
        """Return a reduced anonymous formula at the standard conventional-cell scale.

        :return: The conventional-cell anonymous formula view.
        """
        return FormulatypeView(cast("ProtostructureBackend", self))

    @property
    def label(self) -> "ProtostructureLabel":
        """Return the httk protostructure label of this protostructure.

        The label's unsuffixed part orders classes by their Wyckoff letters, so it is the
        prototype label of the erased anonymous prototype; the suffix lists the class species names.
        This is NOT an AFLOW label: AFLOW orders classes alphabetically by element (see
        :attr:`aflow_label`). Any faithful render is *the* protostructure label; the
        *canonical* protostructure label comes from a normalizer-canonical protostructure.

        :return: The protostructure label view.
        """
        from httk.atomistic.models.protostructure.label import ProtostructureLabel

        return ProtostructureLabel(cast("ProtostructureBackend", self))

    @property
    def aflow_label(self) -> str:
        """Return the AFLOW-style label of this protostructure.

        Unlike :attr:`label`, AFLOW orders classes alphabetically by element symbol and
        reassigns the anonymous symbols in that order, so the unsuffixed prefix depends on
        the chemistry. Provided for interoperability only.

        :return: The AFLOW-style label text.
        """
        return render_aflow_label(
            self.spacegroup, [(occupation.wyckoff, occupation.species.name) for occupation in self.occupations]
        )

    @property
    def protostructure(self) -> Self:
        """Return this protostructure value."""
        return self
