"""The geometry-free protostructure interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.chromaformula_view import ChromaformulaView
from httk.atomistic.models.formula.formula_view import ChemicalFormulaView
from httk.atomistic.models.protochroma.notation import render_aflow_label

if TYPE_CHECKING:
    from httk.atomistic.models.protostructure.backend import ProtostructureBackend
    from httk.atomistic.models.protostructure.label import ProtostructureLabel
    from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
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
    def anonymous_formula(self) -> ChromaformulaView:
        """Return a reduced anonymous formula at the standard conventional-cell scale.

        :return: The conventional-cell anonymous formula view.
        """
        return ChromaformulaView(cast("ProtostructureBackend", self))

    @property
    def label(self) -> "ProtostructureLabel":
        """Return the httk protostructure label of this protostructure.

        The label's unsuffixed part orders classes by their Wyckoff letters, so it is the
        protochroma label of the erased form; the suffix lists the class species names.
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
