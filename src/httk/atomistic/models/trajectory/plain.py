"""Map OPTIMADE trajectory properties to a backend."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

from httk.core import SurdVector

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.backend import TrajectoryBackend

_SCALARS = frozenset({"id", "type", "immutable_id", "last_modified", "nframes", "reference_frames"})
_STRUCTURE_PROPERTIES = frozenset(
    {
        "elements",
        "nelements",
        "elements_ratios",
        "chemical_formula_descriptive",
        "chemical_formula_reduced",
        "chemical_formula_hill",
        "chemical_formula_anonymous",
        "dimension_types",
        "nperiodic_dimensions",
        "lattice_vectors",
        "space_group_symmetry_operations_xyz",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
        "space_group_it_number",
        "cartesian_site_positions",
        "fractional_site_positions",
        "site_coordinate_span",
        "site_coordinate_span_description",
        "nsites",
        "species_at_sites",
        "species",
        "assemblies",
        "wyckoff_positions",
        "structure_features",
        "optimization_type",
    }
)
_COMPACTABLE = frozenset(
    {
        "elements",
        "nelements",
        "elements_ratios",
        "chemical_formula_descriptive",
        "chemical_formula_reduced",
        "chemical_formula_hill",
        "dimension_types",
        "nperiodic_dimensions",
        "lattice_vectors",
        "space_group_symmetry_operations_xyz",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
        "space_group_it_number",
        "fractional_site_positions",
        "site_coordinate_span",
        "site_coordinate_span_description",
        "nsites",
        "species_at_sites",
        "species",
        "assemblies",
        "wyckoff_positions",
        "structure_features",
        "optimization_type",
    }
)


class PlainTrajectory(TrajectoryBackend):
    r"""Represent a mapping whose structure properties have a frame axis.

    A compact constant property is represented by a one-element leading axis,
    e.g. ``nelements=[2]`` for any number of frames. Only properties declaring
    ``constant`` on that axis accept this compact form.

    :param obj: A trajectory property mapping.
    :param \**hints: Backend-selection hints.
    """

    kind: ClassVar[str] = "plain"
    _raw: Mapping[str, Any]
    _nframes: int
    _reference_frames: tuple[int, ...] | None
    _observable_names: tuple[str, ...]

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        return super().__new__(cls) if isinstance(obj, Mapping) else None

    def __init__(self, obj: Mapping[str, Any], **hints: Any) -> None:
        nframes = obj.get("nframes")
        if not isinstance(nframes, int) or isinstance(nframes, bool) or nframes < 1:
            raise ValueError("PlainTrajectory nframes must be a positive integer")
        for name, value in obj.items():
            if name in _SCALARS or value is None:
                continue
            self._validate_axis(name, value, nframes)
        references = obj.get("reference_frames")
        normalized = None
        if references is not None:
            if not isinstance(references, Sequence) or isinstance(references, str | bytes):
                raise ValueError("PlainTrajectory reference_frames must be a sequence")
            checked = []
            for value in references:
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < nframes:
                    raise ValueError(f"PlainTrajectory reference frame {value!r} is out of bounds")
                checked.append(value)
            normalized = tuple(sorted(set(checked)))
        self._raw = obj
        self._nframes = nframes
        self._reference_frames = normalized
        self._observable_names = tuple(
            name for name in obj if name not in _SCALARS and name not in _STRUCTURE_PROPERTIES
        )
        self._validate_composition()

    @staticmethod
    def _validate_axis(name: str, value: Any, nframes: int) -> None:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise ValueError(f"PlainTrajectory property {name!r} must have a frame axis")
        length = len(value)
        if length != nframes and not (length == 1 and name in _COMPACTABLE):
            raise ValueError(f"PlainTrajectory property {name!r} has leading axis length {length}, expected {nframes}")

    def _validate_composition(self) -> None:
        for name in ("species", "species_at_sites"):
            value = self._raw.get(name)
            if value is None:
                continue
            values = self._values(name)
            if any(item != values[0] for item in values[1:]):
                raise ValueError(f"PlainTrajectory property {name!r} varies between frames")

    def _values(self, name: str) -> tuple[Any, ...]:
        value = self._raw[name]
        if value is None:
            return (None,) * self._nframes
        if len(value) == 1 and name in _COMPACTABLE:
            return (value[0],) * self._nframes
        return tuple(value)

    def _value(self, name: str, i: int) -> Any:
        try:
            self._raw[name]
        except KeyError:
            raise KeyError(f"PlainTrajectory has no property {name!r}") from None
        return self._values(name)[i]

    def _index(self, i: int) -> int:
        if not isinstance(i, int):
            raise TypeError("Trajectory frame index must be an integer")
        if i < 0:
            i += self._nframes
        if not 0 <= i < self._nframes:
            raise IndexError(f"Trajectory frame index {i} out of range")
        return i

    def frame(self, i: int) -> UnitcellStructure:
        """Return one frame from the property mapping.

        :param i: Frame index; negative indexes count from the end.
        :return: The requested unit-cell structure.
        :raises IndexError: If the frame index is out of range.
        :raises KeyError: If a required trajectory property is absent.
        :raises TypeError: If the frame index is not an integer.
        :raises ValueError: If the frame cannot be represented as a structure.
        """
        i = self._index(i)
        fractional = self._value("fractional_site_positions", i) if "fractional_site_positions" in self._raw else None
        cartesian = self._value("cartesian_site_positions", i) if "cartesian_site_positions" in self._raw else None
        lattice = self._value("lattice_vectors", i)
        dimensions = self._value("dimension_types", i) if "dimension_types" in self._raw else None
        periodicity = (True, True, True) if dimensions is None else tuple(bool(value) for value in dimensions)
        cell = Cell(lattice, periodicity=periodicity)
        if fractional is None:
            if cartesian is None:
                raise ValueError("PlainTrajectory frame requires fractional_site_positions or cartesian_site_positions")
            sites = Sites(SurdVector.create(cartesian) * cell.basis.inv())
        else:
            sites = Sites(fractional)
        names = self._value("species_at_sites", i)
        species_value = self._value("species", i) if "species" in self._raw else None
        species = None if species_value is None else tuple(Species.create(value) for value in species_value)
        return UnitcellStructure(
            cell,
            sites,
            species,
            names,
            molecular=self._value("site_coordinate_span", i) == "molecular_unit_cell"
            if "site_coordinate_span" in self._raw
            else False,
            chemical_formula_descriptive=self._value("chemical_formula_descriptive", i)
            if "chemical_formula_descriptive" in self._raw
            else None,
            chemical_formula_hill=self._value("chemical_formula_hill", i)
            if "chemical_formula_hill" in self._raw
            else None,
            optimization_type=self._value("optimization_type", i) if "optimization_type" in self._raw else None,
        )

    def frames(self) -> Iterator[UnitcellStructure]:
        """Iterate over all frames in source order.

        :return: An iterator of unit-cell structures.
        """
        return (self.frame(i) for i in range(self._nframes))

    @property
    def nframes(self) -> int:
        """Return the number of frames."""
        return self._nframes

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        """Return normalized reference-frame indexes, or ``None``."""
        return self._reference_frames

    @property
    def species(self) -> tuple[Species, ...]:
        """Return the constant distinct species from the first frame."""
        return self.frame(0).species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the constant species name at each site."""
        return self.frame(0).species_at_sites

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return names outside the recognized trajectory and structure properties."""
        return self._observable_names

    def observable(self, name: str) -> tuple[Any, ...]:
        """Return one mapped observable's values in frame order.

        :param name: Observable property name.
        :return: The observable values.
        :raises KeyError: If the property is not an observable.
        """
        if name not in self._observable_names:
            raise KeyError(name)
        return self._values(name)

    def unwrap(self) -> Mapping[str, Any]:
        """Return the original property mapping."""
        return self._raw
