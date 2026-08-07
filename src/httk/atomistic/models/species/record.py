"""Backend for an exact stored species record."""

from fractions import Fraction
from typing import Any

from httk.atomistic.models.species.backend import SpeciesBackend
from httk.atomistic.storage.records import SpeciesRecord, _concentration_precision_from_record


class RecordSpecies(SpeciesBackend):
    r"""Backend for a species stored in an exact record.

    :param obj: The stored species record.
    :param \**hints: Backend-selection hints.
    """

    _record: SpeciesRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, SpeciesRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: SpeciesRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def name(self) -> str:
        """Return the species name.

        :return: The species name.
        """
        return self._record.name

    @property
    def chemical_symbols(self) -> tuple[str, ...]:
        """Return the constituent symbols.

        :return: The chemical symbols.
        """
        return self._record.chemical_symbols

    @property
    def concentration(self) -> tuple[Fraction, ...]:
        """Return the constituent concentrations.

        :return: The concentrations.
        """
        return self._record.concentration

    @property
    def concentration_precision(self) -> tuple[Fraction | None, ...] | None:
        """Return the concentration precision metadata.

        :return: Per-constituent precision, or ``None`` when unavailable.
        """
        return _concentration_precision_from_record(self._record.concentration_precision)

    @property
    def charges(self) -> tuple[Fraction | None, ...] | None:
        """Return the constituent charges.

        :return: The charges, or ``None`` when unstated.
        """
        return self._record.charges

    @property
    def spins(self) -> tuple[Fraction | None, ...] | None:
        """Return the constituent spins.

        :return: The spins, or ``None`` when unstated.
        """
        return self._record.spins

    @property
    def labels(self) -> tuple[str | None, ...] | None:
        """Return the constituent labels.

        :return: The labels, or ``None`` when unstated.
        """
        return self._record.labels

    @property
    def mass(self) -> tuple[float, ...] | None:
        """Return the constituent masses.

        :return: The masses, or ``None`` when unstated.
        """
        return self._record.mass

    @property
    def attached(self) -> tuple[str, ...] | None:
        """Return the attached constituent symbols.

        :return: The attached symbols, or ``None`` when unstated.
        """
        return self._record.attached

    @property
    def nattached(self) -> tuple[int, ...] | None:
        """Return the attached counts.

        :return: The attached counts, or ``None`` when unstated.
        """
        return self._record.nattached

    @property
    def original_name(self) -> str | None:
        """Return the original source name.

        :return: The original name, or ``None`` when unstated.
        """
        return self._record.original_name

    def unwrap(self) -> SpeciesRecord:
        """Return the stored species record.

        :return: The source record.
        """
        return self._record
