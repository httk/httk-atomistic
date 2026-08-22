"""Exact, private canonical form for CIF structure-reading regression fixtures."""

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

from httk.core import load

from httk.atomistic.cif_structures import asu_structures_from_cif
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

__all__ = ["structreading_golden"]


class _RepairWarnings(logging.Handler):
    """Collect atomistic CIF repair warnings emitted while one file is read."""

    messages: list[str]

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the rendered warning text for categorization.

        :param record: The emitted log record.
        """
        self.messages.append(record.getMessage())


def structreading_golden(path: str | Path) -> dict[str, Any]:
    """Return the exact regression form for one CIF, applying repairs only when needed.

    :param path: CIF file to read.
    :return: JSON-serializable exact interpretation of the file.
    :raises ValueError: If strict and repair reading both fail, or a repair warning is unknown.
    """
    source = str(path)
    try:
        raw = load(source, raw=True)
        structure = asu_structures_from_cif(raw)[0]
    except ValueError as error:
        strict_status = _stable_error_prefix(error)
        handler = _RepairWarnings()
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            raw = load(source, raw=True, repair=True)
            structure = asu_structures_from_cif(raw)[0]
        finally:
            logger.removeHandler(handler)
        repairs = sorted(_repair_category(message) for message in handler.messages)
    else:
        strict_status = "ok"
        repairs = []

    block = raw["blocks"][0]
    species = sorted((_species_record(value) for value in structure.species), key=lambda value: value["name"])
    sites = sorted(
        (
            site.species,
            site.wyckoff,
            tuple(str(value) for value in site.free_params.to_fractions()),
        )
        for site in structure.wyckoff_sites
    )
    expanded = UnitcellStructureView(structure)
    composition = expanded.composition
    setting = structure.setting()
    if setting is None:
        raise ValueError("structure has no identified setting; cannot build a golden record")
    return {
        "strict_status": strict_status,
        "repair_warnings": repairs,
        "hall_entry": setting.hall_entry,
        "it_number": structure.spacegroup.it_number,
        "cell_parameters": [str(Fraction(value)) for value in block["cell_parameters_exact"]],
        "species": species,
        "sites": [[name, letter, list(parameters)] for name, letter, parameters in sites],
        "composition": {
            "amounts": [[symbol, str(amount)] for symbol, amount in composition.amounts],
            "uncertainties": [
                [symbol, None if width is None else str(width)] for symbol, width in composition.uncertainties
            ],
            "complete": composition.complete,
            "exact": composition.exact,
            "normalized": composition.normalized,
            "normalization_status": composition.normalization_status,
            "diagnostics": [
                {
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                    "subject": diagnostic.subject,
                    "total": None if diagnostic.total is None else str(diagnostic.total),
                    "width": None if diagnostic.width is None else str(diagnostic.width),
                }
                for diagnostic in composition.diagnostics
            ],
        },
        "formula": str(expanded.formula),
        "expanded_site_count": len(expanded.sites),
    }


def _species_record(species: Any) -> dict[str, Any]:
    """Return every lossless species field in a JSON-stable form.

    :param species: Species value read from the CIF.
    :return: Exact JSON-serializable species representation.
    """

    def exact_values(values: Any) -> list[str | None] | None:
        if values is None:
            return None
        return [None if value is None else str(value) for value in values]

    return {
        "name": species.name,
        "chemical_symbols": list(species.chemical_symbols),
        "concentration": [str(value) for value in species.concentration],
        "concentration_precision": exact_values(species.concentration_precision),
        "mass": None if species.mass is None else list(species.mass),
        "original_name": species.original_name,
        "attached": None if species.attached is None else list(species.attached),
        "nattached": None if species.nattached is None else list(species.nattached),
        "charges": exact_values(species.charges),
        "spins": exact_values(species.spins),
        "labels": None if species.labels is None else list(species.labels),
    }


def _stable_error_prefix(error: ValueError) -> str:
    """Remove the optional remediation tail from a stable reading error.

    :param error: The strict read error.
    :return: Its stable prefix.
    """
    return str(error).split(" Remedy:", 1)[0]


def _repair_category(message: str) -> str:
    """Translate one documented repair warning into its stable category.

    :param message: The rendered repair warning.
    :return: A category name suitable for a platform-stable golden.
    :raises ValueError: If the warning is not a known structure-reading repair.
    """
    categories = {
        "ignored declared symmetry": "unrecognized_declared_symmetry",
        "snapped its rounded coordinate": "rounded_coordinate_snap",
        "ignored declared Wyckoff data": "invalid_declared_wyckoff",
        "dropped malformed auxiliary loop": "malformed_auxiliary_loop",
    }
    for marker, category in categories.items():
        if marker in message:
            return category
    raise ValueError(f"unrecognized CIF repair warning: {message}")
