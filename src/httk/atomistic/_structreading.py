"""Exact, private canonical form for CIF structure-reading regression fixtures."""

import logging
from collections import Counter
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
    :raises ValueError: If strict and autocorrect reading both fail, or a repair warning is unknown.
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
            raw = load(source, raw=True, autocorrect=True)
            structure = asu_structures_from_cif(raw)[0]
        finally:
            logger.removeHandler(handler)
        repairs = sorted(_repair_category(message) for message in handler.messages)
    else:
        strict_status = "ok"
        repairs = []

    block = raw["blocks"][0]
    species_symbols = {species.name: species.chemical_symbols[0] for species in structure.species}
    sites = sorted(
        (
            species_symbols[site.species],
            site.wyckoff,
            tuple(str(value) for value in site.free_params.to_fractions()),
        )
        for site in structure.wyckoff_sites
    )
    expanded = UnitcellStructureView(structure)
    formula = dict(sorted(Counter(species_symbols[name] for name in expanded.species_at_sites).items()))
    setting = structure.setting()
    if setting is None:
        raise ValueError("structure has no identified setting; cannot build a golden record")
    return {
        "strict_status": strict_status,
        "repair_warnings": repairs,
        "hall_entry": setting.hall_entry,
        "it_number": structure.spacegroup.it_number,
        "cell_parameters": [str(Fraction(value)) for value in block["cell_parameters_exact"]],
        "sites": [[symbol, letter, list(parameters)] for symbol, letter, parameters in sites],
        "formula": formula,
        "expanded_site_count": len(expanded.sites),
    }


def _stable_error_prefix(error: ValueError) -> str:
    """Remove the optional remediation tail from a stable reading error.

    :param error: The strict read error.
    :return: Its stable prefix.
    """
    return str(error).split(" Remedy:", 1)[0]


def _repair_category(message: str) -> str:
    """Translate one documented autocorrect warning into its stable category.

    :param message: The rendered repair warning.
    :return: A category name suitable for a platform-stable golden.
    :raises ValueError: If the warning is not a known structure-reading repair.
    """
    categories = {
        "ignored declared symmetry": "unrecognized_declared_symmetry",
        "dropped co-located disorder site": "co_located_disorder_site",
        "snapped its rounded coordinate": "rounded_coordinate_snap",
        "ignored declared Wyckoff data": "invalid_declared_wyckoff",
        "dropped malformed auxiliary loop": "malformed_auxiliary_loop",
    }
    for marker, category in categories.items():
        if marker in message:
            return category
    raise ValueError(f"unrecognized CIF autocorrect warning: {message}")
