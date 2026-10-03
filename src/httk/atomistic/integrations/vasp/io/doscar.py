"""Read the total DOS block from a VASP DOSCAR file."""

import os
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import TextIO

from ._text import source_lines

__all__ = ["DOSCAR", "DOSChannel", "read_doscar"]

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z")


def _fraction(token: str, line: int) -> Fraction:
    if _NUMBER.fullmatch(token) is None:
        raise ValueError(f"line {line}: invalid finite DOSCAR number {token!r}")
    return Fraction(token.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True, slots=True)
class DOSChannel:
    """Store exact density and integrated density for one spin channel.

    :param name: Channel name, such as ``total``, ``up`` or ``down``.
    :param density: Density samples in states/eV.
    :param integrated_density: Integrated density samples in states.
    """

    name: str
    density: tuple[Fraction, ...]
    integrated_density: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        """Copy samples into immutable exact tuples and validate matching lengths."""
        density = tuple(Fraction(value) for value in self.density)
        integrated = tuple(Fraction(value) for value in self.integrated_density)
        if len(density) != len(integrated):
            raise ValueError("density and integrated_density must have the same length")
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "integrated_density", integrated)


@dataclass(frozen=True, slots=True)
class DOSCAR:
    """Store the exact total-DOS grid and channels from a DOSCAR file.

    :param energies: Strictly increasing energy grid in eV.
    :param fermi_energy: Fermi energy in eV.
    :param spin_mode: Explicit VASP spin mode, ``nonmagnetic`` or ``collinear``.
    :param channels: Total density and integrated density grouped by spin channel.
    """

    energies: tuple[Fraction, ...]
    fermi_energy: Fraction
    spin_mode: str
    channels: tuple[DOSChannel, ...]

    def __post_init__(self) -> None:
        """Copy values into immutable tuples and validate the DOS grid and channels."""
        energies = tuple(Fraction(value) for value in self.energies)
        channels = tuple(self.channels)
        expected = ("total",) if self.spin_mode == "nonmagnetic" else ("up", "down")
        if self.spin_mode not in {"nonmagnetic", "collinear"}:
            raise ValueError("spin_mode must be 'nonmagnetic' or 'collinear'")
        if not energies or any(right <= left for left, right in pairwise(energies)):
            raise ValueError("energies must be a nonempty, strictly increasing grid")
        if tuple(channel.name for channel in channels) != expected:
            raise ValueError(f"{self.spin_mode} DOSCAR channels must be named {expected}")
        if any(len(channel.density) != len(energies) for channel in channels):
            raise ValueError("every DOS channel must match the energy grid length")
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "fermi_energy", Fraction(self.fermi_energy))
        object.__setattr__(self, "channels", channels)


def read_doscar(source: str | os.PathLike[str] | TextIO, *, spin_mode: str) -> DOSCAR:
    """Read only the NEDOS-row total-DOS block and retain decimal values exactly.

    Six standard header lines precede the total DOS block. Any atom-projected
    blocks after it are left unread and unparsed. Paths are opened and closed
    by the reader; caller-owned text streams remain open.

    :param source: DOSCAR path or caller-owned text stream.
    :param spin_mode: Required interpretation, ``nonmagnetic`` or ``collinear``.
    :return: Immutable total-DOS data with exact rational values.
    :raises ValueError: If the mode, header, row count, columns, or values are invalid.
    """
    if spin_mode not in {"nonmagnetic", "collinear"}:
        raise ValueError("spin_mode must be 'nonmagnetic' or 'collinear'")
    with source_lines(source) as (lines, _raw):
        iterator = iter(lines)
        header = [next(iterator, None) for _ in range(6)]
        if any(line is None for line in header):
            raise ValueError("DOSCAR must contain all six header lines")
        fields = header[5].split()  # type: ignore[union-attr]
        if len(fields) < 5 or not fields[2].isdigit() or int(fields[2]) < 1:
            raise ValueError("DOSCAR line 6 must declare a positive NEDOS and Fermi energy")
        nedos = int(fields[2])
        fermi_energy = _fraction(fields[3], 6)
        expected_columns = 3 if spin_mode == "nonmagnetic" else 5
        energies: list[Fraction] = []
        values: list[list[Fraction]] = [[] for _ in range(expected_columns - 1)]
        for row_index in range(nedos):
            line_number = row_index + 7
            line = next(iterator, None)
            if line is None:
                raise ValueError(f"DOSCAR ended before all {nedos} total-DOS rows were read")
            tokens = line.split()
            if len(tokens) != expected_columns:
                raise ValueError(
                    f"line {line_number}: expected {expected_columns} columns for {spin_mode} DOS, got {len(tokens)}"
                )
            parsed = [_fraction(token, line_number) for token in tokens]
            energy = parsed[0]
            if energies and energy <= energies[-1]:
                raise ValueError(f"line {line_number}: DOS energies must be strictly increasing")
            energies.append(energy)
            for column, value in enumerate(parsed[1:]):
                values[column].append(value)

    if spin_mode == "nonmagnetic":
        channels: tuple[DOSChannel, ...] = (DOSChannel("total", tuple(values[0]), tuple(values[1])),)
    else:
        channels = (
            DOSChannel("up", tuple(values[0]), tuple(values[2])),
            DOSChannel("down", tuple(values[1]), tuple(values[3])),
        )
    return DOSCAR(tuple(energies), fermi_energy, spin_mode, channels)
