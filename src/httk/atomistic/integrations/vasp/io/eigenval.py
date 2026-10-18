"""Read band energies and occupations from a VASP EIGENVAL file."""

import os
from dataclasses import dataclass
from typing import TextIO

from ._text import source_lines

__all__ = ["EigenvalFile", "read_eigenval"]


def _floats(bands: tuple[tuple[tuple[str, ...], ...], ...]) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(tuple(tuple(float(t) for t in k) for k in spin) for spin in bands)


@dataclass(frozen=True, slots=True)
class EigenvalFile:
    """Store the contents of a VASP EIGENVAL file with numeric lexemes unchanged.

    Band arrays are shaped ``(spins, kpoints, bands)``. Occupations are per state on 0..1
    for every ``ispin``.

    :param ispin: Number of spin channels declared by the file, 1 or 2.
    :param nions: Ion count from the first header line.
    :param nelect: Electron-count lexeme, printed either as an integer or as a float.
    :param nkpoints: Number of k-points.
    :param nbands: Number of bands per k-point.
    :param system: System name from the header.
    :param kpoints: Fractional k-point coordinate lexemes in the reciprocal basis.
    :param weights: K-point weight lexemes.
    :param energies: Band-energy lexemes shaped ``(spins, kpoints, bands)``.
    :param occupations: Occupation lexemes shaped like ``energies``, or ``None`` for files without them.
    """

    ispin: int
    nions: int
    nelect: str
    nkpoints: int
    nbands: int
    system: str
    kpoints: tuple[tuple[str, str, str], ...]
    weights: tuple[str, ...]
    energies: tuple[tuple[tuple[str, ...], ...], ...]
    occupations: tuple[tuple[tuple[str, ...], ...], ...] | None

    def __post_init__(self) -> None:
        """Validate ``ispin`` and the shape of every array.

        :raises ValueError: If ``ispin`` or an array shape is inconsistent.
        """
        if self.ispin not in (1, 2):
            raise ValueError(f"ispin must be 1 or 2, got {self.ispin!r}")
        if len(self.kpoints) != self.nkpoints or len(self.weights) != self.nkpoints:
            raise ValueError(f"kpoints and weights must have {self.nkpoints} entries")
        for name, bands in (("energies", self.energies), ("occupations", self.occupations)):
            if bands is None:
                continue
            if len(bands) != self.ispin or any(
                len(spin) != self.nkpoints or any(len(k) != self.nbands for k in spin) for spin in bands
            ):
                raise ValueError(f"{name} must be shaped ({self.ispin}, {self.nkpoints}, {self.nbands})")

    def kpoints_floats(self) -> tuple[tuple[float, float, float], ...]:
        """Convert the k-point lexemes to floating-point values.

        :return: Fractional k-point coordinates.
        """
        return tuple((float(k[0]), float(k[1]), float(k[2])) for k in self.kpoints)

    def weights_floats(self) -> tuple[float, ...]:
        """Convert the k-point weight lexemes to floating-point values.

        :return: K-point weights.
        """
        return tuple(float(w) for w in self.weights)

    def energies_floats(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Convert the band energies to floating-point values.

        :return: Energies in eV shaped ``(spins, kpoints, bands)``.
        """
        return _floats(self.energies)

    def occupations_floats(self) -> tuple[tuple[tuple[float, ...], ...], ...] | None:
        """Convert the occupations to floating-point values when present.

        :return: Occupations shaped ``(spins, kpoints, bands)``, or ``None`` when the file has none.
        """
        return None if self.occupations is None else _floats(self.occupations)

    def nelect_float(self) -> float:
        """Convert the electron-count lexeme to a floating-point value.

        :return: The number of electrons.
        """
        return float(self.nelect)


def _number(token: str, line: int) -> str:
    try:
        float(token)
    except ValueError:
        raise ValueError(f"line {line}: invalid number {token!r}") from None
    return token


def _integer(token: str, line: int, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError(f"line {line}: {what} must be an integer, got {token!r}") from None


def read_eigenval(source: str | os.PathLike[str] | TextIO) -> EigenvalFile:
    """Read a VASP 5/6 EIGENVAL file, with or without occupation columns.

    The occupation columns are told apart from the spin channels by ``ISPIN`` on the first
    line, never by column count alone. Paths are opened and closed by the reader; caller-owned
    text streams remain open.

    :param source: EIGENVAL path or caller-owned text stream.
    :return: The file contents with numeric lexemes unchanged.
    :raises ValueError: If the header, a k-point block, a band line, or the trailing content is malformed.
    """
    with source_lines(source) as (lines, _raw):
        iterator = iter(lines)
        lineno = 0

        def take() -> str | None:
            nonlocal lineno
            lineno += 1
            return next(iterator, None)

        header: list[str] = []
        for _ in range(6):
            line = take()
            if line is None:
                raise ValueError(f"line {lineno}: EIGENVAL header ended early (6 header lines are required)")
            header.append(line)
        first = header[0].split()
        if len(first) != 4:
            raise ValueError("line 1: expected four integers (ions, ions, flags, ISPIN)")
        nions = _integer(first[0], 1, "ion count")
        ispin = _integer(first[3], 1, "ISPIN")
        if ispin not in (1, 2):
            raise ValueError(f"line 1: ISPIN must be 1 or 2, got {ispin}")
        if len(header[1].split()) != 5:
            raise ValueError("line 2: expected five numbers (volume, lattice lengths, POTIM)")
        if len(header[2].split()) != 1:
            raise ValueError("line 3: expected the temperature")
        counts = header[5].split()
        if len(counts) != 3:
            raise ValueError("line 6: expected NELECT, NKPTS and NBANDS")
        nelect = _number(counts[0], 6)
        nkpoints = _integer(counts[1], 6, "NKPTS")
        nbands = _integer(counts[2], 6, "NBANDS")
        if nkpoints < 1 or nbands < 1:
            raise ValueError("line 6: NKPTS and NBANDS must be positive")

        # Column counts with / without occupations: one energy and one occupation per spin channel.
        with_occ, without_occ = 1 + 2 * ispin, 1 + ispin
        has_occ: bool | None = None
        kpoints: list[tuple[str, str, str]] = []
        weights: list[str] = []
        energies: list[list[tuple[str, ...]]] = [[] for _ in range(ispin)]
        occupations: list[list[tuple[str, ...]]] = [[] for _ in range(ispin)]
        for _ in range(nkpoints):
            blank = take()
            if blank is None:
                raise ValueError(f"line {lineno}: file ended before all {nkpoints} k-point blocks were read")
            if blank.strip():
                raise ValueError(f"line {lineno}: expected a blank line before a k-point block")
            line = take()
            fields = line.split() if line is not None else []
            if len(fields) != 4:
                raise ValueError(f"line {lineno}: expected a k-point line 'kx ky kz weight'")
            kpoints.append((_number(fields[0], lineno), _number(fields[1], lineno), _number(fields[2], lineno)))
            weights.append(_number(fields[3], lineno))
            e: list[list[str]] = [[] for _ in range(ispin)]
            o: list[list[str]] = [[] for _ in range(ispin)]
            for band in range(1, nbands + 1):
                line = take()
                if line is None:
                    raise ValueError(f"line {lineno}: file ended inside a k-point block (band {band} missing)")
                tokens = line.split()
                if has_occ is None:
                    if len(tokens) not in (with_occ, without_occ):
                        raise ValueError(
                            f"line {lineno}: expected {without_occ} or {with_occ} columns for ISPIN={ispin}, "
                            f"got {len(tokens)}"
                        )
                    has_occ = len(tokens) == with_occ
                elif len(tokens) != (with_occ if has_occ else without_occ):
                    raise ValueError(
                        f"line {lineno}: expected {with_occ if has_occ else without_occ} columns "
                        f"for ISPIN={ispin}, got {len(tokens)}"
                    )
                if tokens[0] != str(band):
                    raise ValueError(f"line {lineno}: band index {tokens[0]!r} out of sequence, expected {band}")
                for s in range(ispin):
                    e[s].append(_number(tokens[1 + s], lineno))
                    if has_occ:
                        o[s].append(_number(tokens[1 + ispin + s], lineno))
            for s in range(ispin):
                energies[s].append(tuple(e[s]))
                occupations[s].append(tuple(o[s]))
        while (line := take()) is not None:
            if line.strip():
                raise ValueError(f"line {lineno}: unexpected content after the last k-point block")

    return EigenvalFile(
        ispin,
        nions,
        nelect,
        nkpoints,
        nbands,
        header[4].strip(),
        tuple(kpoints),
        tuple(weights),
        tuple(tuple(spin) for spin in energies),
        tuple(tuple(spin) for spin in occupations) if has_occ else None,
    )
