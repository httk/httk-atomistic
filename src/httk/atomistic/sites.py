"""
The Sites class for httk-atomistic.
"""

from collections.abc import Iterator, Sequence


class Sites:
    """
    The sites of a crystal structure: the Nx3 matrix of reduced coordinates.

    A Sites object holds N sites as the rows of ``reduced_coords`` and is iterable and
    indexable over those length-3 coordinate rows (with ``len`` giving the number of
    sites).

    Note: the numeric values are stored as interim nested tuples of floats. They are
    intended to be replaced by the httk exact vector representation fairly soon; keep
    numeric access behind the ``reduced_coords`` accessor so that change stays contained.
    """

    _reduced_coords: tuple[tuple[float, ...], ...]

    def __init__(self, reduced_coords: Sequence[Sequence[float]]) -> None:
        norm = tuple(tuple(float(x) for x in row) for row in reduced_coords)
        if any(len(row) != 3 for row in norm):
            raise ValueError("Sites reduced_coords must be a sequence of length-3 coordinates")
        self._reduced_coords = norm

    @property
    def reduced_coords(self) -> tuple[tuple[float, ...], ...]:
        """The Nx3 reduced site coordinates as nested float tuples (one site per row)."""
        return self._reduced_coords

    def __len__(self) -> int:
        return len(self._reduced_coords)

    def __iter__(self) -> Iterator[tuple[float, ...]]:
        return iter(self._reduced_coords)

    def __getitem__(self, index: int) -> tuple[float, ...]:
        return self._reduced_coords[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sites):
            return NotImplemented
        return self._reduced_coords == other._reduced_coords

    def __repr__(self) -> str:
        return f"Sites(reduced_coords={self._reduced_coords!r})"
