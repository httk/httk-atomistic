"""Refusing crystallographic symmetry for anything that is not a 3D crystal.

Space groups describe the symmetry of a three-dimensionally periodic lattice, and every part
of the symmetry machinery in this package is built on that: operations are 3x3 rational
matrices, coordinates are reduced modulo ``Z^3``, Wyckoff membership is a congruence in all
three directions, and the vendored tables hold the 230 space groups in 527 settings.

A slab or a nanowire has symmetry too — a layer group or a rod group — but that is different
tabulated data and different mathematics, and neither is present here. So the honest answer
for a reduced-periodicity structure is a refusal at the door, rather than an answer computed
by machinery whose assumptions it violates. Without this guard the failures are silent: an
orbit generated across a non-periodic direction quietly deletes atoms, and a distance folded
across one makes an unsymmetric structure pass as symmetric.
"""

from typing import Any

__all__ = ["require_full_periodicity"]


def require_full_periodicity(cell: Any, operation: str) -> None:
    """Raise :class:`ValueError` unless ``cell`` is periodic in all three directions.

    ``operation`` names the caller, so the message says which thing was refused.
    """
    periodicity = getattr(cell, "periodicity", (True, True, True))
    if periodicity == (True, True, True):
        return
    count = sum(periodicity)
    raise ValueError(
        f"{operation} requires a fully 3D-periodic structure, but this one is periodic in "
        f"{count} of 3 directions ({periodicity}). Crystallographic space groups describe 3D "
        f"lattices; the symmetry of a slab or a wire is a layer or rod group, which httk does "
        f"not tabulate. Reduced-periodicity structures can be built, compared, measured and "
        f"served, but not symmetry-analysed."
    )
