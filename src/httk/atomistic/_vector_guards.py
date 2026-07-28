"""
Shared vector-family acceptance guards and normalizers for the atomistic backends.

The cell/sites/structure backends accept an input if it can be built through the httk-core
exact-vector family (:class:`~httk.core.FracVector` / :class:`~httk.core.SurdVector`) at the
required shape. Acceptance is therefore "does ``SurdVector.create`` succeed and land on the right
``dim``", replacing the ad-hoc ``_is_number``/``_is_3x3``/``_is_nx3``/``_is_params`` predicates. This
uniformly admits :class:`~fractions.Fraction`, rational strings (``"1/3"``), ``FracVector``,
``SurdVector``, and numpy arrays alongside the plain nested lists/tuples of numbers.
"""

import fractions
from collections.abc import Iterable
from typing import Any

from httk.core import (
    FracVector,
    SurdScalar,
    SurdVector,
    VectorFracView,
    numpy_available,
)


def require_numpy() -> None:
    """Raise :class:`ImportError` (naming the ``httk-atomistic[numpy]`` extra) if numpy is unavailable."""
    if not numpy_available():
        raise ImportError("the numeric layer requires numpy; install the httk-atomistic[numpy] extra")


def to_surdvector(obj: Any) -> SurdVector:
    """Normalize any vector-like input into an exact :class:`~httk.core.SurdVector`."""
    if isinstance(obj, SurdVector):
        return obj
    return SurdVector.create(obj)


def try_surdvector(obj: Any) -> SurdVector | None:
    """Return :func:`to_surdvector` of ``obj``, or None if it is not vector-like."""
    try:
        return to_surdvector(obj)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def to_fracvector(obj: Any) -> FracVector:
    """
    Normalize any vector-like input into an exact :class:`~httk.core.FracVector`.

    A :class:`~httk.core.SurdVector` is routed through the vector family's ``fractions`` hub
    (:class:`~httk.core.VectorFracView`): exact when the value is rational, else a deterministic
    rational reduction (never raises on data).
    """
    if isinstance(obj, FracVector):
        return obj
    if isinstance(obj, SurdVector):
        return FracVector.create(VectorFracView(obj))
    return FracVector.create(obj)


def to_surdscalar(obj: Any) -> SurdScalar:
    """Normalize an int/float/Fraction/str/SurdScalar into an exact :class:`~httk.core.SurdScalar`."""
    value = to_surdvector(obj)
    if value.dim != ():
        raise ValueError(f"expected a scalar value, got shape {value.dim}")
    return value._as_scalar()


def to_precision(obj: Any) -> fractions.Fraction | None:
    """Normalize a stated data precision into an exact positive rational, or ``None``.

    ``None`` means the precision is unknown, which is a real answer and not the same as
    claiming perfect precision. Anything the vector family accepts as a scalar works —
    ``1e-4``, ``"1/10000"``, a :class:`~fractions.Fraction` — and lands exactly, so a
    precision survives storage and comparison without picking up binary noise.

    A non-positive precision is rejected rather than silently treated as unknown: zero
    would claim exactness that no measured value has, and a negative one is meaningless.
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        # Through the decimal spelling, so 1e-4 lands on 1/10000 rather than on the binary
        # value a float literally holds. A precision is a written claim about digits, and
        # embedding it binary-exactly would record a number nobody stated.
        exact = fractions.Fraction(str(obj))
    else:
        value = to_fracvector(obj)
        if value.dim != ():
            raise ValueError(f"a precision must be a single value, got shape {value.dim}")
        exact = value.to_fraction()
    if exact <= 0:
        raise ValueError(f"a precision must be strictly positive, got {exact}; use None for unknown")
    return exact


def to_periodicity(obj: Any) -> tuple[bool, bool, bool]:
    """Normalize a periodicity specification into exactly three booleans.

    One flag per basis row, saying whether that row is a genuine lattice translation.
    ``None`` means fully periodic — the overwhelmingly common case, and the one every
    cell built before periodicity existed meant. Unlike a stated precision there is no
    "unknown" state: a cell you constructed has a periodicity you know.

    Anything three-element and truthy-testable works, so ``(True, True, False)``,
    ``[1, 1, 0]`` and OPTIMADE's own ``dimension_types`` spelling all land the same way.
    """
    if obj is None:
        return (True, True, True)
    if isinstance(obj, (str, bytes)) or not isinstance(obj, Iterable):
        raise ValueError(f"periodicity must be three flags, one per basis row, got {obj!r}")
    flags = tuple(bool(flag) for flag in obj)
    if len(flags) != 3:
        raise ValueError(f"periodicity must be three flags, one per basis row, got {len(flags)}")
    return (flags[0], flags[1], flags[2])


def to_float_tuples(vector: FracVector | SurdVector) -> tuple[tuple[float, ...], ...]:
    """Render an exact 2-D vector to nested float tuples (the primitive-view presentation)."""
    return tuple(tuple(row) for row in vector.to_floats())


def _is_empty_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple)) and len(obj) == 0


def is_basis_3x3(obj: Any) -> bool:
    """True iff ``obj`` builds a vector of shape ``(3, 3)`` (a cell basis)."""
    value = try_surdvector(obj)
    return value is not None and value.dim == (3, 3)


def is_coords_nx3(obj: Any) -> bool:
    """True iff ``obj`` builds a vector of shape ``(N, 3)`` (reduced coordinates); empty allowed."""
    if _is_empty_sequence(obj):
        return True
    value = try_surdvector(obj)
    return value is not None and len(value.dim) == 2 and value.dim[1] == 3


def is_params6(obj: Any) -> bool:
    """True iff ``obj`` builds a flat length-6 vector (cell parameters ``a,b,c,alpha,beta,gamma``)."""
    value = try_surdvector(obj)
    return value is not None and value.dim == (6,)
