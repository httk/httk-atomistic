"""
Shared vector-family acceptance guards and normalizers for the atomistic backends.

The cell/sites/structure backends accept an input if it can be built through the httk-core
exact-vector family (:class:`~httk.core.FracVector` / :class:`~httk.core.SurdVector`) at the
required shape. Acceptance is therefore "does ``SurdVector.create`` succeed and land on the right
``dim``", replacing the ad-hoc ``_is_number``/``_is_3x3``/``_is_nx3``/``_is_params`` predicates. This
uniformly admits :class:`~fractions.Fraction`, rational strings (``"1/3"``), ``FracVector``,
``SurdVector``, and numpy arrays alongside the plain nested lists/tuples of numbers.
"""

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
