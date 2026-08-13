"""Canonical keys for complete space-group operation sets."""

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from .affine_operation import AffineOperation

__all__ = ["symop_key_v1"]


def symop_key_v1(operations: Iterable[AffineOperation | Mapping[str, Any]]) -> str:
    """Return the cross-repository v1 SHA-256 key for complete symmetry operations.

    This mirrors ``data-generators/generate_basics_hall.py:2957``: serialize each full
    (centering-folded) operation as its integer 3x3 rotation and translation reduced
    modulo one, deduplicate and lexically sort the serializations, then hash their
    pipe-join as UTF-8. The ``p_1`` first record keys to
    ``63dbcdb54bd5d8c35ce8ae32cb34369717b95ee5d3c49dba36f5bbf9bc800048``.

    :param operations: Exact affine operations or vendored affine-operation records.
    :return: The canonical v1 lowercase SHA-256 hexadecimal key.
    :raises ValueError: If a rotation coefficient is not an exact integer.
    """
    serialized = set()
    for entry in operations:
        operation = entry if isinstance(entry, AffineOperation) else AffineOperation.from_record(entry)
        matrix = []
        for row in operation.matrix.to_fractions():
            for value in row:
                if value.denominator != 1:
                    raise ValueError(f"symop-key v1 requires integer rotation entries, got {value}")
                matrix.append(str(value.numerator))
        translation = [str(value % 1) for value in operation.vector.to_fractions()]
        serialized.add(f"{','.join(matrix)};{','.join(translation)}")
    return hashlib.sha256("|".join(sorted(serialized)).encode()).hexdigest()
