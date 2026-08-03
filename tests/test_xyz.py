from fractions import Fraction

import pytest

from httk.atomistic.symmetry import (
    Spacegroup,
    operation_from_xyz,
    operation_from_xyzt,
    parse_linear_expression,
)


@pytest.mark.parametrize(
    ("text", "vector"),
    (
        ("x,y,z", (0, 0, 0)),
        ("-x+1/2,y-0.25,-z-1.25", (Fraction(1, 2), Fraction(-1, 4), Fraction(-5, 4))),
        ("x+0.3333,y,z", (Fraction(3333, 10000), 0, 0)),
    ),
)
def test_xyz_is_exact(text: str, vector: tuple[Fraction | int, ...]) -> None:
    assert operation_from_xyz(text).vector.to_fractions() == list(vector)


def test_xyzt_returns_time_reversal() -> None:
    operation, time = operation_from_xyzt("x,y,z,-1")
    assert operation.is_identity()
    assert time == -1


def test_xyzt_rejects_invalid_time_reversal() -> None:
    with pytest.raises(ValueError, match="Time-reversal flag"):
        operation_from_xyzt("1x,1y,1z,+3")


def test_superspace_expression_returns_six_integer_coefficients() -> None:
    assert parse_linear_expression("x1-2x2+x6+1/3") == ((1, -2, 0, 0, 0, 1), Fraction(1, 3))


def test_xyz_round_trip_closes_for_a_spacegroup() -> None:
    operations = Spacegroup.standard(225).symmetry_operations
    assert all(operation_from_xyz(operation.wrapped().to_xyz()) == operation.wrapped() for operation in operations)
