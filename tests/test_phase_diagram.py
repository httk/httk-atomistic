"""Tests for float-LP phase-diagram construction and plotting."""

import pytest

numpy = pytest.importorskip("numpy")

from httk.atomistic import PhaseDiagram, Species, Structure
from httk.atomistic.phase_diagram import _solve_equality_lp

CUBIC = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]


def test_equality_simplex_unique_optimum() -> None:
    value, weights = _solve_equality_lp([1.0, 2.0], [[1.0, 1.0]], [1.0])
    assert value == pytest.approx(1.0)
    assert weights == pytest.approx((1.0, 0.0))


def test_equality_simplex_bland_rule_handles_multirow_degenerate_pivots() -> None:
    # Beale's cycling example, expressed with three equality slacks. The first
    # two basic slack values start at zero, so this exercises genuinely
    # degenerate multi-row pivots rather than merely a non-unique objective.
    value, weights = _solve_equality_lp(
        [-10.0, 57.0, 9.0, 24.0, 0.0, 0.0, 0.0],
        [
            [0.5, -5.5, -2.5, 9.0, 1.0, 0.0, 0.0],
            [0.5, -1.5, -0.5, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [0.0, 0.0, 1.0],
    )
    assert value == pytest.approx(-1.0)
    assert weights[:4] == pytest.approx((1.0, 0.0, 1.0, 0.0))


def test_equality_simplex_reports_infeasible_problem() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        _solve_equality_lp([1.0], [[1.0], [1.0]], [1.0, 2.0])


def test_equality_simplex_guards_unbounded_problem() -> None:
    # Hull objectives cannot be unbounded because sum(w) == 1 and w >= 0. Keep the
    # general solver guard covered with a free negative-cost variable.
    with pytest.raises(ValueError, match="unbounded"):
        _solve_equality_lp([-1.0], [[0.0]], [0.0])


@pytest.mark.parametrize("epsilon", [2e-10, 1e-9, 2e-9])
def test_equality_simplex_near_degenerate_ternary_regression(
    epsilon: float,
) -> None:
    points = numpy.asarray(
        [
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.25 - epsilon, 0.25, 0.5 + epsilon),
            (0.75, 0.25, 0.0),
        ],
        dtype=numpy.float64,
    )
    target = 0.9 * points[0] + 0.05 * points[1] + 0.05 * points[2]
    matrix = numpy.vstack((points.T, numpy.ones(4)))

    value, weights = _solve_equality_lp(
        [0.0, 0.0, 0.0, -1.0],
        matrix,
        [*target, 1.0],
    )

    assert numpy.isfinite(value)
    assert numpy.asarray(weights) @ points == pytest.approx(target, abs=1e-10)
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize("epsilon", [2e-10, 1e-9, 2e-9])
def test_phase_diagram_near_degenerate_ternary_regression(
    epsilon: float,
) -> None:
    points = [
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.25 - epsilon, 0.25, 0.5 + epsilon),
        (0.75, 0.25, 0.0),
    ]
    target = tuple(
        0.9 * points[0][axis] + 0.05 * points[1][axis] + 0.05 * points[2][axis]
        for axis in range(3)
    )
    compositions: list[dict[str, float]] = [
        dict(zip(("A", "B", "C"), point, strict=True)) for point in [*points, target]
    ]

    diagram = PhaseDiagram.from_compositions(
        compositions,
        [0.0, 0.0, 0.0, -1.0, 0.0],
    )

    assert len(diagram) == 5
    assert diagram.elements == ("A", "B", "C")


def test_equality_simplex_ratio_test_and_row_scaling_invariance() -> None:
    matrix = [[1.0, 0.0, 1.0], [0.0, 1.0, 1e12]]
    rhs = [1.0 + 5e-12, 1e12]
    costs = [0.0, 0.0, -1.0]

    value, weights = _solve_equality_lp(costs, matrix, rhs)
    scaled_value, scaled_weights = _solve_equality_lp(
        costs,
        [[1.0, 0.0, 1.0], [0.0, 1e-12, 1.0]],
        [1.0 + 5e-12, 1.0],
    )

    assert value == pytest.approx(-1.0)
    assert weights == pytest.approx((5e-12, 0.0, 1.0), abs=1e-15)
    assert scaled_value == pytest.approx(value)
    assert scaled_weights == pytest.approx(weights)


def test_binary_hull_distances_decomposition_and_subsumed_line() -> None:
    # In per-atom coordinates A=(0, 0), B=(1, 0), AB=(0.5, -1), and
    # AB3=(0.75, -0.25). At x_B=0.75 the AB--B segment is -0.5, so AB3 is
    # 0.25 above hull and decomposes as 0.5 AB + 0.5 B.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}],
        [0.0, 0.0, -2.0, -1.0],
        ["A", "B", "AB", "AB3"],
    )

    assert diagram.elements == ("A", "B")
    assert diagram.energies_per_atom == pytest.approx((0.0, 0.0, -1.0, -0.25))
    assert diagram.hull_indices == (0, 1, 2)
    assert diagram.energy_above_hull == pytest.approx((0.0, 0.0, 0.0, 0.25), abs=1e-9)
    decomposition = diagram.decomposition(3)
    assert decomposition is not None
    assert tuple(index for index, _ in decomposition) == (1, 2)
    assert tuple(weight for _, weight in decomposition) == pytest.approx((0.5, 0.5))
    assert diagram.phase_lines == ((0, 2), (1, 2))


def test_ternary_interior_compound_has_six_supported_lines() -> None:
    # A, B, and C have energy 0; ABC is at the barycentre with e=-1/atom.
    # Each element edge is forced by its zero third-element coordinate, while each
    # element--ABC pair is uniquely supported toward the lowered barycentre. The
    # midpoint test therefore gives all six pairs of the four stable phases.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"C": 1}, {"A": 1, "B": 1, "C": 1}],
        [0.0, 0.0, 0.0, -3.0],
    )

    assert diagram.hull_indices == (0, 1, 2, 3)
    assert diagram.phase_lines == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_midpoint_lines_fix_v1_single_decomposition_neighbor_gap() -> None:
    # A/B/C are the zero-energy corners. AB and AC are at -1/atom. The two
    # compounds subsume the long A--B and A--C segments. Every other pair is
    # midpoint-supported: for example midpoint(B, AC) can also be represented
    # by 0.5 AB + 0.25 B + 0.25 C, at the same -0.5 energy.
    #
    # A v1-style single competitor decomposition for AB finds only A+B, and the
    # one for AC only A+C; the pure corners are composition-extreme. That graph
    # therefore misses B--C, AB--AC, B--AC, and C--AB, all found here.
    diagram = PhaseDiagram.from_compositions(
        [
            {"A": 1},
            {"B": 1},
            {"C": 1},
            {"A": 1, "B": 1},
            {"A": 1, "C": 1},
        ],
        [0.0, 0.0, 0.0, -2.0, -2.0],
    )

    assert diagram.hull_indices == (0, 1, 2, 3, 4)
    assert diagram.phase_lines == (
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    )


def test_uncontested_compositions_are_stable() -> None:
    # Neither endpoint x_B=0 nor x_B=0.5 can be represented by the other alone.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"A": 1, "B": 1}],
        [0.0, -1.0],
    )

    assert diagram.hull_indices == (0, 1)
    assert diagram.energy_above_hull == (0.0, 0.0)
    assert diagram.decomposition(1) is None


def test_duplicate_composition_keeps_only_lower_polymorph_stable() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1, "B": 1}, {"B": 2, "A": 2}],
        [-2.0, -3.0],
        ["low", "high"],
    )

    assert diagram.energies_per_atom == pytest.approx((-1.0, -0.75))
    assert diagram.hull_indices == (0,)
    assert diagram.energy_above_hull == pytest.approx((0.0, 0.25))
    assert diagram.decomposition(1) == ((0, 1.0),)
    assert diagram.phase_lines == ()


def test_duplicate_compositions_tied_within_tolerance_are_all_stable() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"A": 1}],
        [0.0, 1e-12],
    )
    assert diagram.hull_indices == (0, 1)
    assert diagram.energy_above_hull == pytest.approx((0.0, 1e-12))


def test_nonfinite_atom_count_is_rejected_before_normalization() -> None:
    with pytest.raises(ValueError, match="finite atom count"):
        PhaseDiagram.from_compositions(
            [{"A": 1e308, "B": 1e308}],
            [0.0],
        )


def test_nonfinite_energy_per_atom_is_rejected() -> None:
    with pytest.raises(ValueError, match="energy per atom must be finite"):
        PhaseDiagram.from_compositions(
            [{"A": 1e-320}],
            [1e308],
        )


def test_from_structures_weights_disorder_and_ignores_vacancy() -> None:
    mixed = Structure(
        CUBIC,
        [[0, 0, 0]],
        [Species("mix", ("Fe", "Ni"), (0.5, 0.5))],
        ["mix"],
    )
    lithium_with_vacancy = Structure(
        CUBIC,
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [
            Species("Li", ("Li",), (1.0,)),
            Species("vac", ("vacancy",), (1.0,)),
        ],
        ["Li", "vac"],
    )

    diagram = PhaseDiagram.from_structures([mixed, lithium_with_vacancy], [0.0, 0.0])

    assert diagram.elements == ("Fe", "Li", "Ni")
    assert diagram.compositions == ((0.5, 0.0, 0.5), (0.0, 1.0, 0.0))
    assert diagram.ids == ("Fe0.5Ni0.5", "Li")


def test_from_structures_rejects_unknown_element() -> None:
    structure = Structure(
        CUBIC,
        [[0, 0, 0]],
        [Species("unknown", ("X",), (1.0,))],
        ["unknown"],
    )
    with pytest.raises(ValueError, match="unknown chemical symbol"):
        PhaseDiagram.from_structures([structure], [0.0])


def test_from_structures_rejects_negative_species_concentration() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Species("negative", ("Fe",), (-0.5,))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Species("positive", ("Fe",), (1.5,))


def test_default_tolerance_counts_tiny_positive_distance_as_stable() -> None:
    # AB is 1e-12/atom above the A--B segment, well inside the default 1e-8.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}],
        [0.0, 0.0, 2e-12],
    )
    assert diagram.hull_indices == (0, 1, 2)
    assert diagram.energy_above_hull[2] == pytest.approx(1e-12)


def test_distinct_nearby_compositions_receive_a_tie_line() -> None:
    diagram = PhaseDiagram.from_compositions(
        [
            {"A": 1.0},
            {"A": 1.0 - 5e-10, "B": 5e-10},
        ],
        [0.0, 0.0],
    )

    assert diagram.compositions[0] != diagram.compositions[1]
    assert diagram.phase_lines == ((0, 1),)


def test_subsumption_requires_middle_phase_on_energy_segment() -> None:
    # M lies at x_B=0.25 and 1.4e-8 below the A--B energy segment. At the A--B
    # midpoint, 2/3 M + 1/3 B has energy -9.333...e-9, so the requested 1e-8
    # stability tolerance still admits A--B as midpoint-supported. M is not on
    # the A--B energy segment within tolerance, however, and therefore must not
    # subsume that long line. All three pair lines are reported.
    diagram = PhaseDiagram.from_compositions(
        [
            {"A": 1.0},
            {"B": 1.0},
            {"A": 0.75, "B": 0.25},
        ],
        [0.0, 0.0, -1.4e-8],
        tolerance=1e-8,
    )

    assert diagram.hull_indices == (0, 1, 2)
    assert diagram.phase_lines == ((0, 1), (0, 2), (1, 2))


def test_binary_and_ternary_plot_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binary = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}],
        [0.0, 0.0, -2.0, -1.0],
    )
    binary_axes = binary.plot()
    assert len(binary_axes.lines) >= len(binary.phase_lines)
    assert tuple(binary_axes.lines[0].get_xdata()) == pytest.approx((0.0, 0.5))
    assert tuple(binary_axes.lines[0].get_ydata()) == pytest.approx((0.0, -1.0))
    assert tuple(binary_axes.lines[1].get_xdata()) == pytest.approx((0.5, 1.0))
    assert tuple(binary_axes.lines[1].get_ydata()) == pytest.approx((-1.0, 0.0))

    supplied_figure, supplied_axes = plt.subplots()
    figure_numbers = tuple(plt.get_fignums())
    assert binary.plot(ax=supplied_axes) is supplied_axes
    assert tuple(plt.get_fignums()) == figure_numbers

    ternary = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"C": 1}, {"A": 1, "B": 1, "C": 1}],
        [0.0, 0.0, 0.0, -3.0],
    )
    ternary_axes = ternary.plot(label_stable=False)
    assert len(ternary_axes.lines) >= len(ternary.phase_lines)
    plt.close(binary_axes.figure)
    plt.close(supplied_figure)
    plt.close(ternary_axes.figure)
