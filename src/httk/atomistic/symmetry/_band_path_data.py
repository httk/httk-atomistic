"""HPKOT band-structure path tables, transcribed from SeeK-path.

Tables of the high-symmetry k-points, k-point parameters and suggested band-structure
paths for the 29 extended Bravais lattice symbols of Hinuma, Pizzi, Kumagai, Oba and
Tanaka, "Band structure diagram paths based on crystallography", Comput. Mater. Sci.
128, 140 (2017), DOI 10.1016/j.commatsci.2016.10.015 (the HPKOT convention), as
implemented in SeeK-path 2.2.2. Point coordinates are given in the reciprocal basis of
the HPKOT primitive cell. Parameters are exact rational functions of the HPKOT
conventional-cell metric (:class:`HpkotMetric`), rewritten from the SeeK-path
expressions without square roots; the original expression is kept as a comment.

The MIT License (MIT)

Copyright (c), 2016-2023, Giovanni Pizzi, PAUL SCHERRER INSTITUT (Laboratory
for Materials Simulations), ECOLE POLYTECHNIQUE FEDERALE DE
LAUSANNE (Theory and Simulation of Materials (THEOS) and National Centre for
Computational Design and Discovery of Novel Materials (NCCR MARVEL)),
Switzerland. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

__all__ = ["BAND_PATH_DATA", "BandPathEntry", "HpkotMetric", "Parameter"]


@dataclass(frozen=True, slots=True)
class HpkotMetric:
    """Exact metric entries of the HPKOT conventional cell.

    :param a2: Squared length of a (G11).
    :param b2: Squared length of b (G22).
    :param c2: Squared length of c (G33).
    :param ac_cosbeta: The a·c product (G13), i.e. a*c*cos(beta).
    """

    a2: Fraction
    b2: Fraction
    c2: Fraction
    ac_cosbeta: Fraction

    @property
    def sin2beta(self) -> Fraction:
        """Return sin²β."""
        return 1 - self.ac_cosbeta**2 / (self.a2 * self.c2)


type Parameter = Callable[[HpkotMetric, dict[str, Fraction]], Fraction]


@dataclass(frozen=True, slots=True)
class BandPathEntry:
    """One extended Bravais lattice's parameters, points and path.

    :param parameters: Ordered (name, function) pairs; each function receives the metric
        and the parameters computed so far.
    :param points: Ordered (label, coordinate strings) pairs, as listed by SeeK-path.
    :param path: Ordered (start, end) label segments.
    """

    parameters: tuple[tuple[str, Parameter], ...]
    points: tuple[tuple[str, tuple[str, str, str]], ...]
    path: tuple[tuple[str, str], ...]


BAND_PATH_DATA: dict[str, BandPathEntry] = {
    "aP2": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Z", ("0", "0", "1/2")),
            ("Y", ("0", "1/2", "0")),
            ("X", ("1/2", "0", "0")),
            ("V", ("1/2", "1/2", "0")),
            ("U", ("1/2", "0", "1/2")),
            ("T", ("0", "1/2", "1/2")),
            ("R", ("1/2", "1/2", "1/2")),
        ),
        path=(
            ("GAMMA", "X"),
            ("Y", "GAMMA"),
            ("GAMMA", "Z"),
            ("R", "GAMMA"),
            ("GAMMA", "T"),
            ("U", "GAMMA"),
            ("GAMMA", "V"),
        ),
    ),
    "aP3": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Z", ("0", "0", "1/2")),
            ("Y", ("0", "1/2", "0")),
            ("Y_2", ("0", "-1/2", "0")),
            ("X", ("1/2", "0", "0")),
            ("V_2", ("1/2", "-1/2", "0")),
            ("U_2", ("-1/2", "0", "1/2")),
            ("T_2", ("0", "-1/2", "1/2")),
            ("R_2", ("-1/2", "-1/2", "1/2")),
        ),
        path=(
            ("GAMMA", "X"),
            ("Y", "GAMMA"),
            ("GAMMA", "Z"),
            ("R_2", "GAMMA"),
            ("GAMMA", "T_2"),
            ("U_2", "GAMMA"),
            ("GAMMA", "V_2"),
        ),
    ),
    "cF1": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("X", ("1/2", "0", "1/2")),
            ("L", ("1/2", "1/2", "1/2")),
            ("W", ("1/2", "1/4", "3/4")),
            ("W_2", ("3/4", "1/4", "1/2")),
            ("K", ("3/8", "3/8", "3/4")),
            ("U", ("5/8", "1/4", "5/8")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "U"),
            ("K", "GAMMA"),
            ("GAMMA", "L"),
            ("L", "W"),
            ("W", "X"),
            ("X", "W_2"),
        ),
    ),
    "cF2": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("X", ("1/2", "0", "1/2")),
            ("L", ("1/2", "1/2", "1/2")),
            ("W", ("1/2", "1/4", "3/4")),
            ("W_2", ("3/4", "1/4", "1/2")),
            ("K", ("3/8", "3/8", "3/4")),
            ("U", ("5/8", "1/4", "5/8")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "U"),
            ("K", "GAMMA"),
            ("GAMMA", "L"),
            ("L", "W"),
            ("W", "X"),
        ),
    ),
    "cI1": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("H", ("1/2", "-1/2", "1/2")),
            ("P", ("1/4", "1/4", "1/4")),
            ("N", ("0", "0", "1/2")),
        ),
        path=(
            ("GAMMA", "H"),
            ("H", "N"),
            ("N", "GAMMA"),
            ("GAMMA", "P"),
            ("P", "H"),
            ("P", "N"),
        ),
    ),
    "cP1": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("R", ("1/2", "1/2", "1/2")),
            ("M", ("1/2", "1/2", "0")),
            ("X", ("0", "1/2", "0")),
            ("X_1", ("1/2", "0", "0")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "M"),
            ("M", "GAMMA"),
            ("GAMMA", "R"),
            ("R", "X"),
            ("R", "M"),
            ("M", "X_1"),
        ),
    ),
    "cP2": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("R", ("1/2", "1/2", "1/2")),
            ("M", ("1/2", "1/2", "0")),
            ("X", ("0", "1/2", "0")),
            ("X_1", ("1/2", "0", "0")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "M"),
            ("M", "GAMMA"),
            ("GAMMA", "R"),
            ("R", "X"),
            ("R", "M"),
        ),
    ),
    "hP1": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("A", ("0", "0", "1/2")),
            ("K", ("1/3", "1/3", "0")),
            ("H", ("1/3", "1/3", "1/2")),
            ("H_2", ("1/3", "1/3", "-1/2")),
            ("M", ("1/2", "0", "0")),
            ("L", ("1/2", "0", "1/2")),
        ),
        path=(
            ("GAMMA", "M"),
            ("M", "K"),
            ("K", "GAMMA"),
            ("GAMMA", "A"),
            ("A", "L"),
            ("L", "H"),
            ("H", "A"),
            ("L", "M"),
            ("H", "K"),
            ("K", "H_2"),
        ),
    ),
    "hP2": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("A", ("0", "0", "1/2")),
            ("K", ("1/3", "1/3", "0")),
            ("H", ("1/3", "1/3", "1/2")),
            ("H_2", ("1/3", "1/3", "-1/2")),
            ("M", ("1/2", "0", "0")),
            ("L", ("1/2", "0", "1/2")),
        ),
        path=(
            ("GAMMA", "M"),
            ("M", "K"),
            ("K", "GAMMA"),
            ("GAMMA", "A"),
            ("A", "L"),
            ("L", "H"),
            ("H", "A"),
            ("L", "M"),
            ("H", "K"),
        ),
    ),
    "hR1": BandPathEntry(
        parameters=(
            ("D", lambda m, p: m.a2 / (4 * m.c2)),  # a*a/4/c/c
            ("Y", lambda m, p: Fraction(5, 6) - 2 * p["D"]),  # 5/6-2*D
            ("N", lambda m, p: Fraction(1, 3) + p["D"]),  # 1/3+D
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("T", ("1/2", "1/2", "1/2")),
            ("L", ("1/2", "0", "0")),
            ("L_2", ("0", "-1/2", "0")),
            ("L_4", ("0", "0", "-1/2")),
            ("F", ("1/2", "0", "1/2")),
            ("F_2", ("1/2", "1/2", "0")),
            ("S_0", ("N", "-N", "0")),
            ("S_2", ("1-N", "0", "N")),
            ("S_4", ("N", "0", "-N")),
            ("S_6", ("1-N", "N", "0")),
            ("H_0", ("1/2", "-1+Y", "1-Y")),
            ("H_2", ("Y", "1-Y", "1/2")),
            ("H_4", ("Y", "1/2", "1-Y")),
            ("H_6", ("1/2", "1-Y", "-1+Y")),
            ("M_0", ("N", "-1+Y", "N")),
            ("M_2", ("1-N", "1-Y", "1-N")),
            ("M_4", ("Y", "N", "N")),
            ("M_6", ("1-N", "1-N", "1-Y")),
            ("M_8", ("N", "N", "-1+Y")),
        ),
        path=(
            ("GAMMA", "T"),
            ("T", "H_2"),
            ("H_0", "L"),
            ("L", "GAMMA"),
            ("GAMMA", "S_0"),
            ("S_2", "F"),
            ("F", "GAMMA"),
        ),
    ),
    "hR2": BandPathEntry(
        parameters=(
            ("Z", lambda m, p: Fraction(1, 6) - m.c2 / (9 * m.a2)),  # 1/6-c*c/9/a/a
            ("H", lambda m, p: Fraction(1, 2) - 2 * p["Z"]),  # 1/2-2*Z
            ("N", lambda m, p: Fraction(1, 2) + p["Z"]),  # 1/2+Z
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("T", ("1/2", "-1/2", "1/2")),
            ("P_0", ("H", "-1+H", "H")),
            ("P_2", ("H", "H", "H")),
            ("R_0", ("1-H", "-H", "-H")),
            ("M", ("1-N", "-N", "1-N")),
            ("M_2", ("N", "-1+N", "-1+N")),
            ("L", ("1/2", "0", "0")),
            ("F", ("1/2", "-1/2", "0")),
        ),
        path=(
            ("GAMMA", "L"),
            ("L", "T"),
            ("T", "P_0"),
            ("P_2", "GAMMA"),
            ("GAMMA", "F"),
        ),
    ),
    "mC1": BandPathEntry(
        parameters=(
            ("Z", lambda m, p: (2 + m.ac_cosbeta / m.c2) / (4 * m.sin2beta)),  # (2+a/c*cosbeta)/4/sinbeta/sinbeta
            ("H", lambda m, p: Fraction(1, 2) - 2 * p["Z"] * m.ac_cosbeta / m.a2),  # 1/2-2*Z*c*cosbeta/a
            ("S", lambda m, p: Fraction(3, 4) - m.b2 / (4 * m.a2 * m.sin2beta)),  # 3/4-b*b/4/a/a/sinbeta/sinbeta
            ("P", lambda m, p: p["S"] - (Fraction(3, 4) - p["S"]) * m.ac_cosbeta / m.c2),  # S-(3/4-S)*a*cosbeta/c
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y_2", ("-1/2", "1/2", "0")),
            ("Y_4", ("1/2", "-1/2", "0")),
            ("A", ("0", "0", "1/2")),
            ("M_2", ("-1/2", "1/2", "1/2")),
            ("V", ("1/2", "0", "0")),
            ("V_2", ("0", "1/2", "0")),
            ("L_2", ("0", "1/2", "1/2")),
            ("C", ("1-S", "1-S", "0")),
            ("C_2", ("-1+S", "S", "0")),
            ("C_4", ("S", "-1+S", "0")),
            ("D", ("-1+P", "P", "1/2")),
            ("D_2", ("1-P", "1-P", "1/2")),
            ("E", ("-1+Z", "1-Z", "1-H")),
            ("E_2", ("-Z", "Z", "H")),
            ("E_4", ("Z", "-Z", "1-H")),
        ),
        path=(
            ("GAMMA", "C"),
            ("C_2", "Y_2"),
            ("Y_2", "GAMMA"),
            ("GAMMA", "M_2"),
            ("M_2", "D"),
            ("D_2", "A"),
            ("A", "GAMMA"),
            ("L_2", "GAMMA"),
            ("GAMMA", "V_2"),
        ),
    ),
    "mC2": BandPathEntry(
        parameters=(
            (
                "Z",
                lambda m, p: (m.a2 / m.b2 + (1 + m.ac_cosbeta / m.c2) / m.sin2beta) / 4,
            ),  # (a*a/b/b+(1+a/c*cosbeta)/sinbeta/sinbeta)/4
            ("M", lambda m, p: (1 + m.a2 / m.b2) / 4),  # (1+a*a/b/b)/4
            ("D", lambda m, p: -m.ac_cosbeta / (2 * m.b2)),  # -a*c*cosbeta/2/b/b
            ("X", lambda m, p: Fraction(1, 2) - 2 * p["Z"] * m.ac_cosbeta / m.a2),  # 1/2-2*Z*c*cosbeta/a
            ("P", lambda m, p: 1 + p["Z"] - 2 * p["M"]),  # 1+Z-2*M
            ("S", lambda m, p: p["X"] - 2 * p["D"]),  # X-2*D
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y", ("1/2", "1/2", "0")),
            ("A", ("0", "0", "1/2")),
            ("M", ("1/2", "1/2", "1/2")),
            ("V_2", ("0", "1/2", "0")),
            ("L_2", ("0", "1/2", "1/2")),
            ("F", ("-1+P", "1-P", "1-S")),
            ("F_2", ("1-P", "P", "S")),
            ("F_4", ("P", "1-P", "1-S")),
            ("H", ("-Z", "Z", "X")),
            ("H_2", ("Z", "1-Z", "1-X")),
            ("H_4", ("Z", "-Z", "1-X")),
            ("G", ("-M", "M", "D")),
            ("G_2", ("M", "1-M", "-D")),
            ("G_4", ("M", "-M", "-D")),
            ("G_6", ("1-M", "M", "D")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "M"),
            ("M", "A"),
            ("A", "GAMMA"),
            ("L_2", "GAMMA"),
            ("GAMMA", "V_2"),
        ),
    ),
    "mC3": BandPathEntry(
        parameters=(
            (
                "Z",
                lambda m, p: (m.a2 / m.b2 + (1 + m.ac_cosbeta / m.c2) / m.sin2beta) / 4,
            ),  # (a*a/b/b+(1+a/c*cosbeta)/sinbeta/sinbeta)/4
            ("R", lambda m, p: 1 - p["Z"] * m.b2 / m.a2),  # 1-Z*b*b/a/a
            ("E", lambda m, p: Fraction(1, 2) - 2 * p["Z"] * m.ac_cosbeta / m.a2),  # 1/2-2*Z*c*cosbeta/a
            (
                "F",
                lambda m, p: p["E"] / 2 + m.a2 / (4 * m.b2) + m.ac_cosbeta / (2 * m.b2),
            ),  # E/2+a*a/4/b/b+a*c*cosbeta/2/b/b
            ("U", lambda m, p: 2 * p["F"] - p["Z"]),  # 2*F-Z
            (
                "W",
                lambda m, p: m.c2 / (2 * m.ac_cosbeta) * (1 - 4 * p["U"] + m.a2 * m.sin2beta / m.b2),
            ),  # c/2/a/cosbeta*(1-4*U+a*a*sinbeta*sinbeta/b/b)
            ("D", lambda m, p: Fraction(-1, 4) + p["W"] / 2 - p["Z"] * m.ac_cosbeta / m.a2),  # -1/4+W/2-Z*c*cosbeta/a
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y", ("1/2", "1/2", "0")),
            ("A", ("0", "0", "1/2")),
            ("M_2", ("-1/2", "1/2", "1/2")),
            ("V", ("1/2", "0", "0")),
            ("V_2", ("0", "1/2", "0")),
            ("L_2", ("0", "1/2", "1/2")),
            ("I", ("-1+R", "R", "1/2")),
            ("I_2", ("1-R", "1-R", "1/2")),
            ("K", ("-U", "U", "W")),
            ("K_2", ("-1+U", "1-U", "1-W")),
            ("K_4", ("1-U", "U", "W")),
            ("H", ("-Z", "Z", "E")),
            ("H_2", ("Z", "1-Z", "1-E")),
            ("H_4", ("Z", "-Z", "1-E")),
            ("N", ("-F", "F", "D")),
            ("N_2", ("F", "1-F", "-D")),
            ("N_4", ("F", "-F", "-D")),
            ("N_6", ("1-F", "F", "D")),
        ),
        path=(
            ("GAMMA", "A"),
            ("A", "I_2"),
            ("I", "M_2"),
            ("M_2", "GAMMA"),
            ("GAMMA", "Y"),
            ("L_2", "GAMMA"),
            ("GAMMA", "V_2"),
        ),
    ),
    "mP1": BandPathEntry(
        parameters=(
            ("Y", lambda m, p: (1 + m.ac_cosbeta / m.c2) / (2 * m.sin2beta)),  # (1+a/c*cosbeta)/2/sinbeta/sinbeta
            ("N", lambda m, p: Fraction(1, 2) + p["Y"] * m.ac_cosbeta / m.a2),  # 1/2+Y*c*cosbeta/a
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Z", ("0", "1/2", "0")),
            ("B", ("0", "0", "1/2")),
            ("B_2", ("0", "0", "-1/2")),
            ("Y", ("1/2", "0", "0")),
            ("Y_2", ("-1/2", "0", "0")),
            ("C", ("1/2", "1/2", "0")),
            ("C_2", ("-1/2", "1/2", "0")),
            ("D", ("0", "1/2", "1/2")),
            ("D_2", ("0", "1/2", "-1/2")),
            ("A", ("-1/2", "0", "1/2")),
            ("E", ("-1/2", "1/2", "1/2")),
            ("H", ("-Y", "0", "1-N")),
            ("H_2", ("-1+Y", "0", "N")),
            ("H_4", ("-Y", "0", "-N")),
            ("M", ("-Y", "1/2", "1-N")),
            ("M_2", ("-1+Y", "1/2", "N")),
            ("M_4", ("-Y", "1/2", "-N")),
        ),
        path=(
            ("GAMMA", "Z"),
            ("Z", "D"),
            ("D", "B"),
            ("B", "GAMMA"),
            ("GAMMA", "A"),
            ("A", "E"),
            ("E", "Z"),
            ("Z", "C_2"),
            ("C_2", "Y_2"),
            ("Y_2", "GAMMA"),
        ),
    ),
    "oA1": BandPathEntry(
        parameters=(
            ("X", lambda m, p: (1 + m.b2 / m.c2) / 4),  # (1+b*b/c/c)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y", ("-1/2", "1/2", "0")),
            ("T", ("-1/2", "1/2", "1/2")),
            ("Z", ("0", "0", "1/2")),
            ("S", ("0", "1/2", "0")),
            ("R", ("0", "1/2", "1/2")),
            ("SIGMA_0", ("X", "X", "0")),
            ("C_0", ("-X", "1-X", "0")),
            ("A_0", ("X", "X", "1/2")),
            ("E_0", ("-X", "1-X", "1/2")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "C_0"),
            ("SIGMA_0", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z", "A_0"),
            ("E_0", "T"),
            ("T", "Y"),
            ("GAMMA", "S"),
            ("S", "R"),
            ("R", "Z"),
            ("Z", "T"),
        ),
    ),
    "oA2": BandPathEntry(
        parameters=(
            ("X", lambda m, p: (1 + m.c2 / m.b2) / 4),  # (1+c*c/b/b)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y", ("1/2", "1/2", "0")),
            ("T", ("1/2", "1/2", "1/2")),
            ("T_2", ("1/2", "1/2", "-1/2")),
            ("Z", ("0", "0", "1/2")),
            ("Z_2", ("0", "0", "-1/2")),
            ("S", ("0", "1/2", "0")),
            ("R", ("0", "1/2", "1/2")),
            ("R_2", ("0", "1/2", "-1/2")),
            ("DELTA_0", ("-X", "X", "0")),
            ("F_0", ("X", "1-X", "0")),
            ("B_0", ("-X", "X", "1/2")),
            ("B_2", ("-X", "X", "-1/2")),
            ("G_0", ("X", "1-X", "1/2")),
            ("G_2", ("X", "1-X", "-1/2")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "F_0"),
            ("DELTA_0", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z", "B_0"),
            ("G_0", "T"),
            ("T", "Y"),
            ("GAMMA", "S"),
            ("S", "R"),
            ("R", "Z"),
            ("Z", "T"),
        ),
    ),
    "oC1": BandPathEntry(
        parameters=(
            ("X", lambda m, p: (1 + m.a2 / m.b2) / 4),  # (1+a*a/b/b)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y", ("-1/2", "1/2", "0")),
            ("T", ("-1/2", "1/2", "1/2")),
            ("Z", ("0", "0", "1/2")),
            ("S", ("0", "1/2", "0")),
            ("R", ("0", "1/2", "1/2")),
            ("SIGMA_0", ("X", "X", "0")),
            ("C_0", ("-X", "1-X", "0")),
            ("A_0", ("X", "X", "1/2")),
            ("E_0", ("-X", "1-X", "1/2")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "C_0"),
            ("SIGMA_0", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z", "A_0"),
            ("E_0", "T"),
            ("T", "Y"),
            ("GAMMA", "S"),
            ("S", "R"),
            ("R", "Z"),
            ("Z", "T"),
        ),
    ),
    "oC2": BandPathEntry(
        parameters=(
            ("X", lambda m, p: (1 + m.b2 / m.a2) / 4),  # (1+b*b/a/a)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Y", ("1/2", "1/2", "0")),
            ("T", ("1/2", "1/2", "1/2")),
            ("T_2", ("1/2", "1/2", "-1/2")),
            ("Z", ("0", "0", "1/2")),
            ("Z_2", ("0", "0", "-1/2")),
            ("S", ("0", "1/2", "0")),
            ("R", ("0", "1/2", "1/2")),
            ("R_2", ("0", "1/2", "-1/2")),
            ("DELTA_0", ("-X", "X", "0")),
            ("F_0", ("X", "1-X", "0")),
            ("B_0", ("-X", "X", "1/2")),
            ("B_2", ("-X", "X", "-1/2")),
            ("G_0", ("X", "1-X", "1/2")),
            ("G_2", ("X", "1-X", "-1/2")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "F_0"),
            ("DELTA_0", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z", "B_0"),
            ("G_0", "T"),
            ("T", "Y"),
            ("GAMMA", "S"),
            ("S", "R"),
            ("R", "Z"),
            ("Z", "T"),
        ),
    ),
    "oF1": BandPathEntry(
        parameters=(
            ("J", lambda m, p: (1 + m.a2 / m.b2 - m.a2 / m.c2) / 4),  # (1+a*a/b/b-a*a/c/c)/4
            ("H", lambda m, p: (1 + m.a2 / m.b2 + m.a2 / m.c2) / 4),  # (1+a*a/b/b+a*a/c/c)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("T", ("1", "1/2", "1/2")),
            ("Z", ("1/2", "1/2", "0")),
            ("Y", ("1/2", "0", "1/2")),
            ("SIGMA_0", ("0", "H", "H")),
            ("U_0", ("1", "1-H", "1-H")),
            ("A_0", ("1/2", "1/2+J", "J")),
            ("C_0", ("1/2", "1/2-J", "1-J")),
            ("L", ("1/2", "1/2", "1/2")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "T"),
            ("T", "Z"),
            ("Z", "GAMMA"),
            ("GAMMA", "SIGMA_0"),
            ("U_0", "T"),
            ("Y", "C_0"),
            ("A_0", "Z"),
            ("GAMMA", "L"),
        ),
    ),
    "oF2": BandPathEntry(
        parameters=(
            ("J", lambda m, p: (1 + m.c2 / m.a2 - m.c2 / m.b2) / 4),  # (1+c*c/a/a-c*c/b/b)/4
            ("K", lambda m, p: (1 + m.c2 / m.a2 + m.c2 / m.b2) / 4),  # (1+c*c/a/a+c*c/b/b)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("T", ("0", "1/2", "1/2")),
            ("Z", ("1/2", "1/2", "1")),
            ("Y", ("1/2", "0", "1/2")),
            ("LAMBDA_0", ("K", "K", "0")),
            ("Q_0", ("1-K", "1-K", "1")),
            ("G_0", ("1/2-J", "1-J", "1/2")),
            ("H_0", ("1/2+J", "J", "1/2")),
            ("L", ("1/2", "1/2", "1/2")),
        ),
        path=(
            ("GAMMA", "T"),
            ("T", "Z"),
            ("Z", "Y"),
            ("Y", "GAMMA"),
            ("GAMMA", "LAMBDA_0"),
            ("Q_0", "Z"),
            ("T", "G_0"),
            ("H_0", "Y"),
            ("GAMMA", "L"),
        ),
    ),
    "oF3": BandPathEntry(
        parameters=(
            ("H", lambda m, p: (1 + m.a2 / m.b2 - m.a2 / m.c2) / 4),  # (1+a*a/b/b-a*a/c/c)/4
            ("K", lambda m, p: (1 + m.b2 / m.a2 - m.b2 / m.c2) / 4),  # (1+b*b/a/a-b*b/c/c)/4
            ("P", lambda m, p: (1 + m.c2 / m.b2 - m.c2 / m.a2) / 4),  # (1+c*c/b/b-c*c/a/a)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("T", ("0", "1/2", "1/2")),
            ("Z", ("1/2", "1/2", "0")),
            ("Y", ("1/2", "0", "1/2")),
            ("A_0", ("1/2", "1/2+H", "H")),
            ("C_0", ("1/2", "1/2-H", "1-H")),
            ("B_0", ("1/2+K", "1/2", "K")),
            ("D_0", ("1/2-K", "1/2", "1-K")),
            ("G_0", ("P", "1/2+P", "1/2")),
            ("H_0", ("1-P", "1/2-P", "1/2")),
            ("L", ("1/2", "1/2", "1/2")),
        ),
        path=(
            ("GAMMA", "Y"),
            ("Y", "C_0"),
            ("A_0", "Z"),
            ("Z", "B_0"),
            ("D_0", "T"),
            ("T", "G_0"),
            ("H_0", "Y"),
            ("T", "GAMMA"),
            ("GAMMA", "Z"),
            ("GAMMA", "L"),
        ),
    ),
    "oI1": BandPathEntry(
        parameters=(
            ("Z", lambda m, p: (1 + m.a2 / m.c2) / 4),  # (1+a*a/c/c)/4
            ("H", lambda m, p: (1 + m.b2 / m.c2) / 4),  # (1+b*b/c/c)/4
            ("D", lambda m, p: (m.b2 - m.a2) / (4 * m.c2)),  # (b*b-a*a)/4/c/c
            ("N", lambda m, p: (m.a2 + m.b2) / (4 * m.c2)),  # (a*a+b*b)/4/c/c
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("X", ("1/2", "1/2", "-1/2")),
            ("S", ("1/2", "0", "0")),
            ("R", ("0", "1/2", "0")),
            ("T", ("0", "0", "1/2")),
            ("W", ("1/4", "1/4", "1/4")),
            ("SIGMA_0", ("-Z", "Z", "Z")),
            ("F_2", ("Z", "1-Z", "-Z")),
            ("Y_0", ("H", "-H", "H")),
            ("U_0", ("1-H", "H", "-H")),
            ("L_0", ("-N", "N", "1/2-D")),
            ("M_0", ("N", "-N", "1/2+D")),
            ("J_0", ("1/2-D", "1/2+D", "-N")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "F_2"),
            ("SIGMA_0", "GAMMA"),
            ("GAMMA", "Y_0"),
            ("U_0", "X"),
            ("GAMMA", "R"),
            ("R", "W"),
            ("W", "S"),
            ("S", "GAMMA"),
            ("GAMMA", "T"),
            ("T", "W"),
        ),
    ),
    "oI2": BandPathEntry(
        parameters=(
            ("Z", lambda m, p: (1 + m.b2 / m.a2) / 4),  # (1+b*b/a/a)/4
            ("H", lambda m, p: (1 + m.c2 / m.a2) / 4),  # (1+c*c/a/a)/4
            ("D", lambda m, p: (m.c2 - m.b2) / (4 * m.a2)),  # (c*c-b*b)/4/a/a
            ("N", lambda m, p: (m.b2 + m.c2) / (4 * m.a2)),  # (b*b+c*c)/4/a/a
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("X", ("-1/2", "1/2", "1/2")),
            ("S", ("1/2", "0", "0")),
            ("R", ("0", "1/2", "0")),
            ("T", ("0", "0", "1/2")),
            ("W", ("1/4", "1/4", "1/4")),
            ("Y_0", ("Z", "-Z", "Z")),
            ("U_2", ("-Z", "Z", "1-Z")),
            ("LAMBDA_0", ("H", "H", "-H")),
            ("G_2", ("-H", "1-H", "H")),
            ("K", ("1/2-D", "-N", "N")),
            ("K_2", ("1/2+D", "N", "-N")),
            ("K_4", ("-N", "1/2-D", "1/2+D")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "U_2"),
            ("Y_0", "GAMMA"),
            ("GAMMA", "LAMBDA_0"),
            ("G_2", "X"),
            ("GAMMA", "R"),
            ("R", "W"),
            ("W", "S"),
            ("S", "GAMMA"),
            ("GAMMA", "T"),
            ("T", "W"),
        ),
    ),
    "oI3": BandPathEntry(
        parameters=(
            ("Z", lambda m, p: (1 + m.c2 / m.b2) / 4),  # (1+c*c/b/b)/4
            ("Y", lambda m, p: (1 + m.a2 / m.b2) / 4),  # (1+a*a/b/b)/4
            ("D", lambda m, p: (m.a2 - m.c2) / (4 * m.b2)),  # (a*a-c*c)/4/b/b
            ("M", lambda m, p: (m.c2 + m.a2) / (4 * m.b2)),  # (c*c+a*a)/4/b/b
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("X", ("1/2", "-1/2", "1/2")),
            ("S", ("1/2", "0", "0")),
            ("R", ("0", "1/2", "0")),
            ("T", ("0", "0", "1/2")),
            ("W", ("1/4", "1/4", "1/4")),
            ("SIGMA_0", ("-Y", "Y", "Y")),
            ("F_0", ("Y", "-Y", "1-Y")),
            ("LAMBDA_0", ("Z", "Z", "-Z")),
            ("G_0", ("1-Z", "-Z", "Z")),
            ("V_0", ("M", "1/2-D", "-M")),
            ("H_0", ("-M", "1/2+D", "M")),
            ("H_2", ("1/2+D", "-M", "1/2-D")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "F_0"),
            ("SIGMA_0", "GAMMA"),
            ("GAMMA", "LAMBDA_0"),
            ("G_0", "X"),
            ("GAMMA", "R"),
            ("R", "W"),
            ("W", "S"),
            ("S", "GAMMA"),
            ("GAMMA", "T"),
            ("T", "W"),
        ),
    ),
    "oP1": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("X", ("1/2", "0", "0")),
            ("Z", ("0", "0", "1/2")),
            ("U", ("1/2", "0", "1/2")),
            ("Y", ("0", "1/2", "0")),
            ("S", ("1/2", "1/2", "0")),
            ("T", ("0", "1/2", "1/2")),
            ("R", ("1/2", "1/2", "1/2")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "S"),
            ("S", "Y"),
            ("Y", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z", "U"),
            ("U", "R"),
            ("R", "T"),
            ("T", "Z"),
            ("X", "U"),
            ("Y", "T"),
            ("S", "R"),
        ),
    ),
    "tI1": BandPathEntry(
        parameters=(
            ("H", lambda m, p: (1 + m.c2 / m.a2) / 4),  # (1+c*c/a/a)/4
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("M", ("-1/2", "1/2", "1/2")),
            ("X", ("0", "0", "1/2")),
            ("P", ("1/4", "1/4", "1/4")),
            ("Z", ("H", "H", "-H")),
            ("Z_0", ("-H", "1-H", "H")),
            ("N", ("0", "1/2", "0")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "M"),
            ("M", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z_0", "M"),
            ("X", "P"),
            ("P", "N"),
            ("N", "GAMMA"),
        ),
    ),
    "tI2": BandPathEntry(
        parameters=(
            ("H", lambda m, p: (1 + m.a2 / m.c2) / 4),  # (1+a*a/c/c)/4
            ("Z", lambda m, p: m.a2 / (2 * m.c2)),  # a*a/2/c/c
        ),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("M", ("1/2", "1/2", "-1/2")),
            ("X", ("0", "0", "1/2")),
            ("P", ("1/4", "1/4", "1/4")),
            ("N", ("0", "1/2", "0")),
            ("S_0", ("-H", "H", "H")),
            ("S", ("H", "1-H", "-H")),
            ("R", ("-Z", "Z", "1/2")),
            ("G", ("1/2", "1/2", "-Z")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "P"),
            ("P", "N"),
            ("N", "GAMMA"),
            ("GAMMA", "M"),
            ("M", "S"),
            ("S_0", "GAMMA"),
            ("X", "R"),
            ("G", "M"),
        ),
    ),
    "tP1": BandPathEntry(
        parameters=(),
        points=(
            ("GAMMA", ("0", "0", "0")),
            ("Z", ("0", "0", "1/2")),
            ("M", ("1/2", "1/2", "0")),
            ("A", ("1/2", "1/2", "1/2")),
            ("R", ("0", "1/2", "1/2")),
            ("X", ("0", "1/2", "0")),
        ),
        path=(
            ("GAMMA", "X"),
            ("X", "M"),
            ("M", "GAMMA"),
            ("GAMMA", "Z"),
            ("Z", "R"),
            ("R", "A"),
            ("A", "Z"),
            ("X", "R"),
            ("M", "A"),
        ),
    ),
}
