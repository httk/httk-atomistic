"""Tests for exact HPKOT band-structure paths."""

import fractions
import itertools
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    ASUStructure,
    BandPathResult,
    CartesianSiteMoments,
    Cell,
    Spacegroup,
    Species,
    UnitcellStructure,
    WyckoffSite,
    band_path,
    primitive_cell,
    recognize_asu,
    same_crystal,
)
from httk.atomistic.reduction import _is_niggli_parameters, _parameters
from httk.atomistic.symmetry._band_path_data import BAND_PATH_DATA
from httk.atomistic.symmetry.band_paths import (
    _HPKOT_COLUMN_PRIMITIVE,
    _coordinate,
    _preserves_setting,
    _sqrt_sum_sign,
)

F = fractions.Fraction
CUBIC = [5, 5, 5, 90, 90, 90]
_FREE_VALUES = (F(1, 8), F(1, 5), F(2, 7))

#: One exact-ASU representative per HPKOT extended Bravais symbol:
#: ``(symbol, IT number, cell, sites)``. ``cell`` is accepted by :class:`Cell` (six lattice
#: parameters or three basis rows); each site is ``(Wyckoff letter, species)`` and its free
#: parameters, if any, are taken in order from ``_FREE_VALUES``.
REPRESENTATIVES: tuple[tuple[str, int, list[Any], tuple[tuple[str, str], ...]], ...] = (
    ("aP2", 2, [[4, 0, 0], [1, 5, 0], [1, 2, 6]], (("a", "C"),)),
    ("aP3", 2, [[4, 0, 0], [-1, 5, 0], [-1, -2, 6]], (("a", "C"),)),
    ("cF1", 196, CUBIC, (("a", "Zn"), ("c", "S"))),
    ("cF2", 225, CUBIC, (("a", "Na"), ("b", "Cl"))),
    ("cI1", 229, CUBIC, (("a", "W"),)),
    ("cP1", 195, CUBIC, (("a", "C"),)),
    ("cP2", 221, CUBIC, (("a", "Cs"), ("b", "Cl"))),
    ("hP1", 162, [4, 4, 6, 90, 90, 120], (("a", "C"),)),
    ("hP2", 191, [4, 4, 6, 90, 90, 120], (("a", "C"),)),
    ("hR1", 166, [4, 4, 12, 90, 90, 120], (("a", "C"),)),
    ("hR2", 166, [4, 4, 3, 90, 90, 120], (("a", "C"),)),
    ("mC1", 12, [[8, 0, 0], [0, 4, 0], [-2, 0, 7]], (("a", "C"),)),
    ("mC2", 12, [[4, 0, 0], [0, 8, 0], [-1, 0, 6]], (("a", "C"),)),
    ("mC3", 12, [[6, 0, 0], [0, 6, 0], [-2, 0, 5]], (("a", "C"),)),
    ("mP1", 10, [[5, 0, 0], [0, 6, 0], [-2, 0, 7]], (("a", "C"),)),
    ("oA1", 38, [3, 4, 5, 90, 90, 90], (("a", "C"),)),
    ("oA2", 38, [3, 5, 4, 90, 90, 90], (("a", "C"),)),
    ("oC1", 63, [4, 6, 5, 90, 90, 90], (("a", "C"),)),
    ("oC2", 63, [6, 4, 5, 90, 90, 90], (("a", "C"),)),
    ("oF1", 69, [3, 5, 6, 90, 90, 90], (("a", "C"),)),
    ("oF2", 42, [5, 6, 3, 90, 90, 90], (("a", "C"),)),
    ("oF3", 69, [4, 5, 6, 90, 90, 90], (("a", "C"),)),
    ("oI1", 71, [3, 4, 5, 90, 90, 90], (("a", "C"),)),
    ("oI2", 46, [7, 5, 6, 90, 90, 90], (("a", "C"),)),
    ("oI3", 46, [5, 7, 6, 90, 90, 90], (("a", "C"),)),
    ("oP1", 47, [3, 4, 5, 90, 90, 90], (("a", "C"),)),
    ("tI1", 139, [5, 5, 4, 90, 90, 90], (("a", "C"),)),
    ("tI2", 139, [4, 4, 7, 90, 90, 90], (("a", "C"),)),
    ("tP1", 123, [4, 4, 5, 90, 90, 90], (("a", "C"),)),
)


#: Golden exact output per representative, validated against SeeK-path 2.2.2 on same-lattice
#: crystals whose declared group equals the detected one: the metric of ``structure``
#: (row-major) and every point (label -> coordinates).
GOLDEN: dict[str, tuple[str, dict[str, str]]] = {
    "aP2": (
        "16 4 4 4 41 11 4 11 26",
        {
            "GAMMA": "0 0 0",
            "Z": "0 0 1/2",
            "Y": "0 1/2 0",
            "X": "1/2 0 0",
            "V": "1/2 1/2 0",
            "U": "1/2 0 1/2",
            "T": "0 1/2 1/2",
            "R": "1/2 1/2 1/2",
        },
    ),
    "aP3": (
        "16 -4 -4 -4 41 -9 -4 -9 26",
        {
            "GAMMA": "0 0 0",
            "Z": "0 0 1/2",
            "Y": "0 1/2 0",
            "Y_2": "0 -1/2 0",
            "X": "1/2 0 0",
            "V_2": "1/2 -1/2 0",
            "U_2": "-1/2 0 1/2",
            "T_2": "0 -1/2 1/2",
            "R_2": "-1/2 -1/2 1/2",
        },
    ),
    "cF1": (
        "25/2 25/4 25/4 25/4 25/2 25/4 25/4 25/4 25/2",
        {
            "GAMMA": "0 0 0",
            "X": "1/2 0 1/2",
            "L": "1/2 1/2 1/2",
            "W": "1/2 1/4 3/4",
            "W_2": "3/4 1/4 1/2",
            "K": "3/8 3/8 3/4",
            "U": "5/8 1/4 5/8",
        },
    ),
    "cF2": (
        "25/2 25/4 25/4 25/4 25/2 25/4 25/4 25/4 25/2",
        {
            "GAMMA": "0 0 0",
            "X": "1/2 0 1/2",
            "L": "1/2 1/2 1/2",
            "W": "1/2 1/4 3/4",
            "W_2": "3/4 1/4 1/2",
            "K": "3/8 3/8 3/4",
            "U": "5/8 1/4 5/8",
        },
    ),
    "cI1": (
        "75/4 -25/4 -25/4 -25/4 75/4 -25/4 -25/4 -25/4 75/4",
        {"GAMMA": "0 0 0", "H": "1/2 -1/2 1/2", "P": "1/4 1/4 1/4", "N": "0 0 1/2"},
    ),
    "cP1": (
        "25 0 0 0 25 0 0 0 25",
        {"GAMMA": "0 0 0", "R": "1/2 1/2 1/2", "M": "1/2 1/2 0", "X": "0 1/2 0", "X_1": "1/2 0 0"},
    ),
    "cP2": (
        "25 0 0 0 25 0 0 0 25",
        {"GAMMA": "0 0 0", "R": "1/2 1/2 1/2", "M": "1/2 1/2 0", "X": "0 1/2 0", "X_1": "1/2 0 0"},
    ),
    "hP1": (
        "16 -8 0 -8 16 0 0 0 36",
        {
            "GAMMA": "0 0 0",
            "A": "0 0 1/2",
            "K": "1/3 1/3 0",
            "H": "1/3 1/3 1/2",
            "H_2": "1/3 1/3 -1/2",
            "M": "1/2 0 0",
            "L": "1/2 0 1/2",
        },
    ),
    "hP2": (
        "16 -8 0 -8 16 0 0 0 36",
        {
            "GAMMA": "0 0 0",
            "A": "0 0 1/2",
            "K": "1/3 1/3 0",
            "H": "1/3 1/3 1/2",
            "H_2": "1/3 1/3 -1/2",
            "M": "1/2 0 0",
            "L": "1/2 0 1/2",
        },
    ),
    "hR1": (
        "64/3 40/3 40/3 40/3 64/3 40/3 40/3 40/3 64/3",
        {
            "GAMMA": "0 0 0",
            "T": "1/2 1/2 1/2",
            "L": "1/2 0 0",
            "L_2": "0 -1/2 0",
            "L_4": "0 0 -1/2",
            "F": "1/2 0 1/2",
            "F_2": "1/2 1/2 0",
            "S_0": "13/36 -13/36 0",
            "S_2": "23/36 0 13/36",
            "S_4": "13/36 0 -13/36",
            "S_6": "23/36 13/36 0",
            "H_0": "1/2 -2/9 2/9",
            "H_2": "7/9 2/9 1/2",
            "H_4": "7/9 1/2 2/9",
            "H_6": "1/2 2/9 -2/9",
            "M_0": "13/36 -2/9 13/36",
            "M_2": "23/36 2/9 23/36",
            "M_4": "7/9 13/36 13/36",
            "M_6": "23/36 23/36 2/9",
            "M_8": "13/36 13/36 -2/9",
        },
    ),
    "hR2": (
        "19/3 -5/3 -5/3 -5/3 19/3 -5/3 -5/3 -5/3 19/3",
        {
            "GAMMA": "0 0 0",
            "T": "1/2 -1/2 1/2",
            "P_0": "7/24 -17/24 7/24",
            "P_2": "7/24 7/24 7/24",
            "R_0": "17/24 -7/24 -7/24",
            "M": "19/48 -29/48 19/48",
            "M_2": "29/48 -19/48 -19/48",
            "L": "1/2 0 0",
            "F": "1/2 -1/2 0",
        },
    ),
    "mC1": (
        "20 -12 -8 -12 20 8 -8 8 53",
        {
            "GAMMA": "0 0 0",
            "Y_2": "-1/2 1/2 0",
            "Y_4": "1/2 -1/2 0",
            "A": "0 0 1/2",
            "M_2": "-1/2 1/2 1/2",
            "V": "1/2 0 0",
            "V_2": "0 1/2 0",
            "L_2": "0 1/2 1/2",
            "C": "249/784 249/784 0",
            "C_2": "-249/784 535/784 0",
            "C_4": "535/784 -249/784 0",
            "D": "-233/784 551/784 1/2",
            "D_2": "233/784 233/784 1/2",
            "E": "-53/98 53/98 53/196",
            "E_2": "-45/98 45/98 143/196",
            "E_4": "45/98 -45/98 53/196",
        },
    ),
    "mC2": (
        "20 12 -2 12 20 2 -2 2 37",
        {
            "GAMMA": "0 0 0",
            "Y": "1/2 1/2 0",
            "A": "0 0 1/2",
            "M": "1/2 1/2 1/2",
            "V_2": "0 1/2 0",
            "L_2": "0 1/2 1/2",
            "F": "-1/3 1/3 5/12",
            "F_2": "1/3 2/3 7/12",
            "F_4": "2/3 1/3 5/12",
            "H": "-7/24 7/24 31/48",
            "H_2": "7/24 17/24 17/48",
            "H_4": "7/24 -7/24 17/48",
            "G": "-5/16 5/16 1/32",
            "G_2": "5/16 11/16 -1/32",
            "G_4": "5/16 -5/16 -1/32",
            "G_6": "11/16 5/16 1/32",
        },
    ),
    "mC3": (
        "18 0 -6 0 18 6 -6 6 29",
        {
            "GAMMA": "0 0 0",
            "Y": "1/2 1/2 0",
            "A": "0 0 1/2",
            "M_2": "-1/2 1/2 1/2",
            "V": "1/2 0 0",
            "V_2": "0 1/2 0",
            "L_2": "0 1/2 1/2",
            "I": "-21/50 29/50 1/2",
            "I_2": "21/50 21/50 1/2",
            "K": "-79/150 79/150 133/450",
            "K_2": "-71/150 71/150 317/450",
            "K_4": "71/150 79/150 133/450",
            "H": "-21/50 21/50 39/50",
            "H_2": "21/50 29/50 11/50",
            "H_4": "21/50 -21/50 11/50",
            "N": "-71/150 71/150 17/450",
            "N_2": "71/150 79/150 -17/450",
            "N_4": "71/150 -71/150 -17/450",
            "N_6": "79/150 71/150 17/450",
        },
    ),
    "mP1": (
        "25 0 -10 0 36 0 -10 0 53",
        {
            "GAMMA": "0 0 0",
            "Z": "0 1/2 0",
            "B": "0 0 1/2",
            "B_2": "0 0 -1/2",
            "Y": "1/2 0 0",
            "Y_2": "-1/2 0 0",
            "C": "1/2 1/2 0",
            "C_2": "-1/2 1/2 0",
            "D": "0 1/2 1/2",
            "D_2": "0 1/2 -1/2",
            "A": "-1/2 0 1/2",
            "E": "-1/2 1/2 1/2",
            "H": "-43/98 0 331/490",
            "H_2": "-55/98 0 159/490",
            "H_4": "-43/98 0 -159/490",
            "M": "-43/98 1/2 331/490",
            "M_2": "-55/98 1/2 159/490",
            "M_4": "-43/98 1/2 -159/490",
        },
    ),
    "oA1": (
        "41/4 -9/4 0 -9/4 41/4 0 0 0 9",
        {
            "GAMMA": "0 0 0",
            "Y": "-1/2 1/2 0",
            "T": "-1/2 1/2 1/2",
            "Z": "0 0 1/2",
            "S": "0 1/2 0",
            "R": "0 1/2 1/2",
            "SIGMA_0": "41/100 41/100 0",
            "C_0": "-41/100 59/100 0",
            "A_0": "41/100 41/100 1/2",
            "E_0": "-41/100 59/100 1/2",
        },
    ),
    "oA2": (
        "41/4 9/4 0 9/4 41/4 0 0 0 9",
        {
            "GAMMA": "0 0 0",
            "Y": "1/2 1/2 0",
            "T": "1/2 1/2 1/2",
            "T_2": "1/2 1/2 -1/2",
            "Z": "0 0 1/2",
            "Z_2": "0 0 -1/2",
            "S": "0 1/2 0",
            "R": "0 1/2 1/2",
            "R_2": "0 1/2 -1/2",
            "DELTA_0": "-41/100 41/100 0",
            "F_0": "41/100 59/100 0",
            "B_0": "-41/100 41/100 1/2",
            "B_2": "-41/100 41/100 -1/2",
            "G_0": "41/100 59/100 1/2",
            "G_2": "41/100 59/100 -1/2",
        },
    ),
    "oC1": (
        "13 -5 0 -5 13 0 0 0 25",
        {
            "GAMMA": "0 0 0",
            "Y": "-1/2 1/2 0",
            "T": "-1/2 1/2 1/2",
            "Z": "0 0 1/2",
            "S": "0 1/2 0",
            "R": "0 1/2 1/2",
            "SIGMA_0": "13/36 13/36 0",
            "C_0": "-13/36 23/36 0",
            "A_0": "13/36 13/36 1/2",
            "E_0": "-13/36 23/36 1/2",
        },
    ),
    "oC2": (
        "13 5 0 5 13 0 0 0 25",
        {
            "GAMMA": "0 0 0",
            "Y": "1/2 1/2 0",
            "T": "1/2 1/2 1/2",
            "T_2": "1/2 1/2 -1/2",
            "Z": "0 0 1/2",
            "Z_2": "0 0 -1/2",
            "S": "0 1/2 0",
            "R": "0 1/2 1/2",
            "R_2": "0 1/2 -1/2",
            "DELTA_0": "-13/36 13/36 0",
            "F_0": "13/36 23/36 0",
            "B_0": "-13/36 13/36 1/2",
            "B_2": "-13/36 13/36 -1/2",
            "G_0": "13/36 23/36 1/2",
            "G_2": "13/36 23/36 -1/2",
        },
    ),
    "oF1": (
        "61/4 9 25/4 9 45/4 9/4 25/4 9/4 17/2",
        {
            "GAMMA": "0 0 0",
            "T": "1 1/2 1/2",
            "Z": "1/2 1/2 0",
            "Y": "1/2 0 1/2",
            "SIGMA_0": "0 161/400 161/400",
            "U_0": "1 239/400 239/400",
            "A_0": "1/2 311/400 111/400",
            "C_0": "1/2 89/400 289/400",
            "L": "1/2 1/2 1/2",
        },
    ),
    "oF2": (
        "45/4 9/4 9 9/4 17/2 25/4 9 25/4 61/4",
        {
            "GAMMA": "0 0 0",
            "T": "0 1/2 1/2",
            "Z": "1/2 1/2 1",
            "Y": "1/2 0 1/2",
            "LAMBDA_0": "161/400 161/400 0",
            "Q_0": "239/400 239/400 1",
            "G_0": "89/400 289/400 1/2",
            "H_0": "311/400 111/400 1/2",
            "L": "1/2 1/2 1/2",
        },
    ),
    "oF3": (
        "61/4 9 25/4 9 13 4 25/4 4 41/4",
        {
            "GAMMA": "0 0 0",
            "T": "0 1/2 1/2",
            "Z": "1/2 1/2 0",
            "Y": "1/2 0 1/2",
            "A_0": "1/2 719/900 269/900",
            "C_0": "1/2 181/900 631/900",
            "B_0": "557/576 1/2 269/576",
            "D_0": "19/576 1/2 307/576",
            "G_0": "19/400 219/400 1/2",
            "H_0": "381/400 181/400 1/2",
            "L": "1/2 1/2 1/2",
        },
    ),
    "oI1": (
        "25/2 0 -9/2 0 25/2 -8 -9/2 -8 25/2",
        {
            "GAMMA": "0 0 0",
            "X": "1/2 1/2 -1/2",
            "S": "1/2 0 0",
            "R": "0 1/2 0",
            "T": "0 0 1/2",
            "W": "1/4 1/4 1/4",
            "SIGMA_0": "-17/50 17/50 17/50",
            "F_2": "17/50 33/50 -17/50",
            "Y_0": "41/100 -41/100 41/100",
            "U_0": "59/100 41/100 -41/100",
            "L_0": "-1/4 1/4 43/100",
            "M_0": "1/4 -1/4 57/100",
            "J_0": "43/100 57/100 -1/4",
        },
    ),
    "oI2": (
        "55/2 -19/2 -15 -19/2 55/2 -3 -15 -3 55/2",
        {
            "GAMMA": "0 0 0",
            "X": "-1/2 1/2 1/2",
            "S": "1/2 0 0",
            "R": "0 1/2 0",
            "T": "0 0 1/2",
            "W": "1/4 1/4 1/4",
            "Y_0": "37/98 -37/98 37/98",
            "U_2": "-37/98 37/98 61/98",
            "LAMBDA_0": "85/196 85/196 -85/196",
            "G_2": "-85/196 111/196 85/196",
            "K": "87/196 -61/196 61/196",
            "K_2": "109/196 61/196 -61/196",
            "K_4": "-61/196 87/196 109/196",
        },
    ),
    "oI3": (
        "55/2 -19/2 -3 -19/2 55/2 -15 -3 -15 55/2",
        {
            "GAMMA": "0 0 0",
            "X": "1/2 -1/2 1/2",
            "S": "1/2 0 0",
            "R": "0 1/2 0",
            "T": "0 0 1/2",
            "W": "1/4 1/4 1/4",
            "SIGMA_0": "-37/98 37/98 37/98",
            "F_0": "37/98 -37/98 61/98",
            "LAMBDA_0": "85/196 85/196 -85/196",
            "G_0": "111/196 -85/196 85/196",
            "V_0": "61/196 109/196 -61/196",
            "H_0": "-61/196 87/196 61/196",
            "H_2": "87/196 -61/196 109/196",
        },
    ),
    "oP1": (
        "9 0 0 0 16 0 0 0 25",
        {
            "GAMMA": "0 0 0",
            "X": "1/2 0 0",
            "Z": "0 0 1/2",
            "U": "1/2 0 1/2",
            "Y": "0 1/2 0",
            "S": "1/2 1/2 0",
            "T": "0 1/2 1/2",
            "R": "1/2 1/2 1/2",
        },
    ),
    "tI1": (
        "33/2 -17/2 -4 -17/2 33/2 -4 -4 -4 33/2",
        {
            "GAMMA": "0 0 0",
            "M": "-1/2 1/2 1/2",
            "X": "0 0 1/2",
            "P": "1/4 1/4 1/4",
            "Z": "41/100 41/100 -41/100",
            "Z_0": "-41/100 59/100 41/100",
            "N": "0 1/2 0",
        },
    ),
    "tI2": (
        "81/4 17/4 -49/4 17/4 81/4 -49/4 -49/4 -49/4 81/4",
        {
            "GAMMA": "0 0 0",
            "M": "1/2 1/2 -1/2",
            "X": "0 0 1/2",
            "P": "1/4 1/4 1/4",
            "N": "0 1/2 0",
            "S_0": "-65/196 65/196 65/196",
            "S": "65/196 131/196 -65/196",
            "R": "-8/49 8/49 1/2",
            "G": "1/2 1/2 -8/49",
        },
    ),
    "tP1": (
        "16 0 0 0 16 0 0 0 25",
        {"GAMMA": "0 0 0", "Z": "0 0 1/2", "M": "1/2 1/2 0", "A": "1/2 1/2 1/2", "R": "0 1/2 1/2", "X": "0 1/2 0"},
    ),
}


def _fractions(text: str) -> list[fractions.Fraction]:
    return [F(value) for value in text.split()]


# Rows of exact surd bases, each entry (rational coefficient, radicand).
_HR_TIE_BASIS = (((1, 2), (0, 1), (0, 1)), ((F(-1, 2), 2), (F(1, 2), 6), (0, 1)), ((0, 1), (0, 1), (1, 3)))
_HEXAGONAL_BASIS = (((4, 1), (0, 1), (0, 1)), ((-2, 1), (2, 3), (0, 1)), ((0, 1), (0, 1), (12, 1)))


def _surd_basis(rows: tuple[tuple[tuple[Any, int], ...], ...]) -> SurdVector:
    return SurdVector._from_scalar_grid(
        [[SurdVector.sqrt_of(radicand) * coefficient for coefficient, radicand in row] for row in rows], (3, 3)
    )


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _asu(it_number: int, cell: Any, sites: tuple[tuple[str, str], ...] = (("a", "C"),)) -> ASUStructure:
    group = Spacegroup.standard(it_number)
    wyckoff_sites = [
        WyckoffSite(letter, FracVector(list(_FREE_VALUES[: group.wyckoff_position(letter).free_count])), species)
        for letter, species in sites
    ]
    return ASUStructure(Cell(cell), it_number, wyckoff_sites, _species(*sorted({name for _, name in sites})))


def _representative(symbol: str) -> ASUStructure:
    _, it_number, cell, sites = next(entry for entry in REPRESENTATIVES if entry[0] == symbol)
    return _asu(it_number, cell, sites)


def _metric(result: BandPathResult) -> list[list[fractions.Fraction]]:
    return result.structure.cell.metric().coefficient(1).to_fractions()


def _sites(structure: UnitcellStructure, matrix: FracVector | None = None) -> set[tuple[str, tuple[F, ...]]]:
    coordinates = structure.sites.reduced_coords if matrix is None else structure.sites.reduced_coords * matrix
    return {
        (species, tuple(coordinate.normalize().to_fractions()))
        for species, coordinate in zip(structure.species_at_sites, coordinates, strict=True)
    }


def test_representatives_cover_every_extended_symbol() -> None:
    assert sorted(symbol for symbol, *_ in REPRESENTATIVES) == sorted(BAND_PATH_DATA)


@pytest.mark.parametrize(("symbol", "it_number", "cell", "sites"), REPRESENTATIVES, ids=[r[0] for r in REPRESENTATIVES])
def test_representative_reaches_its_extended_symbol(
    symbol: str, it_number: int, cell: Any, sites: tuple[tuple[str, str], ...]
) -> None:
    structure = _asu(it_number, cell, sites)
    result = band_path(structure)
    primitive = primitive_cell(structure)
    entry = BAND_PATH_DATA[symbol]

    assert result.bravais_lattice_extended == symbol
    golden_metric, golden_points = GOLDEN[symbol]
    assert [value for row in _metric(result) for value in row] == _fractions(golden_metric)
    assert {label: point.to_fractions() for label, point in result.points.items()} == {
        label: _fractions(text) for label, text in golden_points.items()
    }
    assert result.bravais_lattice == symbol[:2]
    assert result.spacegroup == Spacegroup.standard(it_number)
    assert result.path == entry.path
    assert tuple(result.points) == tuple(label for label, _ in entry.points)
    for point in result.points.values():
        assert isinstance(point, FracVector)
        assert len(point.to_fractions()) == 3
        assert all(isinstance(value, F) for value in point.to_fractions())
    centring = len(result.spacegroup.centering_translations)
    assert result.structure.cell.volume * centring == primitive.conventional.structure.cell.volume
    assert result.transform.denom == 1
    assert result.transform.det().to_fraction() == 1
    assert result.structure.cell.basis.det().sign() > 0
    with pytest.raises(TypeError):
        result.points["GAMMA"] = FracVector([1, 0, 0])  # type: ignore[index]


@pytest.mark.parametrize("symbol", ["cF2", "tI2", "hR1", "mC1", "oA1", "aP2", "oF2"])
def test_transform_round_trips_to_the_primitive_cell_sites(symbol: str) -> None:
    structure = _representative(symbol)
    result = band_path(structure)
    primitive = primitive_cell(structure).structure

    assert result.structure.cell.basis == SurdVector(result.transform) * primitive.cell.basis
    assert _sites(result.structure, result.transform) == _sites(primitive)
    assert result.structure.species_at_sites == primitive.species_at_sites


def test_hand_derived_exact_point_values() -> None:
    cf2 = band_path(_representative("cF2"))
    assert cf2.points["W"] == FracVector([F(1, 2), F(1, 4), F(3, 4)])

    ti2 = band_path(_representative("tI2"))  # a = 4, c = 7
    h = (1 + F(16, 49)) / 4
    assert ti2.points["S_0"] == FracVector([-h, h, h])
    assert ti2.points["R"] == FracVector([-F(8, 49), F(8, 49), F(1, 2)])

    # mC1: a = 8, b = 4, c = (-2, 0, 7) is already the Parthé–Gelato cell: G11 = 64, G33 = 53,
    # G13 = -16, sin^2(beta) = 49/53, so Z = (2 + G13/G33) / (4 sin^2 beta) = 45/98 and
    # H = 1/2 - 2 Z G13 / G11 = 1/2 + 45/196.
    mc1 = band_path(_representative("mC1"))
    z = F(45, 98)
    assert mc1.points["E_2"] == FracVector([-z, z, F(1, 2) + 2 * z * 16 / 64])


def test_extended_symbol_ties_follow_seekpath() -> None:
    # tI1 iff c^2 <= a^2 (a declared tetragonal cell may have c = a exactly).
    assert band_path(_asu(139, [5, 5, 5, 90, 90, 90])).bravais_lattice_extended == "tI1"
    # oC1 iff a^2 <= b^2; Cmcm (SG 63) admits no axis re-choice.
    assert band_path(_asu(63, [5, 5, 6, 90, 90, 90])).bravais_lattice_extended == "oC1"
    # oA1 iff b^2 <= c^2; Amm2 (SG 38) fixes the axes.
    assert band_path(_asu(38, [3, 5, 5, 90, 90, 90])).bravais_lattice_extended == "oA1"
    # oI: largest squared length with priority b (oI3) > a (oI2) > c (oI1); Ima2 (SG 46) fixes axes.
    assert band_path(_asu(46, [7, 7, 5, 90, 90, 90])).bravais_lattice_extended == "oI3"
    assert band_path(_asu(46, [7, 5, 7, 90, 90, 90])).bravais_lattice_extended == "oI2"
    # oF1 needs 1/a^2 > 1/b^2 + 1/c^2 strictly: at equality (12/5, 3, 4) it falls to oF3.
    assert band_path(_asu(69, [F(12, 5), 3, 4, 90, 90, 90])).bravais_lattice_extended == "oF3"
    # oF2 needs 1/c^2 > 1/a^2 + 1/b^2 strictly; Fmm2 (SG 42) fixes c.
    assert band_path(_asu(42, [3, 4, F(12, 5), 90, 90, 90])).bravais_lattice_extended == "oF3"
    # hR1 iff 3a^2 <= 2c^2, here 3 * 2 == 2 * 3 with an exact surd basis.
    assert band_path(_asu(166, _surd_basis(_HR_TIE_BASIS))).bravais_lattice_extended == "hR1"


def _monoclinic_conventional_metric(result: BandPathResult) -> list[list[fractions.Fraction]]:
    row_primitive = _HPKOT_COLUMN_PRIMITIVE["mC"].T().inv()
    return (row_primitive * result.structure.cell.metric().coefficient(1) * row_primitive.T()).to_fractions()


def test_monoclinic_symbol_ties_follow_seekpath() -> None:
    # mC1 needs b^2 < a^2 - G13^2/G33 strictly: 576/25 == 36 - 18^2/25 is not mC1.
    tie = band_path(_asu(12, [[6, 0, 0], [0, F(24, 5), 0], [-3, 0, 4]]))
    metric = _monoclinic_conventional_metric(tie)
    assert metric == [[36, 0, -18], [0, F(576, 25), 0], [-18, 0, 25]]
    assert tie.bravais_lattice_extended == "mC3"
    # mC2 iff -G13/G33 + (G11 - G13^2/G33)/G22 <= 1: here 21/25 + (49 - 441/25)/196 == 1.
    tie = band_path(_asu(12, [[7, 0, 0], [0, 14, 0], [-3, 0, 4]]))
    assert _monoclinic_conventional_metric(tie) == [[49, 0, -21], [0, 196, 0], [-21, 0, 25]]
    assert tie.bravais_lattice_extended == "mC2"


def _signed_permutation(order: tuple[int, ...], signs: tuple[int, ...]) -> FracVector:
    return FracVector([[signs[i] if j == order[i] else 0 for j in range(3)] for i in range(3)])


def test_setting_preservation_convention() -> None:
    pnma = Spacegroup.standard(62)
    assert _preserves_setting(pnma, _signed_permutation((0, 1, 2), (1, 1, 1)))
    assert _preserves_setting(pnma, _signed_permutation((0, 1, 2), (-1, -1, 1)))
    assert _preserves_setting(pnma, _signed_permutation((0, 1, 2), (1, -1, -1)))
    # Swapping a and b turns Pnma into Pmnb, a different setting.
    assert not _preserves_setting(pnma, _signed_permutation((1, 0, 2), (1, 1, -1)))
    # Pbca allows the cyclic permutations only.
    pbca = Spacegroup.standard(61)
    assert _preserves_setting(pbca, _signed_permutation((1, 2, 0), (1, 1, 1)))
    assert not _preserves_setting(pbca, _signed_permutation((1, 0, 2), (1, 1, -1)))
    # Monoclinic re-choices are not orthogonal, so these also pin A = Q^-T against A = Q^T.
    a_plus_c_first = FracVector([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    a_plus_c_last = FracVector([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    assert _preserves_setting(Spacegroup.standard(14), a_plus_c_first)
    assert not _preserves_setting(Spacegroup.standard(14), a_plus_c_last)  # P2_1/n
    assert not _preserves_setting(Spacegroup.standard(15), a_plus_c_first)  # I-centred
    assert _preserves_setting(Spacegroup.standard(15), a_plus_c_last)
    pmmm = Spacegroup.standard(47)
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = _signed_permutation(order, signs)
            if matrix.det().to_fraction() == 1:
                assert _preserves_setting(pmmm, matrix)


def test_exact_square_root_sum_comparison() -> None:
    assert _sqrt_sum_sign(F(2), F(8), F(18), F(0)) == 0  # sqrt2 + 2 sqrt2 = 3 sqrt2
    assert _sqrt_sum_sign(F(1), F(9), F(4), F(4)) == 0
    assert _sqrt_sum_sign(F(2), F(3), F(1), F(6)) == -1
    assert _sqrt_sum_sign(F(1), F(6), F(2), F(3)) == 1
    assert _sqrt_sum_sign(F(25), F(20), F(25000060000100, 10**12), F(20)) == -1


def test_point_grammar_is_exact_and_closed() -> None:
    parameters = {"X": F(1, 3)}
    assert _coordinate("3/8", parameters) == F(3, 8)
    assert _coordinate("-1/2", parameters) == F(-1, 2)
    assert _coordinate("1-X", parameters) == F(2, 3)
    assert _coordinate("-1+X", parameters) == F(-2, 3)
    assert _coordinate("1/2+X", parameters) == F(5, 6)
    assert _coordinate("-X", parameters) == F(-1, 3)
    for text in ("2*X", "Y", "1/3-X", " 1", "1e3", "0.5", "X+1"):
        with pytest.raises(ValueError, match="unsupported band-path coordinate"):
            _coordinate(text, parameters)


def test_orthorhombic_axis_order_is_rechosen() -> None:
    sorted_axes = band_path(_asu(47, [3, 4, 5, 90, 90, 90]))
    permuted = band_path(_asu(47, [5, 3, 4, 90, 90, 90]))

    assert permuted.bravais_lattice_extended == sorted_axes.bravais_lattice_extended == "oP1"
    assert permuted.points == sorted_axes.points
    assert [_metric(permuted)[i][i] for i in range(3)] == [9, 16, 25]
    assert _metric(permuted) == _metric(sorted_axes)


def test_monoclinic_non_reduced_statement_gives_the_reduced_result() -> None:
    # C2/m: c + a keeps the C-centred cell choice 1 but is not the Parthé–Gelato cell.
    for it_number, reduced, stated in (
        (12, [[8, 0, 0], [0, 4, 0], [-2, 0, 7]], [[8, 0, 0], [0, 4, 0], [6, 0, 7]]),
        (10, [[5, 0, 0], [0, 6, 0], [-2, 0, 7]], [[5, 0, 0], [0, 6, 0], [-17, 0, 7]]),
    ):
        expected = band_path(_asu(it_number, reduced))
        result = band_path(_asu(it_number, stated))
        assert result.bravais_lattice_extended == expected.bravais_lattice_extended
        assert result.points == expected.points
        assert _metric(result) == _metric(expected)


def test_triclinic_sheared_statement_gives_the_same_result() -> None:
    basis = FracVector([[4, 0, 0], [1, 5, 0], [1, 2, 6]])
    shear = FracVector([[1, 0, 0], [1, 1, 0], [2, -1, 1]])
    expected = band_path(_asu(2, basis.to_fractions()))
    result = band_path(_asu(2, (shear * basis).to_fractions()))

    assert result.bravais_lattice_extended == expected.bravais_lattice_extended == "aP2"
    assert result.points == expected.points
    assert _metric(result) == _metric(expected)


def test_nonstandard_setting_asu_gives_the_standard_setting_result() -> None:
    transform = Spacegroup.from_setting("62:cab").transform_from_standard  # Pbnm
    own_basis = Cell([[5, 0, 0], [0, 6, 0], [0, 0, 7]]).basis
    sites = [WyckoffSite("a", FracVector(()), "Fe"), WyckoffSite("b", FracVector(()), "O")]
    nonstandard = ASUStructure(own_basis, 62, sites, _species("Fe", "O"), transform=transform)
    standard = ASUStructure(transform.basis_to_standard(own_basis), 62, sites, _species("Fe", "O"))

    result = band_path(nonstandard)
    expected = band_path(standard)

    assert result.spacegroup == Spacegroup.standard(62)
    assert result.bravais_lattice_extended == expected.bravais_lattice_extended == "oP1"
    assert result.points == expected.points
    assert _metric(result) == _metric(expected)
    assert _sites(result.structure) == _sites(expected.structure)


def test_rhombohedral_axes_asu_gives_the_hexagonal_result() -> None:
    transform = Spacegroup.from_setting("166:R").transform_from_standard
    hexagonal = _surd_basis(_HEXAGONAL_BASIS)
    sites = [WyckoffSite("a", FracVector(()), "Bi")]
    rhombohedral = ASUStructure(transform.basis_to_setting(hexagonal), 166, sites, _species("Bi"), transform=transform)

    result = band_path(rhombohedral)
    expected = band_path(ASUStructure(hexagonal, 166, sites, _species("Bi")))

    assert result.bravais_lattice_extended == expected.bravais_lattice_extended == "hR1"
    assert result.points == expected.points
    assert _metric(result) == _metric(expected)
    assert _sites(result.structure) == _sites(expected.structure)


@pytest.mark.parametrize(("setting", "it_number"), [("14:b2", 14), ("14:c1", 14), ("15:-b1", 15)])
def test_nonstandard_monoclinic_setting_gives_the_standard_result(setting: str, it_number: int) -> None:
    transform = Spacegroup.from_setting(setting).transform_from_standard
    standard_basis = Cell([[5, 0, 0], [0, 6, 0], [-2, 0, 7]]).basis
    sites = [WyckoffSite("a", FracVector(()), "C")]
    own_basis = transform.basis_to_setting(standard_basis)

    result = band_path(ASUStructure(own_basis, it_number, sites, _species("C"), transform=transform))
    expected = band_path(ASUStructure(standard_basis, it_number, sites, _species("C")))

    assert result.spacegroup == Spacegroup.standard(it_number)
    assert result.bravais_lattice_extended == expected.bravais_lattice_extended
    assert result.points == expected.points
    assert _metric(result) == _metric(expected)
    assert same_crystal(result.structure, expected.structure)


@pytest.mark.parametrize(
    ("it_number", "cell", "letter"),
    [
        (14, [[5, 0, 0], [0, 6, 0], [-2, 0, 7]], "e"),
        (15, [[8, 0, 0], [0, 4, 0], [-2, 0, 7]], "f"),
        (62, [5, 6, 7, 90, 90, 90], "d"),
    ],
)
def test_result_structure_recognizes_to_the_same_group(it_number: int, cell: Any, letter: str) -> None:
    result = band_path(_asu(it_number, cell, ((letter, "C"),)))

    assert recognize_asu(result.structure).spacegroup.it_number == it_number


def test_orthorhombic_tie_keeps_a_shortest_for_pbca() -> None:
    # Codex counterexample (a): Pbca allows only cyclic permutations; (4, 6, 4) -> (4, 4, 6).
    result = band_path(_asu(61, [4, 6, 4, 90, 90, 90]))
    metric = _metric(result)
    assert [metric[i][i] for i in range(3)] == [16, 16, 36]
    assert result.bravais_lattice_extended == "oP1"


def test_monoclinic_exact_minimal_cell_beats_a_near_tie() -> None:
    # Codex counterexample (b): (a + c, -b, -c) has a strictly shorter |a'| + |c'| than the stated
    # cell (by about 4e-6), with a'.c' <= 0; httk takes the exact minimum.
    result = band_path(_asu(7, [[F(500001, 100000), 0, 0], [0, 7, 0], [-2, 0, 4]]))

    assert result.bravais_lattice_extended == "mP1"
    assert result.transform == FracVector([[1, 0, 1], [0, -1, 0], [0, 0, -1]])
    assert _metric(result) == [
        [F(250000600001, 10**10), 0, F(-499999, 50000)],
        [0, 49, 0],
        [F(-499999, 50000), 0, 20],
    ]


def test_triclinic_reduced_reciprocal_cell_conditions() -> None:
    # Codex counterexample (c): a reciprocal basis just off a Niggli boundary.
    reciprocal = FracVector([[2, 0, 0], [F(10, 13), F(24, 13) + F(1, 10**9), 0], [F(3, 10), F(2, 5), 3]])
    result = band_path(_asu(1, reciprocal.inv().T().to_fractions()))

    assert result.bravais_lattice_extended == "aP3"
    star = result.structure.cell.metric().coefficient(1).inv()
    values = star.to_fractions()
    cyclic = [FracVector(rows) for rows in ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])]
    cyclic.append(cyclic[1] * cyclic[1])
    assert any(_is_niggli_parameters(_parameters(p * star * p.T())) for p in cyclic)
    off_diagonal = (values[1][2], values[0][2], values[0][1])
    assert all(value > 0 for value in off_diagonal)
    assert abs(values[0][1]) == min(abs(value) for value in off_diagonal)


def test_left_handed_conventional_basis_gives_a_right_handed_result() -> None:
    result = band_path(_asu(1, [[-3, 0, 0], [0, 4, 0], [0, 0, 5]]))

    assert result.structure.cell.basis.det().sign() > 0
    assert result.transform.det().to_fraction() == -1


def test_raw_structure_is_recognized() -> None:
    quarter, half = F(1, 4), F(1, 2)
    coordinates = [
        [0, 0, 0],
        [0, half, half],
        [half, 0, half],
        [half, half, 0],
        [quarter, quarter, quarter],
        [quarter, 3 * quarter, 3 * quarter],
        [3 * quarter, quarter, 3 * quarter],
        [3 * quarter, 3 * quarter, quarter],
    ]
    length = F(543, 100)
    silicon = UnitcellStructure(
        Cell([[length, 0, 0], [0, length, 0], [0, 0, length]]), coordinates, _species("Si"), ["Si"] * 8
    )

    result = band_path(silicon)

    assert result.spacegroup.it_number == 227
    assert result.bravais_lattice_extended == "cF2"
    assert len(result.structure.sites) == 2


def test_unsupported_inputs_are_refused() -> None:
    magnetic = ASUStructure(
        Cell(CUBIC),
        221,
        [WyckoffSite("a", FracVector(()), "Fe", moment=CartesianSiteMoments([[0, 0, 2]]))],
        _species("Fe"),
    )
    with pytest.raises(ValueError, match="magnetic"):
        band_path(magnetic)
    with pytest.raises(ValueError, match="rational-valued exact Gram matrix"):
        band_path(_asu(10, [3, 4, 5, 90, 135, 90]))
    slab = UnitcellStructure(
        Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]], periodicity=(True, True, False)), [[0, 0, 0]], _species("C"), ["C"]
    )
    with pytest.raises(ValueError, match="3D-periodic"):
        band_path(slab)


def test_time_reversal_augmentation_only_without_inversion() -> None:
    zincblende = _representative("cF1")
    plain = band_path(zincblende)
    augmented = band_path(zincblende, time_reversal=False)

    assert band_path(zincblende, time_reversal=True).points == plain.points
    labels = [label for label in plain.points if label != "GAMMA"]
    assert tuple(augmented.points) == (*plain.points, *(f"{label}'" for label in labels))
    for label in labels:
        assert augmented.points[f"{label}'"] == -plain.points[label]

    def primed(label: str) -> str:
        return label if label == "GAMMA" else f"{label}'"

    assert augmented.path == plain.path + tuple((primed(start), primed(end)) for start, end in plain.path)

    rocksalt = _representative("cF2")
    assert band_path(rocksalt, time_reversal=False).points == band_path(rocksalt).points
    assert band_path(rocksalt, time_reversal=False).path == band_path(rocksalt).path
