"""
Minimal periodic table for httk-atomistic.

Provides the IUPAC element symbols in atomic-number order and helpers to convert
between a chemical symbol and its atomic number. The pseudo-symbols ``"X"``
(unknown element) and ``"vacancy"`` are deliberately not elements here; they are
handled at the ``Species`` level.
"""

SYMBOLS: tuple[str, ...] = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
"""The 118 IUPAC element symbols in atomic-number order (``SYMBOLS[0]`` is hydrogen)."""

_NUMBER_OF: dict[str, int] = {symbol: z for z, symbol in enumerate(SYMBOLS, start=1)}


def atomic_number(symbol: str) -> int:
    """
    Return the atomic number (1-118) of an element symbol.

    Raises ValueError for anything that is not one of the 118 element symbols
    (in particular for the ``"X"`` and ``"vacancy"`` pseudo-symbols).
    """
    try:
        return _NUMBER_OF[symbol]
    except KeyError:
        raise ValueError(f"Unknown element symbol: {symbol!r}") from None


def symbol_of(z: int) -> str:
    """
    Return the element symbol for the atomic number z (1-118).

    Raises ValueError for atomic numbers outside the 1-118 range.
    """
    if not 1 <= z <= len(SYMBOLS):
        raise ValueError(f"Unknown atomic number: {z!r}")
    return SYMBOLS[z - 1]
