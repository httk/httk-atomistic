"""Build exact general, orthogonal, and cubic supercells

The v1 Step 2 tutorial demonstrated a particularly useful feature: starting
from a skewed primitive cell, httk could build an exact integer supercell and
search for rectangular or nearly cubic choices automatically.

The v2 interface separates those two jobs. `Structure.supercell()` applies a
transformation you already know. `orthogonal_supercell()` and
`cubic_supercell()` search at an explicitly requested **multiplier** — the
number of original cells, and therefore the factor by which the site count
grows. This replaces v1's `tolerance` argument, which was a search knob rather
than a geometric tolerance.

Every returned `SupercellResult` includes the materialized structure, the exact
integer transformation, and exact dimensionless shape scores. A score of zero
is a proof that the corresponding target was reached exactly.
"""

import fractions

from httk.atomistic import Structure

F = fractions.Fraction

# The skewed cell from the v1 Step 2 POSCAR2 tutorial, represented exactly.
structure = Structure(
    cell=[
        [F(1622, 400), 0, 0],
        [0, F(1622, 400), 0],
        [F(811, 400), F(811, 400), F(2524, 400)],
    ],
    sites=[
        [0, F(1, 2), 0],
        [F(1, 2), 0, 0],
        [F(3, 4), F(1, 4), F(1, 2)],
        [F(1, 4), F(3, 4), F(1, 2)],
        [0, 0, 0],
        [F(351, 1000), F(351, 1000), F(298, 1000)],
        [F(649, 1000), F(649, 1000), F(702, 1000)],
    ],
    species=[
        {"name": "O", "chemical_symbols": ["O"], "concentration": [1]},
        {"name": "Pd", "chemical_symbols": ["Pd"], "concentration": [1]},
        {"name": "La", "chemical_symbols": ["La"], "concentration": [1]},
    ],
    species_at_sites=["O", "O", "O", "O", "Pd", "La", "La"],
)

# A known transformation: double the first lattice direction.
general = structure.supercell([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
print("general transform:", general.transformation)
print("general sites:", len(general.structure.sites))

# Two primitive cells contain an exactly orthogonal basis.
orthogonal = structure.orthogonal_supercell(multiplier=2)
print("orthogonal transform:", orthogonal.transformation)
print("orthogonality score:", orthogonal.orthogonality_score)

# Eighteen primitive cells give a balanced, almost-cubic rectangle.
cubic = structure.cubic_supercell(multiplier=18)
print("cubic transform:", cubic.transformation)
print("cubicity score:", cubic.cubicity_score)
