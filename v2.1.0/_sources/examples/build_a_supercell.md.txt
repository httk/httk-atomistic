# Build exact general, orthogonal, and cubic supercells

The v1 Step 2 tutorial demonstrated a particularly useful feature: starting
from a skewed primitive cell, httk could build an exact integer supercell and
search for rectangular or nearly cubic choices automatically.

The v2 interface separates those two jobs. `UnitcellStructure.supercell()` applies a
transformation you already know. `orthogonal_supercell()` and
`cubic_supercell()` search at an explicitly requested **multiplier** — the
number of original cells, and therefore the factor by which the site count
grows. This replaces v1's `tolerance` argument, which was a search knob rather
than a geometric tolerance.

Every returned `SupercellResult` includes the materialized structure, the exact
integer transformation, and exact dimensionless shape scores. A score of zero
is a proof that the corresponding target was reached exactly.

```{literalinclude} ../../examples/build_a_supercell.py
:language: python
:lines: 18-
```
