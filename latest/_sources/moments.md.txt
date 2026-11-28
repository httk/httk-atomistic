# Site moments

Magnetic order adds a vector (or a signed scalar) to each site. *httk-atomistic*
holds site moments exactly, in Bohr magnetons, in whichever frame the data
arrived in — Cartesian, a bare collinear scalar, or along the lattice axes:

```python
from httk.atomistic import CartesianSiteMoments, CollinearSiteMoments, CrystalAxisSiteMoments, Cell

collinear = CollinearSiteMoments([2, -2, 0])            # signed scalars, no axis assigned
cartesian = CartesianSiteMoments([[0, 0, 2], [0, 0, -2]])  # per-site Cartesian vectors
crystal_axis = CrystalAxisSiteMoments(                  # components along â, b̂, ĉ
    [[0, 0, 3]], cell=Cell([[3, 0, 0], [0, 3, 0], [0, 0, 5]])
)
```

The three are different representations of the same physical quantity, so they
carry different information: `CollinearSiteMoments` fixes only a sign along an
unstated axis, `CartesianSiteMoments` fixes a full spatial direction, and
`CrystalAxisSiteMoments` expresses each moment in the unit lattice frame (`â`,
`b̂`, `ĉ`) — an *axial* quantity that transforms with the lattice, which is what
symmetry operations act on. `CartesianSiteMoments` and `CrystalAxisSiteMoments`
each have a matching `*View` for presentation; `CollinearSiteMoments`
deliberately has none, since its frame-ambiguous scalars cannot be presented as
a directed quantity. `SiteMomentsLike` names any of them where a function
accepts moments.

A `WyckoffSite` carries an optional `moment`, so an asymmetric unit can describe
a magnetic structure symmetry-distinctly; `SymopsStructure` is the
symmetry-explicit form (a cell, its listed sites, and the full magnetic
space-group operation list), with a `site_moments` argument and BNS
number/label fields. Magnetic CIFs are read through
`httk.atomistic.mcif_structures`:

```console
>>> from httk.core import load
>>> magnetic = load("structure.mcif")   # -> SymopsStructure with its site moments
```

The exactly-held moments and the magnetic space group survive loading; expanding
a `SymopsStructure` applies its operations to the axial moments in the lattice
frame. Loading is one-way: **writing magCIF is not yet supported** — no `.mcif`
writer is registered, and saving a loaded `SymopsStructure` to `.cif` degrades to
a nonmagnetic P1 CIF that drops the moments.

## Standardization carries moments through

{py:func}`~httk.atomistic.conventional_cell` and
{py:func}`~httk.atomistic.primitive_cell` are nuclear standardizations: they change the
setting from the atomic positions alone and never treat moments as symmetry input. Because
neither operation rotates Cartesian axes, `CartesianSiteMoments` and `CollinearSiteMoments`
ride through unchanged as per-site data, re-attached to the standardized sites.

That carry-through is only possible when the target cell can still represent the magnetic
order. When translation images collapse onto one site with disagreeing moments, or a
magnetic supercell folds to a smaller cell, both functions raise `ValueError` —
*magnetic order incompatible with the primitive cell; keep the original setting* — rather
than silently dropping the surplus moments. `CrystalAxisSiteMoments` are stated against the
old lattice frame and cannot survive a cell change, so both functions refuse them outright;
convert to `CartesianSiteMoments` first.
