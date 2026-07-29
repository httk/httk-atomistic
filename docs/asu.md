# Asymmetric units

A crystal is mostly repetition. `ASUStructure` represents the part that is not:
a space group, one site per symmetry orbit, and the values of that site's free
parameters. Expanding it produces the full unit cell; recognizing a full cell
produces it back.

```python
from httk.atomistic import UnitcellStructureView, load_asu_structure

asu = load_asu_structure("nacl.cif")
structure = UnitcellStructureView(asu)     # the full cell, exactly
```

`ASUStructure` is part of `StructureLike`, so it can be passed anywhere a
structure is accepted — the OPTIMADE provider, the numeric layer, everything.
The expansion is lazy: building one, or reading its space group, never generates
the cell.

## What it holds

| | |
| --- | --- |
| `cell` | the cell, in the structure's *own* setting |
| `spacegroup` | the space group, as its International Tables **standard** setting |
| `transform` | a {py:class}`~httk.atomistic.SettingTransform` from that standard setting to the structure's own |
| `asu_sites` | one {py:class}`~httk.atomistic.ASUSite` per orbit: a Wyckoff letter, its free parameters, and a species name |
| `species` | the species the sites name |

An `ASUSite` carries a bare Wyckoff letter (`"e"`, not `"4e"`) and one exact
value per degree of freedom — none at all for a fixed position such as an
inversion centre. Partial occupancy needs nothing special: it lives in the
referenced {py:class}`~httk.atomistic.Species`, which already carries a
composition.

## Any setting, including untabulated ones

A space group can be written in many *settings*, and they are genuinely
different coordinate systems. Wyckoff letter `e` of space group 15 reads
`0,y,1/4` in the standard setting `15:b1` and `1/4,0,z` in `15:c1`.

httk records the Wyckoff data against the standard setting **always**, and
carries a transform to whatever setting the structure is actually in. Because
that transform is *stored* rather than looked up, a setting that appears in no
table is representable exactly as well as a tabulated one:

```python
from httk.atomistic import ASUSite, ASUStructure, SettingTransform, Spacegroup
from httk.core import FracVector

# A tabulated setting: look the transform up.
tabulated = Spacegroup.for_setting("15:c1").transform_from_standard

# A setting in no table: state the transform yourself.
shifted = SettingTransform(FracVector.eye((3, 3)), ["1/8", "1/8", "1/8"])
```

`asu.setting()` returns the tabulated `Spacegroup` when there is one and `None`
when there is not — an untabulated setting simply has no name, which does not
stop it working.

### The transform is never derived

httk will not search for a transform that maps one group onto another. Such a
search is massively underdetermined: for space group 15 alone there are **192**
valid candidates using only integer entries in `{-1,0,1}` and translations on a
1/24 grid, and the full solution set — the affine normalizer coset — is
infinite. Crucially, those candidates describe *different crystals*. Picking one
arbitrarily would silently produce the wrong structure, so a setting that cannot
be identified is refused with an explanation instead.

### Direction

The transform maps *standard* coordinates into the structure's own setting.
Under httk's row-vector convention:

```
f_own = f_std * M.T() + v
f_std = (f_own - v) * inv(M).T()
B_own = inv(M).T() * B_std          # cell basis rows
```

Applying it backwards yields a structurally valid but systematically *wrong*
crystal.

`det M` is 1 for 520 of those settings and **3** for the seven
rhombohedral-axes ones (IT numbers 146, 148, 155, 160, 161, 166, 167), where the
standard hexagonal cell holds three primitive rhombohedral cells. Expansion
handles that: the standard orbit is three times too large for the smaller cell
and the surplus points coincide *exactly* under wrapping.

## Expansion is exact

There is no coordinate grid, no snapping, no neighbour search, and no tolerance
anywhere in the path from an `ASUStructure` to a full cell. Reduced coordinates,
symmetry operations, Wyckoff parameters, and the setting transform are all exact
rationals, and the vendored orbit tables are complete and pre-deduplicated, so
expansion is affine arithmetic over ℚ followed by an exact equality test.

```python
structure = UnitcellStructureView(asu)
structure.sites.reduced_coords.to_fractions()
# [[Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)], ...]
```

The cell survives untouched, so a hexagonal cell keeps its exact `sqrt(3)`.

For an existing ASU — including a full-cell view that is still backed by one —
`conventional_cell()` returns the same crystal re-expressed in the space group's IT
standard-setting conventional cell. It keeps the ASU's standard-to-own transform in the
result for provenance, while the result ASU has an identity transform; the basis change and
expansion remain exact. For a plain `Structure`, the operation first runs tolerant
`recognize_asu()`: measured coordinates may be snapped, and the result records the transform
chosen by recognition rather than an unstated transform from the input. Consequently a noisy
plain structure need not be `same_crystal()`-equal to the result. The returned `multiplier`
records the conventional/original site-count ratio, including the factor of three for a
rhombohedral setting.

## Recognition is where tolerance lives

Going the other way — from a full cell to an asymmetric unit — means deciding
which Wyckoff position each site is *meant* to occupy. A measured structure's
coordinates lie *near* a symmetric arrangement rather than on one, so this step
is tolerant, and it is the only one that is.

```python
from httk.atomistic import recognize_asu

asu = recognize_asu(structure, setting=Spacegroup.standard(225))
```

The space group can be supplied three ways, in decreasing order of preference:

`setting=`
: the structure's own tabulated setting, as when a CIF names its group. Nothing
  is searched for and spglib is not involved.

`standard=` and `transform=`
: for a structure in a setting that appears in no table. Also spglib-free.

neither
: the symmetry is found with **spglib**, which must be installed —
  `pip install httk-atomistic[default]`. This is the only path that needs it.

`tolerance` is a **Cartesian distance measured in the real cell**, so it means
the same thing along a short axis and a long one; a fractional tolerance would
not.

Left unspecified it is **derived from how precisely the structure was stated** —
see {doc}`precision`. A file written to four decimals in a 5 Å cell derives
`1e-3`; the same file written to three decimals derives ten times that, and is
matched correctly where a fixed value would silently put its atoms on the wrong
Wyckoff position. A structure that states no precision falls back to
{py:data}`~httk.atomistic.DEFAULT_TOLERANCE`.

### The exactness contract

1. A file's decimals embed as the rational the file literally wrote. `0.3333` is
   `3333/10000` — never silently `1/3`, and never `float("0.3333")`, whose exact
   value is `6004199023210345/18014398509481984` and claims a precision nobody
   stated.
2. Choosing a position is the only tolerant step, and it is a discrete decision.
3. The chosen position then supplies **exact** values for its fixed components.
   `0.2499` does not travel onward; the position says that component is `1/4`.
   Only the free parameters keep the measured value. Idealising those too is
   available through `limit_denominator=`, but it is opt-in, because turning
   `0.3333` into `1/3` is a claim about the data that only you can make.
4. After that the object is exact and closed.

```{admonition} Expansion is lossless; recognition is not
:class: important

`expand → recognize → expand` is idempotent. `recognize → expand` is **not** the
identity on the input coordinates, and must not be relied on as though it were —
recognition deliberately snaps a measured structure onto an idealised symmetric
one.
```

## Round-tripping

`Structure → ASUStructure → Structure` reproduces the same crystal exactly, with
no numerical drift, and for input that was already exact the coordinates come
back *identical*:

```python
from httk.atomistic import same_crystal

rebuilt = UnitcellStructureView(recognize_asu(structure, setting=setting))
assert same_crystal(structure, rebuilt)
```

{py:func}`~httk.atomistic.same_crystal` is the predicate that guarantee is
stated in: same cell and same *multiset* of (species, wrapped coordinate),
ignoring site order and which lattice translate was written down. Site ordering
is not preserved — the rebuilt structure follows canonical expansion order — so
`Structure.__eq__`, which compares site lists element by element, is stricter
than crystallographic identity and is the wrong tool here.

## Reading CIFs

A CIF is the natural source for an asymmetric unit: it lists one site per orbit
and states the operations that generate the rest, so no symmetry search is
needed.

```python
from httk.atomistic import asu_structures_from_cif, load_asu_structure, load_structure
from httk.core import load

asu = load_asu_structure("structure.cif")          # keeps the ASU form
structure = load_structure("structure.cif")        # expands to the full cell
everything = asu_structures_from_cif(load("multi.cif", raw=True))   # one per data block
```

The setting is identified by comparing the file's symmetry **operations**
against the tabulated ones **exactly**, so a file written in a non-standard
setting is recognized as such rather than reinterpreted.

What the file *declares* — a Hall symbol, or an International Tables number — is
treated as a claim to be checked, not a hint. A declaration naming no known
setting, or naming one whose operations are not the file's, is a genuine
inconsistency and raises: the two halves of the file disagree, and quietly
believing one of them is how a wrong structure gets built. Hall symbols are
matched in the conventional spelling a CIF writes (`-C 2yc`), normalized to the
spelling the tables index.

When the operations are the trustworthy half — a file whose symbols are known to
be wrong — pass `trust_declared_symmetry=False` to ignore the declaration and
identify the setting from the operations alone:

```python
asu = load_asu_structure("misdeclared.cif", trust_declared_symmetry=False)
```

The Hermann-Mauguin symbol is deliberately not checked: it has too many
legitimate spellings, and OPTIMADE itself notes that it does not unambiguously
communicate the axis, cell, or origin choice.

The cell is built exactly from `a, b, c, α, β, γ` rather than from a
pre-multiplied floating-point basis, so a cubic cell keeps exact right angles.
Occupancies become the composition of the corresponding `Species`.

Reading is tolerant of blocks that are not structures — a CIF may hold
bibliographic entries or an incomplete draft — so `httk.core.load` never fails
on their account; it records why each was skipped under `unparsed`, and asking
for structures raises with those reasons rather than returning an empty list.

## Serving symmetry over OPTIMADE

`StructureEntryProvider` serves an `ASUStructure`'s symmetry automatically. The standard
OPTIMADE properties — `space_group_it_number`, the Hall and Hermann-Mauguin symbols,
`space_group_symmetry_operations_xyz`, `wyckoff_positions`, `fractional_site_positions`,
and `site_coordinate_span` — come straight from the ASU. A plain `Structure` carries no
symmetry, so they serve `null`; inferring a space group there would mean running a
symmetry search behind the caller's back, with a tolerance nobody chose, on every record.

The symbols and Wyckoff letters describe **the setting the structure is written in**, as
OPTIMADE requires, not the standard setting the ASU stores them against. That distinction
is not cosmetic: setting `224:1` permutes Wyckoff letters `i` and `j`, and the rhombohedral
settings change multiplicities, so a 3a of the standard hexagonal cell is served as the 1a
it is in the smaller cell.

For what OPTIMADE does not standardise — which setting, and the change of basis to it —
httk serves six provider-specific properties whose definitions are taken verbatim from
[schemas.httk.org](https://schemas.httk.org) rather than paraphrased locally:

| Property | Describes |
| --- | --- |
| `_httk_setting_it_nc` | the setting code, e.g. `15:c1` |
| `_httk_hall_entry` | the normalized Hall symbol naming the setting |
| `_httk_is_reference_setting` | whether it is the IT standard setting |
| `_httk_crystal_system` | the crystal system |
| `_httk_centring_type` | the lattice centring letter |
| `_httk_setting_transform` | the standard-to-own change of basis, as exact rationals |

The `_httk_` prefix is what OPTIMADE requires of a database-specific property *name*; the
definition keeps its own `$id`, so a client that follows it reaches the published schema.
A structure in an untabulated setting serves its space-group number and its transform —
enough to reconstruct it — but `null` for the symbols and Wyckoff letters, which name a
setting that in that case has no name.

## The symmetry tables

The Wyckoff positions, symmetry operations, and setting transforms come from two
CC BY 4.0 datasets vendored in `httk.atomistic.data`: 527 space-group settings
(230 of them standard settings) and the change of basis from each to its
standard setting. They are read lazily — nothing is loaded until symmetry is
actually used. See the package's `README.md` for provenance and attribution.

## See also

- {doc}`structures` — the `Structure` model these expand into
- {doc}`precision` — where the default tolerance comes from
- {doc}`examples/asu_from_cif` — a runnable walkthrough of everything above
