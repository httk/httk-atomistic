# Asymmetric units in detail

A crystal is mostly repetition. `ASUStructure` represents the part that is not:
a space group, one site per symmetry orbit, and the values of that site's free
parameters. Expanding it produces the full unit cell; recognizing a full cell
produces it back.

```python
from httk.atomistic import UnitcellStructureView
from httk.core import load

asu = load("nacl.cif")
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
| `spacegroup` | the tabulated setting that names the stored Wyckoff data |
| `transform` | the transform from that stored setting to the structure's own coordinates; normally identity |
| `wyckoff_sites` | one {py:class}`~httk.atomistic.WyckoffSite` per orbit: a Wyckoff letter, its free parameters, and a species name |
| `species` | the species the sites name |
| `bonds` | the optional bond network: one {py:class}`~httk.atomistic.SiteBond` per bond orbit, or `None` |
| `shapes` | optional supplied bond shapes, one symbol or `None` per site (needs `bonds`), or `None` |

An `WyckoffSite` carries a bare Wyckoff letter (`"e"`, not `"4e"`) and one exact
value per degree of freedom — none at all for a fixed position such as an
inversion centre. Partial occupancy needs nothing special: it lives in the
referenced {py:class}`~httk.atomistic.Species`, which already carries a
composition.

## Bond networks

An `ASUStructure` (or `FundamentalDomainStructure`) may carry a bond network,
`bonds`. `None`, the default of every way of building one, means no network is
known; `()` means the structure is known to have no bonds. A
{py:class}`~httk.atomistic.SiteBond` `(site, partner, operation)` bonds the
reference atom of `wyckoff_sites[site]` to `operation` applied to the reference
atom of `wyckoff_sites[partner]`. A reference atom is the first-listed coordinate
of the site's Wyckoff position in the stored setting at its free parameters, not
reduced modulo one, and `operation` is an exact element of the stored setting's
space group, lattice translation included. Each bond stands for its whole orbit
under the group: the constructor replaces it by the least member of its orbit
(first site not after the second; least element of the double coset of the two
reference atoms' site-symmetry groups, compared in the standard setting by the
length of the lattice translation, then operation and translation), so equal
networks in any spelling compare equal and repeats collapse. Self-bonds, an
operation outside the group and a site index out of range raise `ValueError`.
Equality includes the network (`None`, `()` and a network all differ), and the
storage records keep it; a structure without a network keeps its content identity.
The protonet labels can be built from it (see {doc}`/prototypes`); their net segment
then depends on the coordinates through the baseline shapes of the sites without a
supplied shape (below).

Every operation that only re-describes the same sites carries the network with
them: both atoms of every bond are mapped by the exact map between the two
descriptions and located in the new orbits. Where the sites themselves change,
the result has no network:

| carries the network | drops it (`bonds=None`) |
| --- | --- |
| the change to the standard setting (`conventional_cell`, the canonicalizers' standardization) | expansion to a unit cell, and recognition from one (`recognize_asu`, `ASUStructureView` of a unit cell) |
| `canonicalize` / `canonical_asu` (both `symmetry` modes), `canonical_asu_protostructure` and `canonical_asu_protostructure_assignments` (without `lift=True`), through recognition when the recognized group and lattice keep the bond set | subgroup descent and lift (`subgroup_representation`, `rerepresent` into another group, `lift_candidates`, `backward_lift`, `highest_symmetry`, `search_supergroups`) and the paths built on them (`represent_like`, `list_representations`, `canonicalize_full`, `interpolate_structures`) |
| re-describing a left-handed cell right-handed, the mirror image and `normalize_chirality`, the affine-normalizer actions, origin shifts, cell reductions (the reduction of a P1 supercell to its primitive cell only when the bonds keep the smaller cell's translations) and the orientation of the canonical cell | the legacy metric-first canonicalizers (`canonical_asu_legacy`, `canonicalize_legacy`), `lift=True` in `canonical_asu_protostructure` and its assignments, supercells, and the collapse of an exact supercell by a self-lift |
| reducing species to their classes, renaming and reordering sites; CIF reading, which reads `_geom_bond_` loops into the network (see {doc}`cif`) | templates and anonymization (`FundamentalDomainTemplate`, `Structuretype`) and the representatives of `Protostructure` and `Prototype` |

Supplied shapes (below) are carried and dropped with the network; sites with
different supplied shapes that merge drop both, as a bond set that is not
invariant does (`Protonetstructure.from_asu` raises instead).

Through recognition (`canonicalize` with `symmetry="detect"` and
`canonical_asu_protostructure`) every atom is matched once to the nearest
recognized atom, and the match must be one to one. Wherever the group or the
lattice changes, the bond set must be invariant under the new symmetry, checked
modulo both lattices, so a translation the new cell has and the old one lacks is
checked too; otherwise the network is dropped (and `Protonetstructure.from_asu`
raises). Descriptions tied in geometry are told apart by their normalized networks,
and then by their supplied shapes, so equal networks in different spellings
canonicalize to one value.

### Bond shapes

The *bond count* of a site is the number of distinct atoms bonded to one of its
atoms; two sites at one point are two atoms. The *shape* of a site with two or
more bonds is the arrangement of those bonds, named by a symbol that ends in the
bond count: `L2` (linear), `A2` (bent), `TP3`, `T4`, `SP4`, `OC6` and the other
IUPAC polyhedral symbols without the hyphen, a few SHAPE program labels, and
`I<n>` for an irregular arrangement of `n` bonds. Sites with fewer than two bonds
have no shape.

A structure with a network may carry `shapes`: one symbol or `None` per Wyckoff
site, where `None` means "use the baseline". The constructor raises `ValueError`
for shapes without `bonds`, a length other than the number of sites, an unknown
symbol, a symbol on a site with fewer than two bonds, or a symbol whose number is
not the site's bond count (`TypeError` for a value that is not a string or
`None`). Shapes that are all `None` are the same as no shapes. A site placed at a
special point of its own Wyckoff letter (the general letter at an inversion
centre, say) has no defined bond count: counting or checking shapes on it raises
`ValueError`, so give it the special letter.

The *baseline* detector gives the shape of every site without a supplied symbol,
on demand: it compares the sorted angles between the site's bonds with those of
the ideal shapes of its bond count and takes the closest, `I<n>` when that is more
than 15 degrees away (root mean square), and `L2` from 165 degrees for two bonds.
Bond lengths are not used. The angles come from the exact metric averaged over the
point group, each exact squared cosine converted to a float once, so every exact
re-description of one crystal (cell, origin, setting, mirror image) gives the same
angles, bit for bit, and so the same shapes. Crystals that are only approximately
related (independently rounded inputs) can differ where an angle lies within
rounding noise of a threshold or a tie. `math.acos` is not correctly rounded, so
platforms with different C maths libraries may differ in the last bit of an angle;
after the misfit is rounded to 1e-6 that changes a shape only for a value within
about 1e-12 of a rounding boundary. A zero-length bond (two coincident sites bonded to each
other) has no direction; the baseline raises `ValueError` for its site, and a
supplied symbol for that site avoids it. Networks read from CIF `_geom_bond_`
loops have no shapes of their own and use the baseline.

Supplied shapes move with their sites wherever the network is carried and are
dropped wherever it is dropped. Where sites merge (recognition, a cell
reduction, a P-1 supercell collapse), a site without a symbol takes its partner's,
and two different symbols on one merged site count as a network that is not
invariant: the result has neither network nor shapes (the P-1 collapse then keeps
the uncollapsed structure). Supplied shapes are part of equality and of the
content identity and are stored (as `""` for a site without a symbol); baseline
shapes are never stored, so a structure without supplied shapes keeps its content
identity.

### Default bonds from distances

A structure without a network can be given a simple default one:
{py:func}`~httk.atomistic.distance_bonds` returns
`ASUStructureView(structure).resolve()` with `bonds` found from distances. Two
atoms are bonded when their distance lies between 0.5 Å and the sum of their
covalent radii plus `tolerance` (0.45 Å by default): the radius-sum form used by
OpenBabel and Jmol, with a 0.5 Å floor (not their exact radii or floor). The radii
are the single-bond covalent radii of Cordero et al., *Dalton Trans.* 2008,
2832–2838 (low-spin Mn, Fe and Co; sp3 C). A mixed-occupancy site takes the
largest radius of its constituents (positive concentration; vacancies and attached
atoms are ignored), a site with only vacancies gets no bonds, and an element
without a radius raises `ValueError` naming it.

For an asymmetric-unit input the result has its sites unchanged. Any other input
(a unit cell, or a P1, primitive or supercell description) is recognised first,
and the recognised structure is idealised within the recognition tolerance, so a
contact within that tolerance of its cutoff can fall on either side. To control
it, recognise yourself: `distance_bonds(ASUStructureView(structure, tolerance=...))`.

On top of the distance rule, two metals are never bonded in a structure that
contains a non-metal: H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Ge, As, Se, Br,
Kr, Sb, Te, I, Xe, At and Rn (the metalloids count as non-metals, so Zintl pairs
such as Na–Si stay bonded). A mixed site is a metal when all its constituents are,
so even a dilute non-metal on a mixed site switches this rule on for the whole
structure; vacancy-only sites do not count. Without a non-metal (intermetallics,
elements) the distance rule alone applies. The cation radius sums are generous, so
without this rule the cations of ionic solids bond to each other (rutile TiO₂
would give Ti 6 O and 10 Ti neighbours; with it, Ti 6 and O 3).

```python
from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite, distance_bonds

nacl = ASUStructure(
    Cell([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]), 225,
    [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
    [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
)
bonded = distance_bonds(nacl)  # Na-Cl at 2.82 Å; Na-Na is metal-metal; Cl-Cl at 3.99 Å is beyond 2.49 Å
```

This is a heuristic, not a chemical model. Covalent radii suit covalent and
molecular structures better than ionic ones; the metal rule ignores metal–metal
bonds in compounds (Re–Re in Re₂Cl₈²⁻ salts, Hg₂²⁺), and anion–anion contacts
are judged by distance alone. Disordered structures need care:
pairs closer than 0.5 Å (split positions) are never bonded, but partially
occupied neighbours are bonded as if both were present. Distances are floats
computed from the exact structure, so every exact asymmetric-unit re-description of one
crystal gives the same network except where a distance lies within float noise of its
cutoff.

## Any setting, including untabulated ones

A space group can be written in many *settings*, and they are genuinely
different coordinate systems. Wyckoff letter `e` of space group 15 reads
`0,y,1/4` in the standard setting `15:b1` and `1/4,0,z` in `15:c1`.

httk records Wyckoff data directly against a tabulated source setting. This avoids a
change of basis during reading, expansion, storage, and writing. A setting that appears in
no table is still represented against the standard table with an explicit transform:

```python
from httk.atomistic import WyckoffSite, ASUStructure, SettingTransform, Spacegroup
from httk.core import FracVector

# A tabulated setting is stored directly.
tabulated = Spacegroup.for_setting("15:c1")

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

For an untabulated setting, the stored transform maps *standard* coordinates into the
structure's own setting. A tabulated setting derives the same transform only if a caller
explicitly requests standardization.
Under httk's row-vector convention:

```
f_own = f_std * M.T() + v
f_std = (f_own - v) * inv(M).T()
B_own = inv(M).T() * B_std          # cell basis rows
```

Applying it backwards yields a structurally valid but systematically *wrong*
crystal.

`det M` is 1 for 520 tabulated setting conversions and **3** for the seven
rhombohedral-axes ones (IT numbers 146, 148, 155, 160, 161, 166, 167), where the
standard hexagonal cell holds three primitive rhombohedral cells. Ordinary expansion uses
the rhombohedral table directly; the volume change matters only during standardization.

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
standard-setting conventional cell. It returns the derived standard-to-own transform in the
result for provenance, while the result ASU has an identity transform; the basis change and
expansion remain exact. For a plain `UnitcellStructure`, the operation first runs tolerant
`recognize_asu()`: measured coordinates may be snapped, and the result records the transform
chosen by recognition rather than an unstated transform from the input. Consequently a noisy
plain structure need not be `same_crystal()`-equal to the result. The returned `multiplier`
records the conventional/original site-count ratio, including the factor of three for a
rhombohedral setting. Cartesian and collinear site moments carry through as per-site data;
crystal-axis moments and magnetic orders that a cell change cannot represent are refused —
see {doc}`../moments`.

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
   `3333/10000` — never silently `1/3`. A Python float embeds as its shortest
   decimal, so `0.3333` passed as a float is also `3333/10000`, not the binary
   value `6004199023210345/18014398509481984`, which claims a precision nobody
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

`UnitcellStructure → ASUStructure → UnitcellStructure` reproduces the same crystal exactly, with
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
`UnitcellStructure.__eq__`, which compares site lists element by element, is stricter
than crystallographic identity and is the wrong tool here.

## Reading CIFs

A CIF is the natural source for an asymmetric unit: it lists one site per orbit
and states the operations that generate the rest, so no symmetry search is
needed.

```python
from httk.atomistic import UnitcellStructureView, asu_structures_from_cif
from httk.core import load

asu = load("structure.cif")                         # keeps the native ASU form
structure = UnitcellStructureView(asu)               # expand explicitly
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
asu = asu_structures_from_cif(load("misdeclared.cif", raw=True), trust_declared_symmetry=False)[0]
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
and `site_coordinate_span` — come straight from the ASU. A plain `UnitcellStructure` that
states no symmetry serves `null` for all of them, `space_group_symmetry_operations_xyz`
included: that property is the complete operation set fitting what was stated, so an invented
`["x,y,z"]` would be a false P1 claim. OPTIMADE allows the `null` — the property is nullable,
and a `unit_cell` span is reconstructed by lattice translations alone. Inferring a space group
there would mean running a symmetry search behind the caller's back, with a tolerance nobody
chose, on every record. Operations the structure does state are served as given; a
structure that states only a space-group number or symbol serves `null` operations, since the
operations of that group in the structure's own cell cannot be derived from the number alone.

The symbols and Wyckoff letters describe **the setting the structure is written in**, as
OPTIMADE requires. Setting `224:1` therefore stores and serves its own letter `i`, not the
standard setting's corresponding `j`; rhombohedral settings likewise retain their local
multiplicities.

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

- {doc}`structures` — the `UnitcellStructure` model these expand into
- {doc}`precision` — where the default tolerance comes from
- {doc}`/examples/asu_from_cif` — a runnable walkthrough of everything above
