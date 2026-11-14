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
| `bonds` | the optional bond network, a {py:class}`~httk.atomistic.BondNetwork` (one {py:class}`~httk.atomistic.SiteBond` per bond orbit, optional shapes, sources), or `None` |

An `WyckoffSite` carries a bare Wyckoff letter (`"e"`, not `"4e"`) and one exact
value per degree of freedom — none at all for a fixed position such as an
inversion centre. Partial occupancy needs nothing special: it lives in the
referenced {py:class}`~httk.atomistic.Species`, which already carries a
composition.

## Bond networks

Every structure exposes `bonds`: a {py:class}`~httk.atomistic.BondNetwork`, or
`None` when no network is known (the default of every way of building one). A
network holds `pairs`, the bonds (none means the structure is known to have no
bonds); `shapes`, one shape symbol or `None` per site, or `None` when no shapes are
assigned (below); and `source` and `shapes_source`, a
{py:class}`~httk.atomistic.BondSource` each (free-text `description`, `software`,
`version`, `uri`, every field a string or `None`), naming who assigned the bonds
and the shapes (`shapes_source` `None`: the same as `source`). The pairs and the
shapes are content; the sources are provenance and take part neither in equality
nor in the content identity. Constructors take `bonds=` as a network, as a mapping
with its keys (`pairs` required), or as a plain sequence of bonds (no shapes, no
source). A `shapes_source` without shapes, an empty source field (use `None`) and an
unknown key raise `ValueError`.

On an `ASUStructure` (or `FundamentalDomainStructure`) the pairs and shapes index
`wyckoff_sites`, and a
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
Equality includes the network (`None`, no pairs and pairs all differ), and the
storage records keep it; a structure without a network keeps its content identity.
The protonet labels can be built from it (see {doc}`/prototypes`); their net segment
then depends on the coordinates through the baseline shapes of the sites without a
supplied shape (below).

Every operation that describes the same crystal carries the network, with its shapes and
sources: both atoms of every bond are mapped by the exact map between the two descriptions and
located among the new sites. An operation that adds, removes or substitutes atoms drops it.
*When invariant* below means that the bond set and the assigned shapes must be invariant under
the new group and lattice, checked exactly and modulo both lattices (so a translation the new
cell has and the old one lacks is checked too); otherwise the network is dropped.

| operation | network |
| --- | --- |
| the change to the standard setting (`conventional_cell` of a domain, the canonicalizers' standardization) | carried |
| `canonicalize` / `canonical_asu` (both `symmetry` modes), `canonical_asu_protostructure` and `canonical_asu_protostructure_assignments` (without `lift=True`) | carried when invariant |
| re-describing a left-handed cell right-handed, the mirror image and `normalize_chirality`, the affine-normalizer actions, origin shifts and the orientation of the canonical cell | carried |
| cell reductions of a domain (a P1 supercell to its primitive cell) | carried when invariant under the translations given up |
| reducing species to their classes, renaming and reordering sites; CIF reading, which reads `_geom_bond_` loops into the network (see {doc}`cif`) | carried |
| domain to unit cell: `UnitcellStructureView` of a domain and its `unview()`, and the `sites` that `CIFStructure`, `DatastreamStructure` and `RecordStructure` present (below) | carried |
| unit cell to domain by recognition: `recognize_asu`, `ASUStructureView`, `conventional_cell` and `primitive_cell` of a unit cell, and the canonicalizers above given a unit cell | carried when invariant |
| supercells: `build_supercell`, `orthogonal_supercell` and `cubic_supercell` (any input, presented as a unit cell first) | carried |
| cell re-descriptions of a unit cell: `niggli_reduced`, and its reduction to a smaller cell (the last step of `primitive_cell`) | carried; a reduction when invariant |
| `UnitcellStructure.rationalized()`, and `without_charges()` of any structure (whose result is a unit cell, so a domain's network arrives expanded onto its sites) | carried (the same atoms in the same order; snapping moves coordinates, not atoms) |
| subgroup descent: `subgroup_representation`, `rerepresent` into a subgroup, `common_subgroup_representation`, and the paths built of descents, normalizer images and re-descriptions (`represent_like`, `list_representations`, `canonicalize_full`) | carried: a subgroup keeps every network of its group |
| `NumericUnitcellStructureView` | carried (`NumericUnitcellStructure` drops it) |
| lifts: `lift_candidates`, `backward_lift`, `highest_symmetry`, `search_supergroups`, `rerepresent` into a supergroup, the legacy metric-first canonicalizers (`canonical_asu_legacy`, `canonicalize_legacy`), `lift=True` and the collapse of an exact supercell by a self-lift | dropped (below) |
| coincident sites that cannot be told apart: two sites of one species at one point wherever a unit-cell network is carried, and any coincident sites of a unit cell given to the canonicalizers, whose anonymous frames rename the species | dropped |
| `interpolate_structures` | dropped on the frames between the end points (new geometry); the end points keep theirs |
| templates and anonymization (`FundamentalDomainTemplate`, `Structuretype`) and the representatives of `Protostructure` and `Prototype` | dropped (atoms substituted; no identity depends on bonds) |
| adding, removing or substituting atoms; the ASE and pymatgen bridges; file writers (CIF writes no `_geom_bond_` loop); trajectory frames | dropped, or not represented |

Lifts are the one deviation from the rule: a lift describes the same atoms, idealized, yet it
still drops the network; carrying it by matching atoms, as recognition does, is left to a later
version.

Assigned shapes (below) and the sources are carried and dropped with the network; sites with
different assigned shapes that merge drop both, as a bond set that is not
invariant does (`Protonetstructure.from_asu` raises instead).

Through recognition (`recognize_asu`, `ASUStructureView`, `canonicalize` with
`symmetry="detect"` and `canonical_asu_protostructure`, from a domain or a unit cell) every
atom is matched once to the nearest recognized atom, and the match must be one to one; the
bond set must then be invariant under the recognized symmetry. Otherwise the network is dropped
(and `Protonetstructure.from_asu` raises). A unit cell with a network gets the protonet labels
of the asymmetric unit recognition gives it, `ASUStructureView(cell)`: the labels of that
description, as for `BareProtostructureView(cell)`, so two descriptions of one crystal (another
origin, site order or cell) can be labelled differently, while `canonical_protonetstructure`
gives them one label. `ProtonetstructureView(cell)` raises `ValueError` when
recognition drops the network, and for a cell that is not periodic in three dimensions (label a
molecular fundamental domain instead). Descriptions tied in geometry are told
apart by their normalized networks, and then by their assigned shapes, so equal networks in
different spellings canonicalize to one value.

### Bond shapes

The *bond count* of a site is the number of distinct atoms bonded to one of its
atoms; two sites at one point are two atoms. The *shape* of a site with two or
more bonds is the arrangement of those bonds, named by a symbol that ends in the
bond count: `L2` (linear), `A2` (bent), `TP3`, `T4`, `SP4`, `OC6` and the other
IUPAC polyhedral symbols without the hyphen, a few SHAPE program labels, and
`I<n>` for an irregular arrangement of `n` bonds. Sites with fewer than two bonds
have no shape.

A network may carry `shapes`: one symbol or `None` per Wyckoff site, where `None`
means "not assigned: use the baseline". The constructor raises `ValueError` for a
length other than the number of sites, an unknown symbol, a symbol on a site with
fewer than two bonds, or a symbol whose number is not the site's bond count
(`TypeError` for a value that is not a string or `None`). Shapes that are all
`None` are the same as no shapes. A site placed at a
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
loops use the baseline unless the file carries an httk `_httk_shape_` loop.

Assigned shapes move with their sites wherever the network is carried and are
dropped wherever it is dropped. Where sites merge (recognition, a cell
reduction, a P-1 supercell collapse), a site without a symbol takes its partner's,
and two different symbols on one merged site count as a network that is not
invariant: the result has neither network nor shapes (the P-1 collapse then keeps
the uncollapsed structure). Assigned shapes are part of equality and of the
content identity and are stored (as `""` for a site without a symbol); baseline
shapes are never stored, so a structure without assigned shapes keeps its content
identity.

### Networks on unit cells

A {py:class}`~httk.atomistic.UnitcellStructure` takes `bonds=` in the same three
forms, between its explicit sites and without symmetry: the pairs and shapes index
`sites`, and the operation of each {py:class}`~httk.atomistic.SiteBond` must be a
lattice translation `n` (zero along a non-periodic direction): the atom at
`sites[site]`, as stored and not wrapped, is bonded to the atom at `sites[partner]`
moved by `n`, and the bond stands for itself and its lattice translates. A
fractional translation, any other operation, a site index out of range and a bond
of an atom to itself (`site == partner` with `n = 0`) raise `ValueError`. The pairs
are put in a normal form, so equality does not depend on the order or spelling
given: each bond once, oriented `site <= partner` (`(i, j, n)` with `i > j` becomes
`(j, i, -n)`; a bond of a site to its own translate keeps the lexicographically
smaller of `n` and `-n`), sorted by `(site, partner, n·n, n)`. The bond count of a
site is the number of distinct `(partner, translation)` atoms bonded to it, so a
bond of a site to its own translate counts two; shapes are validated against it as
above. The baseline reads the angles from the cell's exact metric, without
averaging: a unit cell lists every atom of an orbit separately, so every exact
re-description (cell, origin, mirror image, site order, supercell) still gives
bit-identical angles. Where the metric keeps the group exactly the unit-cell
baseline equals the fundamental domain's on its expansion; where it does not (the
rhombohedral-axes cells), atoms of one orbit can get different symbols.

`UnitcellStructureView` of a fundamental domain presents the domain's network on
the unit-cell sites it presents: every bond orbit expanded into explicit pairs,
every atom with its site's shape, the sources unchanged (a molecular domain whose
retained representative is not an exact orbit point drops it).
`CIFStructure` and `DatastreamStructure` expose the network of the unit-cell sites
they present, `RecordStructure` that of a fundamental-domain record expanded to its
unit cell, and `NumericUnitcellStructureView` that of its exact structure.
`NumericUnitcellStructure`, `PlainStructureView`, the ASE and pymatgen views, trajectory
frames and the other structure classes have no network (`None`). A unit-cell storage
record keeps its network (below), and `RecordStructure` of one gives it as stored. Where a
carry drops a network, the reason is logged at debug level with the context `bonds`.

### Storing networks

The storage records keep the network with its shapes and sources. A fundamental-domain
or asymmetric-unit record stores its pairs and assigned shapes in its own `bonds` and
`shapes` fields; a unit-cell record references a
`UnitcellBondNetworkRecord`, one exact row
`(site, partner, n1, n2, n3)` per bond in the normal form and the shapes (`""` for a site
without a symbol), so equal networks share one row. The pairs and shapes are part of the
content identity; a structure without a network keeps the identity it had before networks
were stored. The sources are `BondSourceRecord`
references on the structure record (`bonds_source`, `shapes_source`), outside the content
identity: a different source never changes a content id. A `shapes_source` equal to the
`source` is the same provenance and is stored as none. Validation rejects a stored network
that is not in its normal form, all-`""` shapes, a `bonds_source` without bonds, a
`shapes_source` without shapes and a `shapes_source` equal to the `bonds_source`.

Because sources are outside the content identity but kept, **re-saving equal content with a
different source cannot be recorded**: `save`, `replace` and `bulk_ingest` raise
`EntryMetadataConflictError` on every backend and the stored source stays. A stored
*unknown* source counts as a different source too: re-saving a networked row with a source
for the first time raises the same way. That is the case for every networked row of an
upgraded store (below), whose sources are unknown. To record the new assignment, rebuild the
store, or save it as an alternative (`alternative_of`).

Stores written by a development version before unit-cell networks were stored reopen with
`upgrade=True`: the new tables and nullable references are added, and every stored row keeps
its content id and network, with unknown sources. Stores written with *httk-store* storage
protocol 2 (the *httk-store* releases up to 2.1.4) need a rebuild in any case.

### Default bonds from distances

{py:func}`~httk.atomistic.distance_bonds` gives a structure a simple default
network found from distances, in place of any network it has. Two
atoms are bonded when their distance lies between 0.5 Å and the sum of their
covalent radii plus `tolerance` (0.45 Å by default): the radius-sum form used by
OpenBabel and Jmol, with a 0.5 Å floor (not their exact radii or floor). The radii
are the single-bond covalent radii of Cordero et al., *Dalton Trans.* 2008,
2832–2838 (low-spin Mn, Fe and Co; sp3 C). A mixed-occupancy site takes the
largest radius of its constituents (positive concentration; vacancies and attached
atoms are ignored), a site with only vacancies gets no bonds, and an element
without a radius raises `ValueError` naming it.

The result is the kind of structure given, with its sites unchanged. It never
recognises symmetry itself; an `ASUStructureView` input resolves its own
recognition. A fundamental domain or asymmetric unit (an `ASUStructureView` is
resolved) keeps its class and gets a network of bond orbits. A unit cell or
`UnitcellStructureView` gives `UnitcellStructureView(structure).unview()` with
explicit pairs; along a non-periodic direction atoms are neither wrapped nor
repeated, so a molecule in a non-periodic cell is bonded as given. Raw inputs (a
path, a text stream, a storage record) are first adopted as structures. Lazy sources
(`CIFStructure`, a CIF file, `DatastreamStructure`) are resolved, and a
fundamental-domain or asymmetric-unit record (or its `RecordStructure`) gives its
domain; then these two rules apply. A unit-cell record gives a unit cell, and any
other structure, a `NumericUnitcellStructure` included, is presented as a unit cell. For the network of the recognised asymmetric unit, recognise first:
`distance_bonds(ASUStructureView(structure, tolerance=...))`. The recognised
structure is idealised within the recognition tolerance, so a contact within that
tolerance of its cutoff can fall on either side.

The network names its rule as its `source`: software `httk_basic`, version the
installed *httk-atomistic* version (a development install reports the last
release), `uri` `None`, and the one-sentence description `Distance bonds: 0.5 Å <=
d <= r(a) + r(b) + 0.45 Å (Cordero et al. 2008 covalent radii), no metal-metal bonds
where a non-metal is present, and no metal to non-metal bond longer than r(a) + r(b)
across a shared, more electronegative non-metal.`, with the `tolerance` given. With
`shapes=True`, the default, every site with two or more bonds also gets its baseline
shape (`shapes_source` `None`: the same source); `shapes=False` leaves the shapes
unassigned. A baseline that cannot assign a shape raises `ValueError` suggesting
`shapes=False`; that includes a fundamental-domain site placed at a special point of
its own Wyckoff letter. {py:func}`~httk.atomistic.assign_baseline_shapes` assigns the
baseline shapes to a network without shapes, from any source: it keeps the pairs and
`source`, and names the baseline in `shapes_source` (software `httk_basic`, the
installed version). It refuses a structure without a network or with shapes, and
returns one where no site has two bonds unchanged. A new version changes the stored
source, so re-saving an unchanged network into an existing store raises (see
[Storing networks](#storing-networks)).

**Measured cells.** On a cell whose metric keeps the group only approximately, the
two routes can label one crystal differently. On the unit cell every atom gets its
own baseline shape and contacts keep their measured, float-noisy distances, so
recognition may later find the network not invariant and drop it (logged at debug
level with the context `bonds`; `ProtonetstructureView` of the cell then raises). The asymmetric-unit route,
`distance_bonds(ASUStructureView(cell, tolerance=...))`, idealises first. Recognise
first when the labels matter.

On top of the distance rule, two metals are never bonded in a structure that
contains a non-metal: H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Ge, As, Se, Br,
Kr, Sb, Te, I, Xe, At and Rn (the metalloids count as non-metals, so Zintl pairs
such as Na–Si stay bonded). A mixed site is a metal when all its constituents are,
so even a dilute non-metal on a mixed site switches this rule on for the whole
structure; vacancy-only sites do not count. Without a non-metal (intermetallics,
elements) the distance rule alone applies. The cation radius sums are generous, so
without this rule the cations of ionic solids bond to each other (rutile TiO₂
would give Ti 6 O and 10 Ti neighbours; with it, Ti 6 and O 3).

The bonds found so far form the *tentative network*. Finally, a cation does not
bond to an anion centre across a ligand they share: a tentative bond between a
metal *a* and a non-metal *b* is dropped when both

1. it is longer than the sum of the two radii, `d(a, b) > r(a) + r(b)`, so it
   exists only through the tolerance, and
2. some non-metal *c* tentatively bonded to both *a* and *b* is more
   electronegative than both.

All drops are decided on the tentative network in one pass; a drop never
changes another decision. Bonds between two metals or between two non-metals
are never dropped. Without this rule the oxo-anion centres of oxo-salts bond to
the large cations around them: diopside CaMgSi₂O₆ gives calcium 12 bonds, four
of them to silicon, and silicon six; with it, calcium 8 and a silicon
tetrahedron. The electronegativities are Pauling's as compiled by
[WebElements](https://www.webelements.com/periodicity/eneg_pauling/): Allred's
revised values (*J. Inorg. Nucl. Chem.* 17, 215 (1961)), else Pauling's (1960),
and for the noble gases and Am–Cm Huheey, Keiter and Keiter (1993). He, Ne, Ar,
Rn, Pm, Eu, Tb and Yb have no value there, and none is invented. A metal *a*
without a value (Pm, Eu, Tb, Yb) counts as less electronegative than any
non-metal with one, so a valued non-metal *c* parts it from *b* like any other
cation. A non-metal without a value (He, Ne, Ar, Rn) never acts as *b* or *c*,
so its bonds are never dropped. A mixed site is a metal *a* only when all its
constituents are metals and a non-metal *c* only when all are non-metals; it
counts with its constituents' largest value as *a* or *b* and their smallest as
*c*, so a bond is dropped only when the condition holds for every constituent
with a value. A mixed metal site ignores its constituents without a value when
taking the largest (and counts as a valueless metal when none has one); any
other mixed site with a constituent without a value never acts as *b* or *c*.
The conditions depend on distances and sites only and are symmetric in *a* and
*b*, so whole bond orbits are kept or dropped, and the network stays the same
for every exact re-description.

The length condition keeps a metal–non-metal bond that is shorter than the
radius sum. Its cost is that a contact between a very large cation (Cs, Rb, Ba)
and an oxo-anion centre that is shorter than their radius sum survives.
Dropping the length condition was measured on ICSD and COD samples and not
adopted: it parted few more such pairs, and in organometallic entries it also
dropped real metal–carbon and metal–hydrogen bonds. The condition does not
protect every metal–ligand bond, though: see the limitations below.

```python
from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite, distance_bonds

nacl = ASUStructure(
    Cell([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]), 225,
    [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
    [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
)
bonded = distance_bonds(nacl)  # Na-Cl at 2.82 Å; Na-Na is metal-metal; Cl-Cl at 3.99 Å is beyond 2.49 Å
assert bonded.bonds is not None and bonded.bonds.shapes == ("OC6", "OC6")
```

This is a heuristic, not a chemical model. Covalent radii suit covalent and
molecular structures better than ionic ones; the metal rule ignores metal–metal
bonds in compounds (Re–Re in Re₂Cl₈²⁻ salts, Hg₂²⁺), and anion–anion contacts
are judged by distance alone. The shared-ligand rule cannot part a cation from
an anion centre closer than their radius sum or sharing no more electronegative
non-metal. It also drops real metal–ligand bonds that are longer than the radius
sum (the low-spin radii of Mn, Fe and Co are small) and share a cage or ring
atom more electronegative than both: the Fe–B bonds of a ferracarborane
(2.23–2.38 Å, across the cage C), Ir–B in thiaboranes (across S) and Fe–C to
η-bonded heteroarenes (across N). Disordered structures need care:
pairs closer than 0.5 Å (split positions) are never bonded, but partially
occupied neighbours are bonded as if both were present. Distances are floats
computed from the exact structure, so every exact re-description of one
crystal gives the same network except where a distance lies within float noise of its
cutoff or of its radius sum.

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
   Only the free parameters are not snapped: each orbit's free parameters
   are the exact average of its members, every member pulled back onto one point by
   every operation of the group, so no member's value is preferred. For a given
   setting and transform the recognized crystal (its expanded orbits) depends
   neither on the order of the sites nor on their hand; which orbit member's
   parameters represent it may.
   Idealising those too is available through `limit_denominator=` (applied after
   averaging), but it is opt-in, because turning `0.3333` into `1/3` is a claim
   about the data that only you can make.
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

`space_group_symmetry_operations_xyz` of an `ASUStructure` or `FundamentalDomainStructure`
lists the operations of its own setting; with the lattice translations of its cell they
generate every atom. A tabulated setting lists its own operations. An untabulated transform to
a larger cell (say a cell doubled along `a`) also multiplies them by the transform's lattice
cosets, since the conjugated operations alone reach only part of the atoms; operations equal
modulo the own lattice are listed once, so a transform to a smaller cell (an F cell's
primitive cell) lists 48 operations where its conjugates are 192. A transform that does not
keep the point group's integral form (a sheared or doubled cell) gives operations such as
`x-2*y,-y,z+1/2` or `x+1/2,y,z` with translations and coefficients the OPTIMADE
symmetry-operation grammar does not allow; they are served as they are, in the grammar of
{py:meth}`~httk.atomistic.AffineOperation.to_xyz`, because a client needs them to rebuild
`_httk_bonds`.

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

### Serving bond networks

The bond network is served as `_httk_bonds`
(`https://schemas.httk.org/defs/v0.1/properties/chemistry/bonds`, version 0.2.0): a
dictionary of `pairs`, `shapes`, `source` and `shapes_source`, or `null` without a network. It
is large, so it is not a default response field: ask for it with `response_fields`, together
with `space_group_symmetry_operations_xyz`, which it indexes.

Each bond is a row of six integers `[site, partner, op, t1, t2, t3]`: the atom at the served
position of `site` is bonded to the served position of `partner` moved by
`space_group_symmetry_operations_xyz[op]` and then by the lattice translation `(t1, t2, t3)`,
without reduction modulo the lattice. When no operations are served, `op` is 0 and means the
identity. A unit cell serves its stored coordinates, so `op` names the identity (0, or the
index of `x,y,z` in a stated list) and the translation is the bond's own. A fundamental
domain serves representative positions in its own setting, which can differ from the stored
reference atoms by a lattice translation, another orbit member, the setting transform or, for
a position stated to limited precision, up to three times that precision; each bond's
operation is corrected for its two sites and then looked up, exactly and modulo the lattice,
in the served list, which holds every operation of the group for the served cell. A client
therefore rebuilds every bond from the served positions, `space_group_symmetry_operations_xyz`
and the lattice alone. A row stands for its bond orbit on a fundamental domain and for its
lattice translates on a unit cell.

`shapes` is sparse: `{"sites": [...], "symbols": [...]}` lists only the sites with an
assigned shape, or is `null` when none is assigned. An irregular shape `I<n>` is served as the
single symbol `I`, without its count: the count is the number of bonded atoms the shape covers,
and reading `I` back gives `I<n>` with that number (a matching `I<n>` is accepted too). The
rule that a symbol's number equals the atoms covered applies to the named symbols. The definition also allows an optional
third list, `pair_rows`, for a shape over only some rows of `pairs` (and then for several
shapes on one site); httk never serves it, because a network holds one shape per site over all
of its bonds.

A network that cannot be served is served as `null` with a warning, as is a stored record
whose shapes are no longer valid; numeric structures and CIF records serve `null`.
`OptimadeStructure` reads `_httk_bonds` back for `unit_cell` and `molecular_unit_cell` spans,
checking the rules the definition states in prose (rows of six, sites and operations in range,
an identity operation, aligned shape lists, rows that touch their site, a named symbol's
number equal to the atoms its shape covers, and at least three for `I`); a violation is an
`IncompleteOptimadeResourceError`. Shapes it cannot represent, `pair_rows` over only some of a
site's rows or several shapes on one site, are dropped with a warning and the bonds are kept.
For the other spans it gives `None`.

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
