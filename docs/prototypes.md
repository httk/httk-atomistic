# Material-information levels

*httk-atomistic* describes a crystalline material at several levels of
information, organised as a matrix. The rows differ in how much *geometrical*
information is kept; the columns differ in whether the site identities are
anonymous placeholders or real chemical species. Each cell is a value family
(a backend value plus its views) in `httk.atomistic`.

| Level of geometrical information | Anonymous occupation | Assigned species |
| --- | --- | --- |
| None (site counts per class) | {py:class}`~httk.atomistic.Formulatype` | |
| Wyckoff positions only | {py:class}`~httk.atomistic.BarePrototype` | {py:class}`~httk.atomistic.BareProtostructure` |
| Wyckoff positions + covalent bonding net | {py:class}`~httk.atomistic.Protonettype` | {py:class}`~httk.atomistic.Protonetstructure` |
| Geometrical class | {py:class}`~httk.atomistic.Prototype` | {py:class}`~httk.atomistic.Protostructure` |
| Exact geometry | {py:class}`~httk.atomistic.Structuretype` | {doc}`Structure <structures>` |

The top row keeps only how many sites each species class occupies; its
assigned cell is deliberately empty. {py:class}`~httk.atomistic.ChemicalFormula`
(with the composition-based {py:class}`~httk.atomistic.AnonymousFormula`) sits
outside the matrix as the occupation-based composition: unlike `Formulatype`, it
weights each site by its occupancy (see {doc}`composition`). The bare row keeps a standard-setting
space group and its occupied Wyckoff positions. The geometrical-class row adds
an exact fundamental-domain *representative*, an externally assigned nonempty
*discriminator* string, or both. The bottom row fixes the exact continuous
degrees of freedom (cell parameters and free coordinates). Reading upward
loses geometrical information; reading right assigns real species to anonymous
placeholders.

`BarePrototype` and `BareProtostructure` cannot carry a representative or
discriminator. `Prototype` and `Protostructure` require at least one of those
refinements. Refined equality and content identity include them, so a
representative-only value never equals a discriminator-only value. Ordinary
recognition and label parsing return bare values; refinement is always explicit.

```python
from httk.atomistic import BareProtostructure, BareProtostructureView, Protostructure

bare = BareProtostructure(225, [("a", "Na"), ("b", "Cl")])
refined = Protostructure(bare.spacegroup, bare.occupations, discriminator="001")
assert BareProtostructureView(refined).unview() == bare
assert str(refined.label) == str(bare.label)
assert refined != bare
```

The representative's continuous degrees of freedom are a **class anchor, not
exact-structure data**: its coordinates and cell are retained exactly so the
anchor can be reconstructed, while the structural key stays a coarse
classification. A prototype or protostructure therefore corresponds to *many*
exact structures — every realization that shares the class — whereas a
{py:class}`~httk.atomistic.Structuretype` or a `Structure` fixes the continuous
degrees of freedom and so names a single exact geometry.
{py:class}`~httk.atomistic.Prototype` uses a
{py:class}`~httk.atomistic.FundamentalDomainTemplate` representative and
anonymous {py:class}`~httk.atomistic.PrototypeOccupation` values;
{py:class}`~httk.atomistic.Protostructure` uses a
{py:class}`~httk.atomistic.FundamentalDomainStructure` representative and
real-species {py:class}`~httk.atomistic.WyckoffOccupation` values.

## How the levels relate

The levels combine by adding one piece of information at a time. These are
information-content relationships, not class inheritance:

- BarePrototype + species assignment → BareProtostructure
- BarePrototype + representative/discriminator → Prototype
- BareProtostructure + representative/discriminator → Protostructure
- Prototype + species assignment → Protostructure
- Structuretype + species assignment → Structure
- Projection onto class site counts → Formulatype
- Projection onto composition → ChemicalFormula or AnonymousFormula

A bare view of a refined value retains its source, so rewrapping can recover
existing refinement. Calling `unview()` explicitly materializes a standalone
bare value; its discarded representative cannot be recovered. Refined views do
not infer a class from bare values, raw structures, or label strings. Construct
`Protostructure(representative=asu)` or `Prototype(representative=template)` to
choose an exact class anchor explicitly.

## Naming and capitalization

The canonical taxonomy terms are single-capital compound words: `Formulatype`,
`Prototype`, `Protostructure`, and `Structuretype`; the broader Wyckoff-only
families are `BarePrototype` and `BareProtostructure`. The suffix `-type` marks the
anonymous-occupation column (`Structuretype` and `Prototype` are the anonymous
counterparts of `Structure` and `Protostructure`; `Formulatype` has no assigned
counterpart); the `Bare` prefix distinguishes cutoff-free Wyckoff classification from a
refined geometrical class.

The word "Template" no longer names the exact anonymous family — that family is
`Structuretype`. "Template" survives only for the exact *fundamental-domain*
anonymous values used as class anchors and ASU keys:
{py:class}`~httk.atomistic.FundamentalDomainTemplate` and
{py:class}`~httk.atomistic.ASUTemplate` (and their view family, e.g.
{py:class}`~httk.atomistic.FundamentalDomainTemplateView`). The name `Prototype`
is the already-established term for what a fully systematic naming scheme would
call a "templatetype"; there is no code alias for the latter (see
{doc}`details/structural_classes`).

The older names survive as aliases for discoverability only; documentation and
new code use the canonical names. `AnonymousFormula` and `AnonymousFormulaView`
are no longer aliases: they name the composition-based anonymous formula, while
`Formulatype` counts sites.

| Alias | Canonical |
| --- | --- |
| `AnonymousStructure`, `AnonymousStructureView`, `AnonymousStructureLike` | `Structuretype` family |

## Dummy species

The anonymous exact values ({py:class}`~httk.atomistic.Structuretype`,
{py:class}`~httk.atomistic.FundamentalDomainTemplate`, and the representative
held by a {py:class}`~httk.atomistic.Prototype`) use a deliberately narrow
dummy-species shape. The label is carried through the `labels` decoration and
the species name; it is never encoded as a chemical symbol:

```python
from httk.atomistic import Species
from httk.atomistic.models.structuretype.anonymize import dummy_species, is_dummy_species

species = dummy_species("A")
assert species == Species("A", ("X",), (1,), labels=("A",))
assert is_dummy_species(species)
```

`is_dummy_species` requires exactly one `"X"` chemical symbol, unit
concentration, matching name/label, and no mass, attachments, charge, spin,
original name, or concentration decoration. Consequently a species named `A`
with `labels=("other",)` is not a dummy species. A `BarePrototype` carries anonymous class labels (`A`, `B`, `C`, ...)
directly, without dummy `Species` objects. A `Prototype` also uses these labels
for its occupations; its optional representative contains dummy species.

## What crosses the boundary

The conversion boundary is intentionally explicit. The following table lists
features rejected during conversion and features deliberately erased when a
conversion is otherwise valid.

| Conversion boundary | Rejected | Deliberately erased |
| --- | --- | --- |
| Structure → `Structuretype`/`FundamentalDomainTemplate` | disorder or partial occupancy; duplicate- or multi-element species; a species whose symbol is `"X"` or `"vacancy"`; assemblies; `chemical_composition`; site moments | species identities become dummy labels; charge, spin, mass, formula metadata, `optimization_type`, `immutable_id`, `last_modified`, and molecular metadata |
| Structure → `BareProtostructure` | assemblies; molecular structures; `chemical_composition`; site moments; a species containing `"X"` (including attached `"X"`) | charge/formula metadata, `optimization_type`, `immutable_id`, `last_modified`, and molecular metadata |

`BareProtostructure` is different here: its `Species` objects retain disorder and
partial occupancy, including their real chemical symbols, concentrations, and
decorations. Those are not rejected merely because they are non-singleton.

The useful conversion matrix is below. Views either present an existing value
(exact, pass-through), extract a coarser level from a finer one (exact), or
recognize a level from a plain structure (tolerant, needs the
symmetry-recognition path, spglib).

| Construction | Result | Boundary |
| --- | --- | --- |
| `PrototypeView(Prototype)` | prototype view | exact/pass-through |
| `PrototypeView(Protostructure)` | anonymous prototype (species erased; any explicit representative/discriminator carried over) | exact erasure of species |
| `BarePrototypeView(FundamentalDomainTemplate)` | bare prototype | exact extraction/discretization |
| `BarePrototypeView(Structuretype)` | bare prototype recognized from the exact anonymous geometry | tolerant recognition (spglib) |
| `BarePrototypeView(ASUStructureView(s, setting=...))` | standard-setting bare prototype | exact ASU path after the requested setting is chosen |
| `BarePrototypeView(unitcell or ordinary structure)` | recognized bare prototype | tolerant recognition (spglib) |
| `StructuretypeView(Structuretype)` | structuretype view | exact/pass-through |
| `StructuretypeView(FundamentalDomainTemplate)` | expanded unit cell | exact |
| `StructuretypeView(structure)` | anonymized projection | exact; validates the rejection rules above |
| `ProtostructureView(Protostructure)` | protostructure view | exact/pass-through |
| `BareProtostructureView(ASUStructureView(s, setting=...))` | geometry-free real-species key | exact ASU path |
| `BareProtostructureView(unitcell or ordinary structure)` | recognized bare protostructure | tolerant recognition (spglib) |
| `BareProtostructureView(Structuretype or FundamentalDomainTemplate)` | — | raises: dummy species are not real species |
| `UnitcellStructureView(Structuretype or FundamentalDomainTemplate)` | — | raises: dummy species are not real species |

Recognition from a plain structure is the tolerant/spglib boundary. Existing
ASU, structuretype, prototype, and protostructure values use exact data, with no
recognition tolerance. For a source that needs a particular setting, use the
sanctioned idiom shown above:
`BarePrototypeView(ASUStructureView(s, setting=...))`. Recognition of a raw
structure resolves the standard setting.

## Formula conveniences

Every structural family below the top row exposes `anonymous_formula`, the
`Formulatype` of its class site counts, which is also the prefix of its label;
the assigned families additionally expose `formula`, their composition. A
`Formulatype` has no `anonymous_formula` attribute — it *is* the anonymous
formula, rendered as its string value (`str(FormulatypeView(...))`, e.g.
`"A3B2"`). Wyckoff-family counts use conventional Wyckoff multiplicities, and
rendering removes a common GCD. The composition-based anonymous formula is
`AnonymousFormulaView(value)`:

```python
from httk.atomistic import Structuretype, StructuretypeView

template = Structuretype(
    [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
    [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
    species_at_sites=("A", "B"),
)
view = StructuretypeView(template)
assert view.anonymous_formula == "AB"
assert view.unwrap() is template
```

`BareProtostructure` is the Wyckoff-with-species key: it has no cell or
coordinates. Equality uses its standard-setting space group and occupied
Wyckoff positions with associated `Species` values, independent of construction
order. `BarePrototype` is its anonymous counterpart. Both are hashable.
Refined values additionally compare their representative and discriminator;
values differing only in representative may hash alike but remain unequal.

Once their input views have been materialized, the four canonicalizers operate
at the classification level:

```python
from httk.atomistic import (
    canonical_bare_prototype,
    canonical_bare_protostructure,
    canonical_prototype,
    canonical_protostructure,
)

bare_assigned = canonical_bare_protostructure(bare_assigned_source)
bare_anonymous = canonical_bare_prototype(bare_anonymous_source)
refined_assigned = canonical_protostructure(refined_assigned_source)
refined_anonymous = canonical_prototype(refined_anonymous_source)

assert canonical_bare_protostructure(bare_assigned) == bare_assigned
assert canonical_bare_prototype(bare_anonymous) == bare_anonymous
assert canonical_protostructure(refined_assigned) == refined_assigned
assert canonical_prototype(refined_anonymous) == refined_anonymous
```

View materialization preserves the conversion boundaries in the table above, so
a raw ordinary structure accepted by a bare view can first require tolerant
spglib recognition. The bare minimization itself uses no geometry and enumerates
finite exact letter-changing actions. The refined operations canonicalize an
exact retained representative when one exists; with only a discriminator they
canonicalize the bare base and retain that discriminator. A refined geometry
terminal can demote special positions or collapse an exact supercell, operations
for which a bare value has no geometry. Canonicalize the projected value when a
canonical bare result is required; projection and canonicalization do not
generally commute.
`FundamentalDomainTemplate.bare_prototype` extracts the discrete anonymous key.
The full procedure and retained-coordinate policy are specified in
{doc}`canonicalization`.

## Labels

An httk label is a compact string encoding the information content of an
unsuffixed AFLOW-style prototype label: a space group, its occupied Wyckoff
letters, and the partition of those occupations into species classes. The
single home of the notation is
`httk.atomistic.models.prototype.notation`. The grammar is:

```
ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*                 # prototype label
ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*:NAME(-NAME)*    # protostructure label
```

A `GROUP` is the concatenation of one class's Wyckoff letters, sorted
alphabetically, a letter occupied `k >= 2` times prefixed by the integer `k`
(`2e`); count `1` is omitted. `ANON` is the anonymous formula (`A`, `B`, `C`,
...) built in group order with per-group summed conventional multiplicities
reduced by their overall GCD, so it is the {py:class}`~httk.atomistic.Formulatype`
of the class site counts.

The prefix is also every level's `anonymous_formula`, including for disordered or
mixed-species occupations: a `BareProtostructure` with an equal Fe/Ni mixture on
`225` `a` and O on `b` labels as `AB_cF8_225_a_b:FeNi-O` and its
`anonymous_formula` is `AB`; only its composition-based `AnonymousFormulaView` is
`A2BC`. Erasing such a bare value still gives a `BarePrototype`
(`AB_cF8_225_a_b`), but a `BarePrototype` cannot be derived from a structure
carrying such species; structure anonymization requires single-element species.

Known limit: a structure's classes are its species names, as are those of its
`BareProtostructureView` and that view's `BarePrototype` erasure, but
`Structuretype` and direct `BarePrototype` recognition (structure →
`Structuretype` template → `BarePrototype`) merge species by element. Where
distinct species names share one element-pure chemistry, those two therefore give
a different `Formulatype` from the structure's own and from its
`BareProtostructure` erasure. COD 1001844 has two oxygen species, `O1` and `O2`:
`BarePrototypeView(structure)` labels as `A12B3C_aP32_1_24a_6a_2a`, while the
structure's `Formulatype` and `BarePrototypeView(BareProtostructureView(structure))`
give `A24B6CD` (`A24B6CD_aP32_1_24a_6a_a_a`).

A structure's canonicalization preserves chirality by default. Pass
`preserve_chirality=False` to the corresponding classification canonicalizer
when a canonical `BareProtostructure` or `BarePrototype` label should identify
the two members of an enantiomorphic pair. The same policy is available through
`canonical_asu(preserve_chirality=False)` and `normalize_chirality`; see
{doc}`asu`.
The structure methods `canonical_bare_protostructure()` and
`canonical_bare_prototype()` retain their chirality-normalized convention: they
recognize and canonicalize the structure, then canonicalize its bare projection.

### httk labels are not AFLOW labels

The **httk group-ordering convention** follows `Formulatype`: the class with
the largest summed conventional Wyckoff multiplicity (its site count, not an
occupancy-weighted amount) comes first; classes with equal counts are ordered
lexicographically by their sorted Wyckoff-letter sequences; a remaining tie is
broken by the species name in a protostructure label. The anonymous symbols
`A`, `B`, ... are assigned in the same order, and the canonicalizers use the
same key to choose among equivalent descriptions, so a canonical value and its
label agree. This ordering is *element-agnostic*, so a protostructure label is
exactly its erased prototype label plus the `:` species suffix. AFLOW, by
contrast, orders the classes by element symbol **alphabetically**, so its
unsuffixed prefix depends on the chemistry. The two are therefore genuinely different strings, and an httk label
is **not** an AFLOW label.

The assigned-species classes expose both, as distinct properties: `label` (the
httk convention, a {py:class}`~httk.atomistic.BareProtostructureLabel` for a bare value) and
`aflow_label` (the AFLOW-style rendering, a plain `str`). For calcite,
`BareProtostructure(167, a:Ca, b:C, e:O)`:

```python
from httk.atomistic import BareProtostructure, Species

Ca, C, O = Species("Ca", ("Ca",), (1,)), Species("C", ("C",), (1,)), Species("O", ("O",), (1,))
calcite = BareProtostructure(167, [("a", Ca), ("b", C), ("e", O)])
assert calcite.label == "A3BC_hR10_167_e_a_b:O-Ca-C"
assert calcite.aflow_label == "ABC3_hR10_167_b_a_e:C-Ca-O"
```

The httk label puts O on `e` (18 sites) first; Ca on `a` and C on `b` tie
with 6 sites each and follow by Wyckoff letter. The AFLOW label orders the
groups `b`, `a`, `e` to follow the alphabetical elements `C`, `Ca`, `O`.

### Pearson symbol

The Pearson symbol is `system + centring + count`. The system letter follows
the space group's crystal system (`a`, `m`, `o`, `t`, `h`, `h`, `c` for
triclinic through cubic, trigonal and hexagonal both mapping to `h`). The
centring letter follows the centring type, with the base-centred variants
`A`, `B`, `C`, and `S` folded to `C` (the `A` case fires for groups 38–41). The
count is the conventional-cell site count, except a rhombohedral `R` setting —
tabulated on hexagonal axes — divides it by three (and asserts divisibility).
Calcite's 30 conventional sites give `hR10`.

The 27th Wyckoff letter used by a few high-multiplicity settings (group 47's
eightfold orbit, internally `'α'`) renders as `A` and parses back from it;
positionally a group token never collides with the leading anonymous formula.

### Strict parser

The parser is strict and canonical-only: it resolves the standard setting,
validates every Wyckoff letter, recomputes the Pearson symbol, the reduced
anonymous counts, and the group ordering, and rejects any string that deviates
from the recomputed canonical form. Suffix names must be known element symbols
and become `Species(name, (name,), (1,))`. Round trips are pinned in both
directions: `parse(render(x)) == x` for element-pure bare values and
`render(parse(s)) == s` for canonical strings. This mirrors
`parse_anonymous_formula` for `Formulatype`. Use
`parse_bare_prototype_label` and `parse_bare_protostructure_label` from the
notation module, or the corresponding bare views. Parsing refined label text
returns only its bare classification; the text cannot encode its refinement.

### Canonical vs plain labels

Any faithful render of an object is *the* prototype or protostructure label.
The *canonical* prototype or protostructure label is the one obtained from a
normalizer-canonical object produced by the corresponding
`canonical_bare_*`/`canonical_*` function. A value derived from `canonical_asu`
must first be projected to the required family and then passed through that
family canonicalizer because projection and canonicalization do not generally
commute. The renderer itself performs no affine-normalizer pass, so labels from
hand-built, non-canonical objects are faithful but not necessarily canonical.
Whenever text speaks of the label of an arbitrary value it uses the plain form ("the
protostructure label"), reserving "the canonical … label" for a
normalizer-canonical source.

The AFLOW-style `-001` discriminators belong to a `Prototype` or a
`Protostructure` (their `discriminator` field), which name a species-independent
geometrical class. They are **never** part of the label.

### Bonding-net labels

{py:class}`~httk.atomistic.Protonettype` and
{py:class}`~httk.atomistic.Protonetstructure` extend a bare prototype or bare
protostructure with a canonical descriptor of its symmetric covalent bonding
net. The label is the bare label, `__`, the net, and for the assigned level the
usual species suffix:

```
label      = bare_label "__" net [ ":" name { "-" name } ]
net        = group { "_" group }                  (may be empty)
group      = site "~" target { "." target }
target     = site [ "@" op [ "@" klm ] ]
klm        = digit digit digit                    (each digit = translation + 5)
```

The anonymous levels and the protonetstructure nest as string prefixes; the
canonical bare protostructure label is not always one (see
[the species-order rule](#species-order-of-protonetstructure-labels)).
For K₂AgSbS₄ (COD 1509425), where no classes tie:

```
A4B2CD                                                                   Formulatype
A4B2CD_oP32_34_4c_abc_ab_c                                               BarePrototype
A4B2CD_oP32_34_4c_abc_ab_c__1~8.10_2~8.10_3~9.10_4~9.10@3                Protonettype
A4B2CD_oP32_34_4c_abc_ab_c:S-K-Ag-Sb                                     BareProtostructure
A4B2CD_oP32_34_4c_abc_ab_c__1~8.10_2~8.10_3~9.10_4~9.10@3:S-K-Ag-Sb      Protonetstructure
```

**What the parts are.** Everything after `__` describes which atoms are
covalently bonded. The numbers are the symmetry-distinct sites of the
prototype, counted in the order the label lists them: here 1-4 are the four S
sites on `4c`, 5-7 are K on `2a`, `2b` and `4c`, 8-9 are Ag on `2a` and `2b`,
and 10 is Sb on `4c`. Each number stands for one reference atom of that site.
Groups are separated by `_`, and each group lists the bonds that start at its
first site. `~` means "is covalently bonded to" (bond order unspecified, dative
bonds counted as covalent), and `.` separates the destinations. So `1~8.10`
says that site 1 is bonded to site 8 and to site 10; it is a list, not a
chain. A destination may carry `@n`, meaning the bond goes to the copy of that
site's reference atom produced by symmetry operation number `n` of the space
group, not to the reference atom itself. A second `@` adds a lattice
translation as three digits, each being the shift along a, b, c plus 5, so
`10@3@545` is operation 3 followed by one cell back along b. No `@` means the
reference atom itself, and a zero translation is never written. A reference
atom is the first-listed Wyckoff coordinate at the site's free parameters, not
reduced modulo one.

**How to read it.** Each entry is one symmetry-distinct bond, and all its
symmetry copies are implied. The example therefore reads: S1 bonds to Ag8 and
Sb10; S2 likewise; S3 bonds to Ag9 and Sb10; S4 bonds to Ag9 and to the image
of Sb10 under operation 3 (`-x+1/2,y+1/2,z+1/2`). Those eight entries generate
all 32 Ag-S and Sb-S bonds in the unit cell. Sites that appear nowhere, here the
three K sites, have no covalent bonds. The net records covalent bonds only: a
missing row means "no covalent bond", not "no contact". Which atom pairs are
bonded is decided by the caller, who supplies the bond list; *httk₂* does not
infer it. The string is canonical: which symmetry copy serves as each site's
reference atom, and how interchangeable sites such as the four S are numbered,
is chosen so that the string comes out smallest. The intent is that two
crystals with the same prototype and the same bonding network get the same
string, whatever their coordinates or choice of origin.

**Relation to CIF.** This is the CIF bond list in compressed form. CIF records
a bond as two atom labels plus a symmetry code for each
(`_geom_bond_site_symmetry_1` and `_2`), written `n_klm`: operation number `n`,
then the three translation digits offset by 5, with `.` for "no operation". The
`10@3@545` above is CIF's `3_545` applied to the second atom, and the first
atom's code is always `.`. There are four differences. CIF's operation number
refers to the symmetry list inside that particular file, whereas here it refers
to a fixed list for the space group's standard setting, with the identity as
number 1. CIF's atoms are whichever ones the file happens to list, with
coordinates, whereas here they are the reference atoms fixed by the canonical
rule. CIF may list every bond around every atom, with distances, whereas here
each symmetry-distinct bond appears exactly once and no geometry is stored.
Finally, the redundant `_555` is dropped.

**Strictness.** As for the other labels the parser is canonical-only
(`parse_protonettype_label` and `parse_protonetstructure_label` in the
notation module): it builds the value, re-renders it and rejects any string that
differs, so `1~8@1`, `1~8@3@555`, `1~8.`, `1~8@3@55` and unsorted or repeated
groups all fail. The following also hold.

- `__` splits the bare label from the net at its first occurrence; neither part
  can contain `__`, since no field or group is empty, and a second `__` is
  rejected. The species suffix is split off at the last `:`.
- The characters `-`, `=`, `#`, `$`, `:`, `/` and `\` are reserved inside the net
  for future bond types and are rejected.
- A translation must lie in -5..+4 per axis, the range one digit can write.
  Input rows and bonds take any integer translation; only a canonical net that
  still needs a longer one raises `ValueError` (D4).
- A well-formed but non-canonical net, such as the same bonds against other
  reference atoms or another numbering of interchangeable sites, is rejected:
  the parser's value is canonical, so its label differs from the input.
- An operation is written whenever a translation is, also operation 1
  (`10@1@655`); operation 1 with zero translation is not written at all.
- Operation `n` (1-based) is the `n`-th branch of the general Wyckoff position of
  the standard setting, with the identity as 1. This is not always the order of
  the International Tables: for group 34 the numbering is 1 `x,y,z`, 2 `-x,-y,z`,
  3 `-x+1/2,y+1/2,z+1/2`, 4 `x+1/2,-y+1/2,z+1/2`, so 3 and 4 are exchanged. It
  is also not the order of `Spacegroup.symmetry_operations`, which for group 34
  lists the identity last.
- A pair of sites may repeat (`1~7.7@4`, two different bond orbits between the
  same two sites), and `i~i@...` is a bond between symmetry copies of one site
  (diamond: `1~1@14@654`).
- An empty net is written with a trailing `__`: `AB_cF8_225_a_b__`. Sites
  without bonds do not appear in the net but keep their number.

The example below builds K₂AgSbS₄ from an exact asymmetric unit and an explicit
bond list: the eight Ag-S and Sb-S bonds that a 3.0 angstrom cutoff finds,
written `(i, j, k, n1, n2, n3)` against the input site order. The caller
supplies the bonds; `from_asu` renumbers them into site order and returns the
constructor's value, with the net canonicalized. Bonds may carry any integer
translation, so the free parameters need not be reduced into [0, 1). The
asymmetric unit is the vendored COD entry; the label is pinned by
`tests/test_protonet_canonical_api.py`.

```python
from fractions import Fraction as F

from httk.atomistic import (
    ASUStructure,
    Cell,
    Protonetstructure,
    Species,
    WyckoffSite,
    canonical_protonetstructure,
)
from httk.core import FracVector

# K2AgSbS4 (COD 1509425, Pnn2, no. 34) as an exact asymmetric unit, sites in input order
rows = [
    ("S", "c", (F(444, 625), F(2759, 5000), F(3161, 5000))),
    ("S", "c", (F(7201, 10000), F(2307, 2500), F(367, 625))),
    ("S", "c", (F(4619, 5000), F(901, 1250), F(3037, 10000))),
    ("Ag", "a", (F(9519, 10000),)),
    ("K", "c", (F(617, 2500), F(2349, 10000), F(237, 250))),
    ("S", "c", (F(11, 20), F(7051, 10000), F(167, 625))),
    ("K", "a", (F(56, 125),)),
    ("Ag", "b", (F(2249, 10000),)),
    ("K", "b", (F(379, 500),)),
    ("Sb", "c", (F(909, 1250), F(3627, 5000), F(897, 2000))),
]
species = [Species(name, (name,), (1,)) for name in ("S", "Ag", "K", "Sb")]
cell = Cell([[F(10348, 1000), 0, 0], [0, F(10522, 1000), 0], [0, 0, F(7946, 1000)]])
asu = ASUStructure(
    cell, 34, [WyckoffSite(letter, FracVector(params), name) for name, letter, params in rows], species
)

# Ag-S and Sb-S bonds, found by the caller: (i, j, operation, n1, n2, n3)
bonds = [
    (1, 4, 3, 0, 0, -1), (1, 10, 1, 0, 0, 0), (2, 8, 3, 0, 0, 0), (2, 10, 1, 0, 0, 0),
    (3, 8, 1, 1, 0, 0), (3, 10, 1, 0, 0, 0), (4, 6, 3, 0, -1, 0), (6, 10, 1, 0, 0, 0),
]
value = Protonetstructure.from_asu(asu, bonds)
canonical = canonical_protonetstructure(value)
assert canonical.label == "A4B2CD_oP32_34_4c_abc_ab_c__1~8.10_2~8.10_3~9.10_4~9.10@3:S-K-Ag-Sb"
```

### Canonical protonet labels

One net is one value. The constructors `Protonettype(...)` and
`Protonetstructure(...)`, and `from_asu`, keep the *anonymous* bare description
(no affine-normalizer pass, as for `BarePrototype(...)`) and canonicalize the
net over site numbering, reference atoms and the equivalent descriptions with
the same anonymous bare key, orientation-preserving, with the default timeout.
Two spellings of one net therefore give one equal value, and for a canonical
bare part the constructor's value is the canonical one.

At the assigned level the D3 tie-break may move species between the Wyckoff
letters of those equivalent descriptions:
`Protonetstructure(225, [("a", Na), ("b", Cl)])` gives
`AB_cF8_225_a_b__:Cl-Na`. This differs from `BareProtostructure(...)`, which
keeps any assigned description as given.

`canonical_protonettype` and `canonical_protonetstructure` return the canonical
value for any protonet-like input. Both take `preserve_chirality` (as the bare
canonicalizers do) and `timeout`, the cooperative seconds allowed for the search
(default 120, `None` disables it); a search that exceeds it raises
`CanonicalizationLimitError`. The net is chosen before the species. A canonical
`Protonetstructure` is therefore canonicalized anonymously first, and species
names only break ties between descriptions that give the identical net. When
classes tie in site count and letters, its species-to-letter assignment, and so
its label without the `__net` part, can differ from that of
{py:func}`~httk.atomistic.canonical_bare_protostructure`; no equality between the
two is promised. Erasing a `Protonetstructure` to a `Protonettype` keeps class
positions: class `c` becomes `anonymous_symbol(c)`. In the monoclinic groups and
in P-1 the equivalent descriptions include the changes between valid cells, so
labels, and nets that are not periodic in any direction (molecular crystals), do
not depend on the cell. Monoclinic and P-1 nets with any periodic component,
including monoclinic nets periodic only along b, are canonical only relative to the
cell that httk's standardisation chooses. P1 is excluded from the guarantee: it has no special positions, so its
labels are unaffected, but its periodic nets stay canonical only relative to the
cell (D7). See {doc}`canonicalization` for the algorithm and its limits.

### Species order of protonetstructure labels

A canonical `Protonetstructure` label is the canonical `Protonettype` label,
`:`, and the species names, which fill the classes in the class order of the
canonical net. Where several descriptions give the same net, the smallest tuple
of names is taken. The bare protostructure label instead orders tied classes by
name alone. So `BarePrototype` → `Protonettype` → `Protonetstructure` nest as
string prefixes, but `BarePrototype` → `BareProtostructure` →
`Protonetstructure` do not always: cutting `__net` from a protonetstructure
label gives a correct description of its bare protostructure, but not always
the canonical bare protostructure label. The two orders differ when exchanging
two species leaves the anonymous prototype label unchanged and the net tells
them apart against alphabetical order. That happens when

- two classes tie in site count and Wyckoff letters, or
- an equivalent description exchanges their positions, e.g. two inversion
  centres of one Wyckoff set.

Examples in P2₁/c (no. 14), where operation 2 is the inversion `-x,-y,-z`:

| Structure | Protonetstructure label | Canonical bare protostructure label |
| --------- | ----------------------- | ----------------------------------- |
| N on `e` bonded to its own inversion image and to H on `e` | `AB_mP8_14_e_e__1~1@2.2:N-H` | `AB_mP8_14_e_e:H-N` |
| C on `e`, the same net | `AB_mP8_14_e_e__1~1@2.2:C-H` | `AB_mP8_14_e_e:C-H` |
| N and H on `e` joined by two bonds, neither self-bonded | `AB_mP8_14_e_e__1~2.2@2:H-N` | `AB_mP8_14_e_e:H-N` |
| O on `e`, Cu on `a`, Zn on `b`, O bonded to Cu | `A2BC_mP8_14_e_a_b__1~2:O-Cu-Zn` | `A2BC_mP8_14_e_a_b:O-Cu-Zn` |
| The same sites, O bonded to Zn | `A2BC_mP8_14_e_a_b__1~2:O-Zn-Cu` | `A2BC_mP8_14_e_a_b:O-Cu-Zn` |

The first two rows share the protonettype `AB_mP8_14_e_e__1~1@2.2`: the net puts
the self-bonded class first, whichever name it has. In the last row the two
inversion centres `a` and `b` are exchanged by an origin shift, and the net puts
the centre bonded to O on `a`.

Which projections are string operations:

- protonetstructure → protonettype: cut the `:` suffix;
- protonettype → bare prototype: cut `__net`;
- protonetstructure → bare protostructure: **not** a string operation.
  `BareProtostructureView` of a `Protonetstructure`, and its
  `bare_protostructure` property, re-canonicalize its occupations, so both equal {py:func}`~httk.atomistic.canonical_bare_protostructure`
  of the same value.

## Similarity and atom travel

`Prototype.similar` and `Protostructure.similar` first compare their discrete
space-group and occupation keys, then apply discriminator compatibility. If
both values have representatives, the continuous comparison is the total
Cartesian atom travel returned by the public `structure_delta(first, second)`;
missing representatives do not invent a distance. This method belongs to
refined families; compare bare keys with equality. `structure_delta` maps the
structures into a common subgroup and setting, pairs compatible Wyckoff
orbits, and sums the shortest periodic Cartesian travel of their atoms. Each
endpoint uses its own cell, so lattice changes contribute through the atom
positions. It is not a content-id or label comparison. For compatible discrete keys and discriminators, geometrical comparison returns
`False` when travel exceeds the budget or no common representation exists (`NoCommonRepresentation`, a
`ValueError` subclass in `httk.atomistic.symmetry.paths`); other errors from a
broken representative — a singular cell basis, a non-three-dimensional cell, or
non-finite travel — propagate.

For approximate clustering, install *httk-atomistic* with its `numpy` extra and pass
`use_numpy=True` to either `similar(other, delta, use_numpy=True)` or
`structure_delta(first, second, use_numpy=True)`. This converts each endpoint's
expanded coordinates once to temporary NumPy float64 arrays and computes orbit
distance matrices with vectorized arithmetic. Storage for coordinates is linear in
the expanded atom count; temporary distance matrices are limited to one orbit pair.
The periodic-image search still handles skew cells, and atom/orbit assignment keeps
the same minimum-cost matching. Discrete symmetry and canonicalization remain exact.

This option permits rounding differences in ties and near a comparison threshold,
including variation across floating-point platforms. It also gives up exact
cancellation before coordinate subtraction. The default comparison remains available.
Neither mode modifies representatives, stored records, or content identity.

With `use_numpy=True`, `similar` uses conservative lower bounds to stop scoring an
alignment that cannot meet `delta`. Rejection still considers the other allowed
alignments; acceptance verifies travel in the returned setting. Roundoff allowances
make rejection conservative without increasing the requested budget.
`structure_delta` continues to compute the complete distance.

For repeated comparisons, pass the same
`httk.atomistic.symmetry.comparison_cache.StructureComparisonCache` as `cache=` to
`similar` or `structure_delta`. Preparation is lazy: the cache retains exact
canonical structures and, with `use_numpy=True`, temporary float64 orbit arrays.
Use one cache per clustering group and release it afterward, or call `clear()`.
The `max_structures` and `max_geometries` capacities bound retained entries; eviction
only repeats preparation and does not change comparison results. Different
rerepresentation tolerances have separate canonicalization entries.

## Storage records

The families have durable, layout-independent storage records in
`httk.atomistic.storage.records`:

| Record | Storage name | Value |
| --- | --- | --- |
| `BarePrototypeRecord` | `atomistic_bare_prototype` | `BarePrototype` |
| `BareProtostructureRecord` | `atomistic_bare_protostructure` | `BareProtostructure` |
| `PrototypeRecord` | `atomistic_prototype` | `Prototype` |
| `ProtostructureRecord` | `atomistic_protostructure` | `Protostructure` |
| `FundamentalDomainTemplateRecord` | `atomistic_fundamental_domain_template` | `FundamentalDomainTemplate` |
| `FundamentalDomainStructureRecord` | `atomistic_fundamental_domain_structure` | `FundamentalDomainStructure` |

Each record carries the value identity of its family, so two equal values
produce records with the same content id (the deduplication key) and unequal
values differ. Bare records store only discrete classification.
`PrototypeRecord` and `ProtostructureRecord` require a representative or
discriminator, and store the optional representative as a nested record
(`FundamentalDomainTemplateRecord` for a prototype,
`FundamentalDomainStructureRecord` for a protostructure) and the optional
discriminator as a plain column. `Structuretype` itself stays non-storable.

All four classification records render the **httk label** (for example `AB_cF8_225_a_b` and `AB_cF8_225_a_b:Na-Cl`) as a queryable
`label` column. The content ids are unchanged by this — the label is a
convenience and query column, not the record's identity, and it is not unique:
the discriminator is not part of the label, so records that share occupations
but differ in class collide on it, and two protostructures whose species share a
name but differ in another `Species` field also collide. Count and deduplicate
by row (content id), never by label.

The bare registry records are `atomistic-bare-prototype` (family
`bare_prototypes`) and `atomistic-bare-protostructure` (family
`bare_protostructures`). The refined records are `atomistic-prototype` (family `prototypes`) and
`atomistic-protostructure` (family `protostructures`), with
`atomistic-fundamental-domain-structure` in the `structures` family.
`FundamentalDomainTemplateRecord` is an embedded component record (nested inside
`PrototypeRecord` as the optional representative) and deliberately has no
registry entry of its own.

A store can retain one bare parent and multiple refined classes sharing its
Wyckoff label. The COD canonicalization pass writes bare entries; discrimination
writes refined entries and their bare parents to the final database, linked by
bare content ID. These classification families need no OPTIMADE serving
definitions to be saved and queried through storage APIs.

Development databases using the previous taxonomy must be rebuilt. No legacy
registry aliases or upgrade machinery are provided.

The six records with a canonical label (`BarePrototypeRecord`,
`BareProtostructureRecord`, `PrototypeRecord`, `ProtostructureRecord`,
`ProtonettypeRecord` and `ProtonetstructureRecord`) store a `canonical_form`
integer, the version of the canonical rules they were built with. It is part of
their content identity. There are three versions, one per family:

| Family | Records | Bumped when |
|---|---|---|
| label | `BarePrototypeRecord`, `BareProtostructureRecord` | the label rules change, `Formulatype` order included |
| refined | `PrototypeRecord`, `ProtostructureRecord` | the label version is bumped, or the representative rule changes |
| net | `ProtonettypeRecord`, `ProtonetstructureRecord` | the label version is bumped, or the net rule changes |

A label-rule change therefore bumps all three, while a change to the
representative or to the net leaves the other families' records untouched. Each
version is now 2, since the monoclinic and P-1 labels became independent of the
cell. Saving requires the family's current version, and turning a record of
another version back into a value raises `ValueError` naming the family: its
canonical form changed, so rebuild the database. Saving a protonet record also checks that its net
is the canonical one for its bare description; a non-canonical net is rejected,
never rewritten. Constructing a protonet value, parsing its label and saving its
record (`__httk_validate__`) each run the net search; loading a value from a
record does not.

## Deferred features

The following are deliberately not part of this round and not part of the
conversion contracts above:

- `same_prototype()`.
- OPTIMADE serving (definitions, providers, and bindings) for the prototype and
  protostructure families.
- The species-assignment convenience constructors
  (`Protostructure(prototype, species=...)`, `Structure(structuretype, species=...)`).
- Automatic normalizer canonicalization inside label rendering. Call the
  classification canonicalizer explicitly before rendering when canonical text
  is required.

The full guide, {doc}`details/structural_classes`, covers the naming rationale
and how the classes relate to isopointal/isoconfigurational structures and
AFLOW labels.

## Grid candidate filtering

For a batch of geometry-carrying prototypes, protostructures, or exact structures
in the same declared space group, use
`httk.atomistic.symmetry.comparison_grid.StructureComparisonGrid` to avoid
comparisons that cannot meet a Cartesian travel threshold:

```python
from httk.atomistic.symmetry.comparison_cache import StructureComparisonCache
from httk.atomistic.symmetry.comparison_grid import StructureComparisonGrid

cache = StructureComparisonCache(max_structures=max(1, 2 * len(values)))
grid = StructureComparisonGrid(values, delta, dimensions=2, strategy="variance", cache=cache)
# Within the existing clustering loop:
if grid.might_match(i, j):
    matches = values[i].similar(values[j], delta, use_numpy=True, cache=cache)
```

The grid indexes one, two, or three Cartesian projections of expanded Wyckoff
coordinates. It includes normalizer alternatives, periodic images, and repeated
orbit members. `first` selects axes in order, `variance` prefers axes with larger
coordinate variance, and `occupancy` prefers axes with more occupied scalar bins.
These are selection strategies, not changes to the matching threshold. A grid
candidate still needs the ordinary comparison; an excluded pair needs neither
an assignment solve nor alignment scoring.

The necessary neighborhood radius follows the endpoint-cell travel metric:
if an atom contributes at most `delta`, a candidate point must be within
`sqrt(2) * delta` of a periodic image in the reference cell. Projecting that bound
onto fewer coordinates admits extra candidates without removing valid ones.
Numerical padding makes the filter conservative near float boundaries. Both
alignment directions remain possible, and the retained exact representatives
are unaffected.

Sparse buckets and a bounded query cache limit retained index data. `max_points`
and `max_images` cap extra preparation and periodic-image work. Unsupported
preparations, missing geometry, mixed groups, a zero threshold, or poorly
conditioned cells fall back to unfiltered comparison; `fallback_reason` reports
a group-wide fallback. `selected_axes` and `indexed_points` expose index diagnostics.
Use the same shared cache for index construction and subsequent comparisons so
canonicalization is reused. Index preparation may cost more than it saves for
small or densely matching groups; benchmark complete groups when selecting settings.


## Unified canonicalization

`canonicalize(value)` dispatches explicitly on the four classification backend
families (including their views) and returns the matching standalone value.
It is equivalent to `canonical_bare_protostructure`, `canonical_bare_prototype`,
`canonical_protostructure` or `canonical_prototype` as appropriate, with the
unified cooperative `timeout` option. Refined examples are transformed with their
classification; discriminators are retained. Raw structure sources instead follow
the measured structure path and return an ASUStructure. To canonicalize a bare
classification derived from a structure, construct its bare view explicitly.
See {doc}`canonicalization` for the numbered algorithm and completion contract.
