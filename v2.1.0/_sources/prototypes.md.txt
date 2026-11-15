# Material-information levels

*httk-atomistic* describes a crystalline material at several levels of
information, organised as a matrix. The rows differ in how much *geometrical*
information is kept; the columns differ in whether the site identities are
anonymous placeholders or real chemical species. Each cell is a value family
(a backend value plus its views) in `httk.atomistic`.

| Level of geometrical information | Anonymous occupation | Assigned species |
| --- | --- | --- |
| None (composition only) | {py:class}`~httk.atomistic.Formulatype` | {py:class}`~httk.atomistic.ChemicalFormula` |
| Wyckoff positions, optionally with a representative/discriminator | {py:class}`~httk.atomistic.Prototype` | {py:class}`~httk.atomistic.Protostructure` |
| Exact geometry | {py:class}`~httk.atomistic.Structuretype` | {doc}`Structure <structures>` |

The top row keeps only the composition. The middle row keeps a standard-setting
space group and its occupied Wyckoff positions. The bottom row fixes the exact
continuous degrees of freedom (cell parameters and free coordinates). Reading
down a column loses geometrical information; reading right across a row assigns
real species to anonymous placeholders.

{py:class}`~httk.atomistic.Prototype` and
{py:class}`~httk.atomistic.Protostructure` are the two middle-row
geometrical-classification keys. A base value contains only its standard-setting
space group and occupied Wyckoff positions. Either may additionally carry an
exact fundamental-domain *representative* (a standard-setting value holding one
exact realization), an externally assigned *discriminator* string (AFLOW
`-001`-style), or both. These optional fields participate in equality and
content identity, so a representative-only value never equals a discriminator-only
value, and a base-only value is distinct from either refined form. Recognizing a
key from a structure, and deriving one key from another, always return a base
value; the representative and discriminator are supplied only by explicit
construction.

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

- Prototype + species assignment → Protostructure
- Prototype + exact geometrical parameters → Structuretype
- Protostructure + exact geometrical parameters → Structure
- Structuretype + species assignment → Structure
- Prototype or Protostructure projected onto composition only → Formulatype or ChemicalFormula

A representative and/or a discriminator *refine* a base Prototype or
Protostructure — pinning a specific geometrical class — without changing which
row it occupies; both refined and base forms are middle-row keys.

## Naming and capitalization

The canonical taxonomy terms are single-capital compound words: `Formulatype`,
`Prototype`, `Protostructure`, and `Structuretype`. The suffix `-type` marks the
anonymous-occupation column (`Formulatype`, `Structuretype`, and `Prototype` are
the anonymous counterparts of `ChemicalFormula`, `Structure`, and
`Protostructure`); the prefix `Proto` marks the cutoff-free Wyckoff
classification.

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
new code use the canonical names.

| Alias | Canonical |
| --- | --- |
| `AnonymousStructure`, `AnonymousStructureView`, `AnonymousStructureLike` | `Structuretype` family |
| `AnonymousFormula`, `AnonymousFormulaView` | `Formulatype` family |

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
with `labels=("other",)` is not a dummy species. A `Prototype` carries the
anonymous class labels (`A`, `B`, `C`, ...) directly and has no dummy `Species`
objects at all.

## What crosses the boundary

The conversion boundary is intentionally explicit. The following table lists
features rejected during conversion and features deliberately erased when a
conversion is otherwise valid.

| Conversion boundary | Rejected | Deliberately erased |
| --- | --- | --- |
| Structure → `Structuretype`/`FundamentalDomainTemplate` | disorder or partial occupancy; duplicate- or multi-element species; a species whose symbol is `"X"` or `"vacancy"`; assemblies; `chemical_composition`; site moments | species identities become dummy labels; charge, spin, mass, formula metadata, `optimization_type`, `immutable_id`, `last_modified`, and molecular metadata |
| Structure → `Protostructure` | assemblies; molecular structures; `chemical_composition`; site moments; a species containing `"X"` (including attached `"X"`) | charge/formula metadata, `optimization_type`, `immutable_id`, `last_modified`, and molecular metadata |

`Protostructure` is different here: its `Species` objects retain disorder and
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
| `PrototypeView(FundamentalDomainTemplate)` | folded base prototype | exact extraction/discretization |
| `PrototypeView(Structuretype)` | prototype recognized from the exact anonymous geometry | tolerant recognition (spglib) |
| `PrototypeView(ASUStructureView(s, setting=...))` | standard-setting prototype | exact ASU path after the requested setting is chosen |
| `PrototypeView(unitcell or ordinary structure)` | recognized prototype | tolerant recognition (spglib) |
| `StructuretypeView(Structuretype)` | structuretype view | exact/pass-through |
| `StructuretypeView(FundamentalDomainTemplate)` | expanded unit cell | exact |
| `StructuretypeView(structure)` | anonymized projection | exact; validates the rejection rules above |
| `ProtostructureView(Protostructure)` | protostructure view | exact/pass-through |
| `ProtostructureView(ASUStructureView(s, setting=...))` | geometry-free real-species key | exact ASU path |
| `ProtostructureView(unitcell or ordinary structure)` | recognized protostructure | tolerant recognition (spglib) |
| `ProtostructureView(Structuretype or FundamentalDomainTemplate)` | — | raises: dummy species are not real species |
| `UnitcellStructureView(Structuretype or FundamentalDomainTemplate)` | — | raises: dummy species are not real species |

Recognition from a plain structure is the tolerant/spglib boundary. Existing
ASU, structuretype, prototype, and protostructure values use exact data, with no
recognition tolerance. For a source that needs a particular setting, use the
sanctioned idiom shown above:
`PrototypeView(ASUStructureView(s, setting=...))`. Recognition of a raw
structure resolves the standard setting.

## Formula conveniences

The geometry-bearing anonymous cells (`Structuretype` and `Prototype`, through
their views) expose `anonymous_formula`; the assigned cells expose both
`formula` (real species) and `anonymous_formula` (site amounts anonymized). A
`Formulatype` has no `anonymous_formula` attribute — it *is* the anonymous
formula, rendered as its string value (`str(FormulatypeView(...))`, e.g.
`"A3B2"`). Formula projections use Wyckoff multiplicities, and reduced rendering
removes a
common GCD:

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

`Protostructure` is the isopointal-with-species key: it has no cell or
coordinates. Equality uses its standard-setting space group and its occupied
Wyckoff positions together with the associated `Species` values (plus any
representative or discriminator), so equivalent construction order does not
change it. `Prototype` is the anonymous counterpart of that key. Both families
are **hashable** and safe as dictionary or set keys: hashing uses the base key
(space group, occupied Wyckoff positions, species or anonymous occupation, and
the discriminator), while equality additionally compares a representative when
one is present. Equal objects therefore hash equal; two values that differ only
in their representative may collide on the hash but remain unequal.

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
reduced by their overall GCD.

A structure's canonicalization preserves chirality by default. The canonical
`Protostructure`/`Prototype` label is instead built from the chirality-normalized
result (`canonical_asu(preserve_chirality=False)`, or `normalize_chirality`
applied to a chirality-preserved result; see {doc}`asu`), so the two members of an
enantiomorphic pair share one canonical label.

### httk labels are not AFLOW labels

The **httk group-ordering convention** orders the occupation groups
lexicographically by their sorted Wyckoff-letter sequences. This ordering is
*element-agnostic*, so a protostructure label is exactly its erased prototype
label plus the `:` species suffix. AFLOW, by contrast, orders the classes by
element symbol **alphabetically**, so its unsuffixed prefix depends on the
chemistry. The two are therefore genuinely different strings, and an httk label
is **not** an AFLOW label.

The assigned-species classes expose both, as distinct properties: `label` (the
httk convention, a {py:class}`~httk.atomistic.ProtostructureLabel`) and
`aflow_label` (the AFLOW-style rendering, a plain `str`). For calcite,
`Protostructure(167, a:Ca, b:C, e:O)`:

```python
from httk.atomistic import Protostructure, Species

Ca, C, O = Species("Ca", ("Ca",), (1,)), Species("C", ("C",), (1,)), Species("O", ("O",), (1,))
calcite = Protostructure(167, [("a", Ca), ("b", C), ("e", O)])
assert calcite.label == "ABC3_hR10_167_a_b_e:Ca-C-O"
assert calcite.aflow_label == "ABC3_hR10_167_b_a_e:C-Ca-O"
```

The httk label orders the groups `a`, `b`, `e` by Wyckoff letter; the AFLOW
label orders them `b`, `a`, `e` to follow the alphabetical elements
`C`, `Ca`, `O`.

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
directions: `parse(render(x)) == x` for element-pure values and
`render(parse(s)) == s` for canonical strings. This mirrors
`parse_anonymous_formula` for `Formulatype`.

### Canonical vs plain labels

Any faithful render of an object is *the* prototype or protostructure label.
The *canonical* prototype or protostructure label is the one obtained from a
normalizer-canonical object — one derived via `canonical_asu`. The renderer
performs no affine-normalizer pass this round, so labels from hand-built,
non-canonical objects are faithful but not necessarily canonical. Whenever text
speaks of the label of an arbitrary value it uses the plain form ("the
protostructure label"), reserving "the canonical … label" for a
normalizer-canonical source.

The AFLOW-style `-001` discriminators belong to a `Prototype` or a
`Protostructure` (their `discriminator` field), which name a species-independent
geometrical class. They are **never** part of the label.

## Similarity and exact travel

`Prototype.similar` and `Protostructure.similar` first compare their discrete
space-group and occupation keys, then apply discriminator compatibility. If
both values have representatives, the continuous comparison is the total
Cartesian atom travel returned by the public `structure_delta(first, second)`;
missing representatives do not invent a distance. `structure_delta` maps the
structures into a common subgroup and setting, pairs compatible Wyckoff
orbits, and sums the shortest periodic Cartesian travel of their atoms. Each
endpoint uses its own cell, so lattice changes contribute through the atom
positions. It is not a content-id or label comparison. `similar` returns
`False` only when no common representation exists (`NoCommonRepresentation`, a
`ValueError` subclass in `httk.atomistic.symmetry.paths`); other errors from a
broken representative — a singular cell basis, a non-three-dimensional cell, or
non-finite travel — propagate.

## Storage records

The families have durable, layout-independent storage records in
`httk.atomistic.storage.records`:

| Record | Storage name | Value |
| --- | --- | --- |
| `PrototypeRecord` | `atomistic_prototype` | `Prototype` |
| `ProtostructureRecord` | `atomistic_protostructure` | `Protostructure` |
| `FundamentalDomainTemplateRecord` | `atomistic_fundamental_domain_template` | `FundamentalDomainTemplate` |
| `FundamentalDomainStructureRecord` | `atomistic_fundamental_domain_structure` | `FundamentalDomainStructure` |

Each record carries the value identity of its family, so two equal values
produce records with the same content id (the deduplication key) and unequal
values differ. `PrototypeRecord` and `ProtostructureRecord` accept base-only
values and store the optional representative as a nested record
(`FundamentalDomainTemplateRecord` for a prototype,
`FundamentalDomainStructureRecord` for a protostructure) and the optional
discriminator as a plain column. `Structuretype` itself stays non-storable.

Both `PrototypeRecord.label` and `ProtostructureRecord.label` render the **httk
label** (for example `AB_cF8_225_a_b` and `AB_cF8_225_a_b:Na-Cl`) as a queryable
`label` column. The content ids are unchanged by this — the label is a
convenience and query column, not the record's identity, and it is not unique:
the discriminator is not part of the label, so records that share occupations
but differ in class collide on it, and two protostructures whose species share a
name but differ in another `Species` field also collide. Count and deduplicate
by row (content id), never by label.

The registry record names are `atomistic-prototype` (family `prototypes`) and
`atomistic-protostructure` (family `protostructures`), with
`atomistic-fundamental-domain-structure` in the `structures` family.
`FundamentalDomainTemplateRecord` is an embedded component record (nested inside
`PrototypeRecord` as the optional representative) and deliberately has no
registry entry of its own.

Because the taxonomy and the storage layout were redesigned, **pre-existing
stores carry orphaned tables and, where the label format changed, stale label
columns**. Rebuilding the store from its source values is the documented remedy;
no compatibility registry keys are provided. Concretely, the following tables
are orphaned — their rows are not migrated, and (because the identity name
participates in hashing) re-ingesting the source values produces new content ids
under the current records:

- the retired four-class layout's `atomistic_prototemplate` and
  `atomistic_structuretype` tables (and any earlier `atomistic_prototype_v1`
  tables);
- the `atomistic_protostructure_v1` and `atomistic_wyckoff_occupation_v1` tables,
  orphaned by removing the `_v1` storage-name suffixes (now
  `atomistic_protostructure` and `atomistic_wyckoff_occupation`);
- the pre-"Pattern"→"Template" rename tables `atomistic_protopattern` and
  `atomistic_fundamental_domain_pattern`.

Separately, a store written across the label-format switch holds mixed formats
in the `ProtostructureRecord.label` column — old `"225/b:Cl,a:Na"`-style rows
alongside httk-label rows. Record identity (the content id) is unaffected there;
a rebuild simply normalizes the column.

## Deferred features

The following are deliberately not part of this round and not part of the
conversion contracts above:

- `same_prototype()`.
- OPTIMADE serving (definitions, providers, and bindings) for the prototype and
  protostructure families.
- The species-assignment convenience constructors
  (`Protostructure(prototype, species=...)`, `Structure(structuretype, species=...)`).
- Normalizer-canonicalized label rendering (the affine-normalizer pass that
  would make every faithful label canonical).

The full guide, {doc}`details/structural_classes`, covers the naming rationale
and how the classes relate to isopointal/isoconfigurational structures and
AFLOW labels.
