# Material-information levels

*httk-atomistic* describes a crystalline material at several levels of
information, organised as a matrix. The rows differ in how much *geometrical*
information is kept; the columns differ in whether the site identities are
anonymous placeholders or real chemical species. Each cell is a value family
(a backend value plus its views) in `httk.atomistic`.

| Level of geometrical information | Anonymous occupation | Assigned species |
| --- | --- | --- |
| None (composition only) | {py:class}`~httk.atomistic.Formulatype` | {py:class}`~httk.atomistic.ChemicalFormula` |
| Wyckoff positions only | {py:class}`~httk.atomistic.BarePrototype` | {py:class}`~httk.atomistic.BareProtostructure` |
| Geometrical class | {py:class}`~httk.atomistic.Prototype` | {py:class}`~httk.atomistic.Protostructure` |
| Exact geometry | {py:class}`~httk.atomistic.Structuretype` | {doc}`Structure <structures>` |

The top row keeps only composition. The bare row keeps a standard-setting
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
- Projection onto composition only → Formulatype or ChemicalFormula

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
anonymous-occupation column (`Formulatype`, `Structuretype`, and `Prototype` are
the anonymous counterparts of `ChemicalFormula`, `Structure`, and
`Protostructure`); the `Bare` prefix distinguishes cutoff-free Wyckoff classification from a
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

The anonymous families (`Structuretype`, `BarePrototype`, and `Prototype`,
through their views) expose `anonymous_formula`; the assigned cells expose both
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

`BareProtostructure` is the Wyckoff-with-species key: it has no cell or
coordinates. Equality uses its standard-setting space group and occupied
Wyckoff positions with associated `Species` values, independent of construction
order. `BarePrototype` is its anonymous counterpart. Both are hashable.
Refined values additionally compare their representative and discriminator;
values differing only in representative may hash alike but remain unequal.

The structure conveniences `canonical_bare_protostructure()` and
`canonical_bare_prototype()` return standalone chirality-normalized bare keys.
`FundamentalDomainTemplate.bare_prototype` extracts the discrete anonymous key.

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
`BareProtostructure`/`BarePrototype` label is instead built from the chirality-normalized
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
httk convention, a {py:class}`~httk.atomistic.BareProtostructureLabel` for a bare value) and
`aflow_label` (the AFLOW-style rendering, a plain `str`). For calcite,
`BareProtostructure(167, a:Ca, b:C, e:O)`:

```python
from httk.atomistic import BareProtostructure, Species

Ca, C, O = Species("Ca", ("Ca",), (1,)), Species("C", ("C",), (1,)), Species("O", ("O",), (1,))
calcite = BareProtostructure(167, [("a", Ca), ("b", C), ("e", O)])
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
directions: `parse(render(x)) == x` for element-pure bare values and
`render(parse(s)) == s` for canonical strings. This mirrors
`parse_anonymous_formula` for `Formulatype`. Use
`parse_bare_prototype_label` and `parse_bare_protostructure_label` from the
notation module, or the corresponding bare views. Parsing refined label text
returns only its bare classification; the text cannot encode its refinement.

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
