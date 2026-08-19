# Material-information levels

*httk-atomistic* describes a crystalline material at six levels of information,
organised as a 2×3 matrix. The rows differ in how much *geometrical*
information is kept; the columns differ in whether the site identities are
anonymous placeholders or real chemical species. Each cell is a value family
(a backend value plus its views) in `httk.atomistic`.

| Level of geometrical information | Anonymous species | Assigned species |
| --- | --- | --- |
| Wyckoff positions only | {py:class}`~httk.atomistic.Protopattern` | {py:class}`~httk.atomistic.Protostructure` |
| Geometrical class | {py:class}`~httk.atomistic.Prototype` | {py:class}`~httk.atomistic.Structuretype` |
| Exact geometry | {py:class}`~httk.atomistic.CrystalPattern` | {doc}`Structure <structures>` |

The top row keeps only a standard-setting space group and its occupied Wyckoff
positions. The middle row adds a *geometrical-class* distinction on top of that.
The bottom row fixes the exact continuous degrees of freedom (cell parameters
and free coordinates). Reading down a column loses geometrical information;
reading right across a row assigns real species to anonymous placeholders.

## How the levels relate

The levels combine by adding one piece of information at a time. These are
information-content relationships, not class inheritance:

- Protopattern + geometrical-class information → Prototype
- Protopattern + species assignment → Protostructure
- Prototype + species assignment → Structuretype
- Protostructure + geometrical-class information → Structuretype
- Prototype + exact geometrical parameters → CrystalPattern
- Structuretype + exact parameters → Structure
- CrystalPattern + species assignment → Structure

A {py:class}`~httk.atomistic.Prototype` and a
{py:class}`~httk.atomistic.Structuretype` pin their geometrical class by a
canonical *representative* (a standard-setting fundamental-domain value holding
one exact realization) and/or an externally assigned *discriminator* string
(AFLOW `-001`-style). At least one of the two is required; both may be given,
and equality compares exactly the information present, so a representative-only
value never equals a discriminator-only value.

The representative's continuous degrees of freedom are a **class anchor, not
exact-structure data**. A prototype or structuretype therefore corresponds to
*many* exact structures within its class — every realization that shares the
class — whereas a {py:class}`~httk.atomistic.CrystalPattern` or a `Structure`
fixes the continuous degrees of freedom and so names a single exact geometry.

## Naming and capitalization

The canonical taxonomy terms are single-capital compound words:
`Protopattern`, `Protostructure`, `Prototype`, `Structuretype`, and
`Formulapattern`. `CrystalPattern` carries two capitals because the bare word
"Pattern" — like the bare word "Formula" — would be ambiguous across contexts;
an extra word disambiguates it. Compounds formed *from* `CrystalPattern` may
drop "Crystal": `FundamentalDomainPattern` and `ASUPattern`.

The older names survive as aliases for discoverability only; documentation and
new code use the canonical names.

| Alias | Canonical |
| --- | --- |
| `AnonymousStructure`, `AnonymousStructureView`, `AnonymousStructureLike` | `CrystalPattern` family |
| `WyckoffPrototype` | `Protopattern` |
| `ProtopatternType` | `Prototype` |
| `AnonymousFormula`, `AnonymousFormulaView` | `Formulapattern` family |

## Dummy species

The two anonymous columns of the exact and geometrical-class rows
(`CrystalPattern`, `FundamentalDomainPattern`, and the representative held by a
`Prototype`) use a deliberately narrow dummy-species shape. The label is
carried through the `labels` decoration and the species name; it is never
encoded as a chemical symbol:

```python
from httk.atomistic import Species
from httk.atomistic.models.crystalpattern import dummy_species, is_dummy_species

species = dummy_species("A")
assert species == Species("A", ("X",), (1,), labels=("A",))
assert is_dummy_species(species)
```

`is_dummy_species` requires exactly one `"X"` chemical symbol, unit
concentration, matching name/label, and no mass, attachments, charge, spin,
original name, or concentration decoration. Consequently a species named `A`
with `labels=("other",)` is not a dummy species. A `Protopattern` and a
`Prototype` carry the anonymous class labels (`A`, `B`, `C`, ...) directly and
have no dummy `Species` objects at all.

## What crosses the boundary

The conversion boundary is intentionally explicit. The following table lists
features rejected during conversion and features deliberately erased when a
conversion is otherwise valid.

| Conversion boundary | Rejected | Deliberately erased |
| --- | --- | --- |
| Structure → `CrystalPattern`/`FundamentalDomainPattern` | disorder or partial occupancy; duplicate-element species; assemblies; `chemical_composition`; site moments | species identities become dummy labels; charge, formula metadata, `optimization_type`, `immutable_id`, `last_modified`, molecular metadata, and representatives |
| Structure → `Protostructure` | assemblies; `chemical_composition`; site moments; a species containing `"X"` (including attached `"X"`) | charge, formula metadata, `optimization_type`, `immutable_id`, `last_modified`, molecular metadata, and representatives |

`Protostructure` is different here: its `Species` objects retain disorder and
partial occupancy, including their real chemical symbols, concentrations, and
decorations. Those are not rejected merely because they are non-singleton.

The useful conversion matrix is below. Views either present an existing value
(exact, pass-through), extract a coarser level from a finer one (exact), or
recognize a level from a plain structure (tolerant, needs the
symmetry-recognition path, spglib).

| Construction | Result | Boundary |
| --- | --- | --- |
| `ProtopatternView(Protostructure)` | erased anonymous protopattern | exact erasure of species |
| `ProtopatternView(Prototype or Structuretype)` | the geometrical class's protopattern | exact extraction/erasure |
| `ProtopatternView(FundamentalDomainPattern)` | folded protopattern | exact discretization |
| `ProtopatternView(unitcell or ordinary structure)` | recognized protopattern | tolerant recognition (spglib) |
| `CrystalPatternView(CrystalPattern)` | crystal-pattern view | exact/pass-through |
| `CrystalPatternView(FundamentalDomainPattern)` | expanded unit cell | exact |
| `CrystalPatternView(structure)` | anonymized projection | validates the rejection rules above |
| `PrototypeView(Prototype)` | prototype view | exact/pass-through |
| `PrototypeView(Structuretype)` | anonymized prototype (discriminator carried over) | exact; the discriminator names the species-independent class |
| `PrototypeView(ASUStructureView(s, setting=...))` | standard-setting prototype | exact ASU path after the requested setting is chosen |
| `PrototypeView(unitcell or ordinary structure)` | recognized prototype | tolerant recognition (spglib) |
| `StructuretypeView(Structuretype)` | structuretype view | exact/pass-through |
| `StructuretypeView(unitcell or ordinary structure)` | recognized structuretype | tolerant recognition via the conventional-cell path (spglib) |
| `ProtostructureView(Protostructure)` | protostructure view | exact/pass-through |
| `ProtostructureView(Structuretype)` | protostructure (drops the geometrical class) | exact |
| `ProtostructureView(ASUStructureView(s, setting=...))` | geometry-free real-species key | exact ASU path |
| `ProtostructureView(unitcell or ordinary structure)` | recognized protostructure | tolerant recognition (spglib) |
| `UnitcellStructureView(CrystalPattern or FundamentalDomainPattern)` | — | raises: dummy species are not real species |
| `ProtostructureView(CrystalPattern)` | — | raises: dummy species are not real species |
| `PrototypeView(Protostructure)` | — | raises: the families have different species semantics |

Recognition from a plain structure is the tolerant/spglib boundary. Existing
ASU, pattern, prototype, and protostructure values use exact data, with no
recognition tolerance. For a source that needs a particular setting, use the
sanctioned idiom shown in the `PrototypeView` row:
`PrototypeView(ASUStructureView(s, setting=...))`. `Prototype` itself is
standard-setting-only; it does not accept `setting=`, `standard=`, or
`transform=` as recognition arguments.

## Formula conveniences

The anonymous cells expose `anonymous_formula`; the assigned cells expose both
`formula` (real species) and `anonymous_formula` (site amounts anonymized).
Formula projections use Wyckoff multiplicities, and reduced rendering removes a
common GCD:

```python
from httk.atomistic import CrystalPattern, CrystalPatternView

pattern = CrystalPattern(
    [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
    [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
    species_at_sites=("A", "B"),
)
view = CrystalPatternView(pattern)
assert view.anonymous_formula == "AB"
assert view.unwrap() is pattern
```

`Protostructure` is the isopointal key: it has no cell or coordinates, is
hashable, and is safe as a dictionary key. Equality and hashing use its
standard-setting space group and its occupied Wyckoff positions together with
the associated `Species` values, so equivalent construction order does not
change the key. `Protopattern` is the anonymous counterpart of that key.

## Labels

An httk label is a compact string encoding the information content of an
unsuffixed AFLOW-style prototype label: a space group, its occupied Wyckoff
letters, and the partition of those occupations into species classes. The
single home of the notation is
`httk.atomistic.models.protopattern.notation`. The grammar is:

```
ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*                 # protopattern label
ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*:NAME(-NAME)*    # protostructure label
```

A `GROUP` is the concatenation of one class's Wyckoff letters, sorted
alphabetically, a letter occupied `k >= 2` times prefixed by the integer `k`
(`2e`); count `1` is omitted. `ANON` is the anonymous formula (`A`, `B`, `C`,
...) built in group order with per-group summed conventional multiplicities
reduced by their overall GCD.

### httk labels are not AFLOW labels

The **httk group-ordering convention** orders the occupation groups
lexicographically by their sorted Wyckoff-letter sequences. This ordering is
*element-agnostic*, so a `ProtostructureLabel` is exactly its erased
`ProtopatternLabel` plus the `:` species suffix. AFLOW, by contrast, orders the
classes by element symbol **alphabetically**, so its unsuffixed prefix depends
on the chemistry. The two are therefore genuinely different strings, and an
httk label is **not** an AFLOW label.

The assigned-species classes expose both, as distinct properties: `label` (the
httk convention, a `ProtostructureLabel`) and `aflow_label` (the AFLOW-style
rendering, a plain `str`). For calcite, `Protostructure(167, a:Ca, b:C, e:O)`:

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
`parse_anonymous_formula` for `Formulapattern`.

### Canonical vs plain labels

Any faithful render of an object is *the* protopattern or protostructure label.
The *canonical* protopattern or protostructure label is the one obtained from a
normalizer-canonical object — one derived via `canonical_asu`. The renderer
performs no affine-normalizer pass this round, so labels from hand-built,
non-canonical objects are faithful but not necessarily canonical. Whenever text
speaks of the label of an arbitrary value it uses the plain form ("the
protostructure label"), reserving "the canonical … label" for a
normalizer-canonical source.

The AFLOW-style `-001` discriminators belong to a `Prototype` or
`Structuretype` (their `discriminator` field), which name a species-independent
geometrical class. They are **never** part of the label.

## Storage records

The families have durable, layout-independent storage records in
`httk.atomistic.storage.records`. Alongside the existing
`ProtostructureRecord` (`atomistic_protostructure`) and
`FundamentalDomainPatternRecord` (`atomistic_fundamental_domain_pattern`), the
matrix adds four records:

| Record | Storage name | Value |
| --- | --- | --- |
| `ProtopatternRecord` | `atomistic_protopattern` | `Protopattern` |
| `PrototypeRecord` | `atomistic_prototype` | `Prototype` |
| `StructuretypeRecord` | `atomistic_structuretype` | `Structuretype` |
| `FundamentalDomainPatternRecord` | `atomistic_fundamental_domain_pattern` | `FundamentalDomainPattern` |

Each record carries the value identity of its family, so two equal values
produce records with the same content id (the deduplication key) and unequal
values differ. `PrototypeRecord` and `StructuretypeRecord` enforce the
"at least one of representative or discriminator" rule and store the optional
representative as a nested record. `CrystalPattern` itself stays non-storable,
as before.

`ProtostructureRecord.label` now renders the **httk protostructure label** (for
example `AB_cF8_225_a_b:Na-Cl`) as its queryable `label` column, switched from
the older compact format. The content ids are unchanged — the label is a
convenience and query column, not the record's identity, and is not unique
across protostructures whose species share a name but differ in another
`Species` field. Count and deduplicate by row (content id), never by label.
`StructuretypeRecord` exposes the same label column; its discriminator is not
part of the label, so structuretypes that share a protostructure but differ in
class collide on it.

The registry record name `atomistic-prototype` now resolves to the new
class-level `PrototypeRecord`; the renamed `FundamentalDomainPatternRecord` is
an embedded component record (nested inside `PrototypeRecord` and
`StructuretypeRecord` as the optional representative) and deliberately has no
registry entry of its own.

Because the label column switched format and the prototype tables were
redesigned, **pre-existing stores carry stale label columns and orphaned
prototype tables**. Rebuilding the store from its source values is the
documented remedy. Concretely: a pre-series store's `atomistic_prototype_v1`
tables are orphaned by the rename to `atomistic_fundamental_domain_pattern` and
`atomistic_prototype` — their rows are not migrated, and re-ingesting the source
values produces new content ids under the new records. And a store written
across the label switch holds mixed formats in the `ProtostructureRecord.label`
column — old `"225/b:Cl,a:Na"`-style rows alongside httk-label rows; record
identity (the content id) is unaffected, and a rebuild normalizes the column.
Likewise, the `atomistic_protostructure_v1` and `atomistic_wyckoff_occupation_v1`
tables are orphaned by the removal of the `_v1` storage-name suffixes (to
`atomistic_protostructure` and `atomistic_wyckoff_occupation`): their rows are not
migrated, and because the identity name participates in hashing the content ids
change, so re-ingesting the source values under the new records is the remedy.

## Deferred features

The following are deliberately not part of this round and not part of the
conversion contracts above:

- `same_prototype()`.
- OPTIMADE serving (definitions and providers) for these families.
- The `Structuretype(prototype, species=...)` species-assignment convenience
  constructor.
- Normalizer-canonicalized label rendering (the affine-normalizer pass that
  would make every faithful label canonical).
