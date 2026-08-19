# Prototype families

The prototype APIs describe crystal geometry at three different resolutions.
`AnonymousStructure` keeps a unit cell, sites, and dummy species labels;
`Prototype` keeps a standard-setting space group and one Wyckoff site per
orbit, with dummy species labels; `Protostructure` keeps only a standard-setting
space group and occupied Wyckoff positions with real `Species` objects. Views
(`AnonymousStructureView`, `PrototypeView`, and `ProtostructureView`) present
existing values or recognize them from compatible sources. The first two
representations form the dummy-species/geometry family; `Protostructure` is the
real-species/geometry-free family. `PrototypeLike` is the umbrella union over
both families.

## Dummy species

Anonymous structures and prototypes use a deliberately narrow dummy-species
shape. The label is carried through the `labels` decoration and the species
name; it is never encoded as a chemical symbol:

```python
from httk.atomistic import Species
from httk.atomistic.models.prototype import dummy_species, is_dummy_species

species = dummy_species("A")
assert species == Species("A", ("X",), (1,), labels=("A",))
assert is_dummy_species(species)
```

`is_dummy_species` requires exactly one `"X"` chemical symbol, unit
concentration, matching name/label, and no mass, attachments, charge, spin,
original name, or concentration decoration. Consequently a species named `A`
with `labels=("other",)` is not a dummy species.

## What crosses the boundary

The conversion boundary is intentionally explicit. The following table lists
features rejected during conversion and features deliberately erased when a
conversion is otherwise valid.

| Conversion boundary | Rejected | Deliberately erased |
| --- | --- | --- |
| Structure → `AnonymousStructure`/`Prototype` | disorder or partial occupancy; duplicate-element species; assemblies; `chemical_composition`; site moments | species identities become dummy labels; charge, formula metadata, `optimization_type`, `immutable_id`, `last_modified`, molecular metadata, and representatives |
| Structure → `Protostructure` | assemblies; `chemical_composition`; site moments; a species containing `"X"` (including attached `"X"`) | charge, formula metadata, `optimization_type`, `immutable_id`, `last_modified`, molecular metadata, and representatives |

Protostructure is different here: its `Species` objects retain disorder and
partial occupancy, including their real chemical symbols, concentrations, and
decorations. Those are not rejected merely because they are non-singleton.

The useful conversion matrix is:

| Construction | Result | Boundary |
| --- | --- | --- |
| `AnonymousStructureView(AnonymousStructure)` | lazy anonymous view | exact/pass-through |
| `AnonymousStructureView(structure)` | lazy anonymous projection | validates the rejection rules above |
| `PrototypeView(Prototype)` | prototype view | exact/pass-through |
| `PrototypeView(ASUStructureView(s, setting=...))` | standard-setting prototype | exact ASU path after the requested setting is chosen |
| `PrototypeView(unitcell or ordinary structure)` | recognized prototype | tolerant recognition; requires the symmetry-recognition path (spglib) |
| `ProtostructureView(Protostructure)` | protostructure view | exact/pass-through |
| `ProtostructureView(ASUStructureView(s, setting=...))` | geometry-free real-species key | exact ASU path |
| `ProtostructureView(unitcell or ordinary structure)` | recognized protostructure | tolerant recognition; requires the symmetry-recognition path |
| `UnitcellStructureView(AnonymousStructure or Protostructure)` | — | raises: neither family is a unit-cell structure input |
| `ProtostructureView(AnonymousStructure)` | — | raises: dummy species are not real species |
| `PrototypeView(Protostructure)` | — | raises: the families have different species semantics |

Recognition from a plain structure is the tolerant/spglib boundary. Existing
ASU and prototype/protostructure values use exact data, with no recognition
tolerance. For a source that needs a particular setting, use the sanctioned
idiom shown in the `PrototypeView` row: `PrototypeView(ASUStructureView(s,
setting=...))`. `Prototype` itself is standard-setting-only; it does not accept
`setting=`, `standard=`, or `transform=` as recognition arguments.

## Formula conveniences

`AnonymousStructure` and `Prototype` expose `anonymous_formula`. Their views
retain the same convenience. `Protostructure` and `ProtostructureView` expose
both `formula` and `anonymous_formula`; the former uses the real species and
the latter anonymizes their site amounts. Formula projections use Wyckoff
multiplicities, and reduced rendering removes a common GCD:

```python
from httk.atomistic import AnonymousStructure, AnonymousStructureView

anonymous = AnonymousStructure(
    [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
    [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
    species_at_sites=("A", "B"),
)
view = AnonymousStructureView(anonymous)
assert view.anonymous_formula == "AB"
assert view.unwrap() is anonymous
```

`Protostructure` is the isopointal key: it has no cell or coordinates, is
hashable, and is safe as a dictionary key. Equality and hashing use its
standard-setting space group and its occupied Wyckoff positions together with
the associated `Species` values, so equivalent construction order does not
change the key.

## Storage records

Both prototype families have durable, layout-independent storage records in
`httk.atomistic.storage.records`.

`ProtostructureRecord` (`atomistic_protostructure_v1`) carries exactly the value
identity of `Protostructure`: its standard-setting space group (as
`spacegroup_it_number` plus the standard `spacegroup_hall_entry`, mirroring the
structure records) and its `occupations` — a tuple of `WyckoffOccupationRecord`
(`atomistic_wyckoff_occupation_v1`), each a Wyckoff letter and a real
`SpeciesRecord`, in the protostructure's canonical order (sorted by species name
then Wyckoff letter). It has no cell or coordinates. Because the record carries
the same value identity, two equal `Protostructure` values (`==`, including a
permuted construction order) produce records with the same content id, and
unequal values differ; this is the deduplication key for a COD-scale
protostructure catalog (count by rows; filter by `spacegroup_it_number`).

`ProtostructureRecord` exposes a queryable `label` stored property with the
deterministic compact format `"<it_number>/<wyckoff>:<species_name>,..."`, listing
the occupations in the record's stored canonical order (for example
`"225/b:Cl,a:Na"`). The label is a convenience and query column only; it is **not**
the record's identity — the content id is — and it is **not unique** across distinct
protostructures: species sharing a name but differing in any other `Species` field
(concentration, charges, spins, mass, precision, ...) collide on the same label, so
counting or grouping by label may under-count distinct protostructures; count by row
(content id) instead. The record declares composite indexes
on `("spacegroup_it_number",)` and `("label",)`.

`PrototypeRecord` (`atomistic_prototype_v1`) is the geometric per-structure
object: a surd-capable `CellRecord`, `WyckoffSiteRecord` sites with their exact
free parameters, distinct dummy `SpeciesRecord` species, the standard-setting
space group, and the reduced-coordinate precision. Distinct free parameters make
distinct values, so no content deduplication is expected; the content id is still
deterministic. It indexes `("spacegroup_it_number",)` and
`("spacegroup_hall_entry",)`.

Conversion follows the established record idiom: each record declares
`__httk_canonical_source__` and `__httk_project__` (so `content_id` and a store
project a source value directly), and the module provides
`_protostructure_record_from_value`/`_protostructure_from_record` and
`_prototype_record_from_value`/`_prototype_from_record` for building records and
recovering values. The records register under the logical `protostructures` and
`prototypes` entry families (`ProtostructureEntry`/`PrototypeEntry`). A
`Prototype` value stores directly through its record; a `Protostructure` value is
stored through its `ProtostructureRecord` (the derived `label` column is computed
from the record).

## Deferred features

AFLOW/symgen labels and Pearson symbols, Wyckoff-sequence strings, OPTIMADE
serving (definitions and providers) for the prototype families, and
`same_prototype()` are deferred. They are not part of the conversion contracts
described here.
