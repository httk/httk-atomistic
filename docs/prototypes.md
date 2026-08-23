# Material-information levels

*httk₂* describes a crystalline material with three geometrical resolutions. The
anonymous column records equivalence classes; the assigned column records real
species.

| Geometrical information | Anonymous occupation | Assigned species |
| --- | --- | --- |
| None | `Formulatemplate` | `ChemicalFormula` |
| Wyckoff positions, optionally with a representative/discriminator | `Prototype` | `Protostructure` |
| Exact geometry | `CrystalTemplate` | `Structure` |

`Prototype` and `Protostructure` are the two geometrical-classification keys. A
base value contains only its standard-setting space group and occupied Wyckoff
positions. Either may additionally carry an exact fundamental-domain
representative, an externally assigned discriminator, or both. These optional
fields participate in equality and content identity; a representative-only value
and a discriminator-only value are therefore distinct. A base-only value is
valid and is distinct from either refined form.

The representative is a class anchor, not the identity of the bottom-row exact structure:
its continuous coordinates and cell are retained exactly so that the anchor can
be reconstructed, while the structural key remains a coarse classification.
`Prototype` uses a `FundamentalDomainTemplate` representative and anonymous
`PrototypeOccupation` values. `Protostructure` uses a
`FundamentalDomainStructure` representative and real-species
`WyckoffOccupation` values. A discriminator is text naming a class; it is not
part of the label, and is not interchangeable with a representative.

## Labels and recognition

The label grammar is shared by the two retained keys:

```
ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*                 # Prototype
ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*:NAME(-NAME)*    # Protostructure
```

Prototype labels use anonymous class labels (`A`, `B`, …). Protostructure
labels append real species names to the same class ordering. A discriminator
is deliberately absent from both strings: labels identify the Wyckoff
occupation shape, while the discriminator distinguishes optional class records.
The notation implementation lives in
`httk.atomistic.models.prototype.notation`.

Views preserve the source backend and are lazy when recognition is required.
Erasing a `Protostructure` to a `Prototype` preserves the space group and
occupation partition while replacing species names with canonical anonymous
labels. Recognition from an ordinary structure is tolerant and requires the
symmetry-recognition dependencies; construction from an existing exact
fundamental-domain value is exact.

## Similarity and exact travel

`Prototype.similar` and `Protostructure.similar` first compare their discrete
space-group and occupation keys, then apply discriminator compatibility. If
both values have representatives, the continuous comparison is the total
Cartesian atom travel returned by the public `structure_delta(first, second)`;
missing representatives do not invent a distance. `structure_delta` maps the
structures into a common subgroup and setting, pairs compatible Wyckoff
orbits, and sums the shortest periodic Cartesian travel of their atoms. Each
endpoint uses its own cell, so lattice changes contribute through the atom
positions. It is not a content-id or label comparison.

## Storage

The durable records are `PrototypeRecord` and `ProtostructureRecord`, stored as
`atomistic_prototype` and `atomistic_protostructure`. Their canonical source is
the corresponding value, and optional representatives/discriminators are
projected into the same content identity. Existing stores using the retired
taxonomy or old layouts must be rebuilt; no compatibility registry keys are
provided.

`FundamentalDomainTemplateRecord` remains the exact storage record for the
anonymous representative. It is independent of the coarse `PrototypeRecord`
and retains the source `FundamentalDomainTemplate` semantics.

## Exact boundary

The conversion boundary is explicit. `CrystalTemplate`/`FundamentalDomainTemplate`
reject real-species assumptions when producing anonymous data; `Protostructure`
retains real species, including their exact metadata. No conversion silently
approximates a representative: use an explicit lossy presentation when a
numeric approximation is desired.
