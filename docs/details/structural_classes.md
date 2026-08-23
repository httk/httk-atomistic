# Naming of *httk₂* structural classes

## The three-row matrix

| Geometrical information | Anonymous occupation | Assigned species |
| --- | --- | --- |
| None | `Formulatemplate` | `ChemicalFormula` |
| Wyckoff positions, optionally with a representative/discriminator | `Prototype` | `Protostructure` |
| Exact geometry | `CrystalTemplate` | `Structure` |

The middle row is intentionally represented by only two structural families.
`Prototype` stores anonymous `PrototypeOccupation` values; `Protostructure`
stores assigned `WyckoffOccupation` values. Both may be base-only, or may carry
an exact fundamental-domain representative, an assigned discriminator, or both.
The optional fields are part of equality and content identity. A discriminator
is a class name supplied by a caller (for example an AFLOW-style suffix), while
a representative is exact geometric evidence; neither is folded into the
human-readable label.

The bottom row is the exact geometry. A representative held by a middle-row
value is an exact class anchor, not a claim that the middle-row key fixes every
continuous coordinate. The source records are correspondingly separate:
`FundamentalDomainTemplateRecord` is the anonymous representative record and
`FundamentalDomainStructureRecord` is the assigned representative record.

## Similarity

The `similar` methods compare the discrete space group and occupied Wyckoff
positions first. Discriminators conflict when both are present and differ. If
both values carry representatives, their continuous distance is the total
Cartesian atom travel from the public `structure_delta` function. It maps the
structures into a common subgroup and setting, pairs compatible Wyckoff
orbits, and sums shortest periodic travel using each endpoint's own cell. It is
not a label comparison or a content-id comparison. A missing representative
leaves the continuous portion unspecified rather than fabricating a distance.

## Labels

The prototype notation is shared by the middle-row families. Anonymous labels
render class groups as `A`, `B`, and so on. Assigned labels append the real
species names after `:`. The discriminator remains separate, so two values can
render the same label while remaining unequal and having different content
identities.

## Storage migration

The only middle-row storage identities are `atomistic_prototype` and
`atomistic_protostructure`; their registry records are
`atomistic-prototype` and `atomistic-protostructure` in the `prototypes` and
`protostructures` families. Optional representatives and discriminators are
projected into the canonical identity. Stores built with the retired four-class
layout or its table names must be rebuilt; old registry keys are intentionally
not accepted.
