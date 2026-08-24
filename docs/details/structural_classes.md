# Naming of *httk₂* structural classes

## Overview

| Geometrical information | Anonymous occupation | Assigned species |
| ----------------------- | -------------------- | ----------------- |
| None                    | `Formulatype`        | `ChemicalFormula` |
| Wyckoff (± representative/discriminator) | `Prototype` | `Protostructure`  |
| Exact geometry          | `Structuretype`      | `Structure`       |

The middle row is represented by exactly two structural families. `Prototype`
stores anonymous `PrototypeOccupation` values; `Protostructure` stores assigned
`WyckoffOccupation` values. Either may be base-only, or may additionally carry
an exact fundamental-domain representative, an assigned discriminator, or both —
the optional refinements that pin a specific geometrical class.

## Details

The naming of the structural classes in *httk* starts from:

* A **crystal structure**, `Structure`, has both precise geometry (atomic coordinates and cell parameters) and species information (chemical elements and may have, e.g., isotopes, oxidation states, etc.).

Structures can be given a “geometrical classification” representing less precise geometrical information. This notion is adopted from the related (but not identical) concept of isoconfigurational structures for which the geometrical constraint is: “the crystallographic point configurations (crystallographic orbits) and their geometrical interrelationships are similar” [https://doi.org/10.1107/S0108767307038081]. Specifically, we adopt this idea with the more specific meaning that two structures have the same “geometrical classification” if there exists a representation of them in the same spacegroup and setting (origin, orientation and cell choice; but usually disregarding enantiomorphs) where the same species occupy the same Wyckoff positions with similar values for all Wyckoff degrees of freedom (i.e., “similar point configurations”) in a unit cell where the cell parameters have similar values modulo an overall scaling factor. The exact degree of similarity has to be defined by numerical cutoff parameters. In *httk₂* this cutoff-based geometrical class is not a separate structural family; it *refines* a `Protostructure` (or its anonymous counterpart `Prototype`) through an optional exact **representative** — a standard-setting fundamental-domain value that holds one exact realization of the class — and/or an assigned **discriminator** string. Two such values are compared at a chosen tolerance through the `similar` method, which measures the continuous distance between their representatives (see *Similarity*, below).

Removing that cutoff-based refinement leaves the higher-level classification that avoids the numerical cutoff by classifying only by the spacegroup, the occupied Wyckoff positions, and the species that occupy them. We use the prefix “Proto” for this. Hence:

* A **protostructure**, `Protostructure`, is a structural representation specifying only the spacegroup (also distinguishing enantiomorphs), the occupied Wyckoff positions, and the species that occupy them.

These classifications still *distinguish* the two members of an enantiomorphic pair, as stated above. Note, however, that the default canonicalization pipeline (`canonicalize`, `canonical_asu`) instead maps an enantiomorphic pair to a single canonical representative — the lower-numbered member — mirroring the crystal exactly when needed. Passing `preserve_chirality=True` keeps each member in its own group and so retains the distinction these classifications draw.

We can take the weaker geometrical representation one step further, down to:

* A **chemical formula**, `ChemicalFormula`, represents the structure only in terms of composition of the elements. The “Normalized Formula” gives the composition only in relative terms; any other unit cell representation can be condensed into a formula.

Going in another direction, one can strip the absolute meaning of the chemical species and classify occupation only in terms of equivalent and non-equivalent species (the sites carry placeholder/equivalence assignments — a template into which real species may later be filled in). We mark this anonymous-occupation direction with the suffix “-type”. Hence:

* A **structuretype**, `Structuretype`, has precise geometry but only an abstract representation that indicates which sites are occupied by equivalent species, i.e., an anonymous occupation template (and where the order of assignments is not regarded as relevant). It is the anonymous-occupation counterpart of `Structure`.

Following the same naming to a less precise geometrical classification takes us to:

* A **prototype**, `Prototype`, is the anonymous-occupation counterpart of `Protostructure`: a structural representation based only on the spacegroup, the occupied Wyckoff positions, and their occupation template. Following the “-type” scheme systematically it would be a **templatetype**, but the more commonly used **prototype** is the already-established name for this, so that is the one we adopt. (“Templatetype” is only a prose explanation of where `Prototype` sits in the scheme; there is no such code name or alias.) A `Prototype` may likewise carry an exact fundamental-domain representative (a `FundamentalDomainTemplate`) and/or a discriminator to pin a geometrical class.

The word “Template” itself survives only for the exact fundamental-domain anonymous values that serve as class anchors and ASU keys — `FundamentalDomainTemplate` and `ASUTemplate` — not for the exact anonymous family as a whole, which is `Structuretype`.

To cover a couple of other related names and explain how they differ:

* **Isopointal structures**: account for the spacegroup (with or without distinguishing enantiomorphic groups) and the same complete Wyckoff sequence, including repeated independent occupations but excluding the chemical identity of the occupying atoms.
* **Isoconfigurational structures**: isopointal structures further distinguished by geometrical classification (i.e., similarities in atomic coordinates and cell parameters, but sometimes with further requirements), however still described by colorless occupation.
* **AFLOW labels**: very similar to httk prototypes, except occupations are ordered by occupying element symbol, making the AFLOW labels different for, e.g., ZrO₂ (A2B_oP12_29_2a_a) and FeS₂ (AB2_oP12_29_a_2a).
* **Extended AFLOW labels**: an extension of three digits is added to describe geometrical similarity (-001, -002, …), bringing them closer to discriminator-carrying prototypes.
* **Decorated AFLOW labels**: AFLOW labels are sometimes used with a “decoration” to indicate chemical species, although there does not seem to be one universal endorsed format. These decorations take AFLOW labels closer to httk protostructures.

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
identities. See {doc}`../prototypes` for the full grammar, the httk-vs-AFLOW
comparison, and the Pearson-symbol and parser rules.

## Storage migration

The only middle-row storage identities are `atomistic_prototype` and
`atomistic_protostructure`; their registry records are `atomistic-prototype`
and `atomistic-protostructure` in the `prototypes` and `protostructures`
families. Optional representatives and discriminators are projected into the
canonical identity. Stores built with the retired four-class layout or its
table names must be rebuilt; old registry keys are intentionally not accepted.
