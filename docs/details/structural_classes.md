# Some thoughts on the naming of httk crystal structure classes

* A **crystal structure**, or “structure”, has both precise geometry (atomic coordinates and cell parameters) and species information (chemical elements and may have, e.g., isotopes, oxidation states, etc.).

Structures can be given a “geometrical classification” representing less precise geometrical information. This notion is adopted from the related (but not identical) concept of isoconfigurational structures for which the geometrical constraint is: “the crystallographic point configurations (crystallographic orbits) and their geometrical interrelationships are similar” [https://doi.org/10.1107/S0108767307038081]. Specifically, we adopt this idea with the more specific meaning that two structures have the same “geometrical classification” if there exist a representation of them in the same spacegroup and setting (origin, orientation and cell choice; but usually disregarding enantiomorphs) where the same species occupy the same Wyckoff positions with similar values for all Wyckoff degrees of freedom (i.e., “similar point configurations”) in a unit cell where the cell parameters have similar values modulo an overall scaling factor. The exact degree of similarity has to be defined by numerical cutoff parameters. We use the suffix “-type” to designate this lower geometrical structural resolution. Hence:

* A **crystal structure type** or “structuretype” is a structural representation based on this geometrical classification (with enantiomorphs counted as different). Two structuretypes are “the same” if they match by the criteria above.

An even higher-level geometrical classification avoids the numerical cutoff by classifying only by the spacegroup, the occupied Wyckoff positions, and the species that occupy them.
We use the prefix “Proto” for this. Hence

* A **protostructure** is a structural representation specifying only the spacegroup (also distinguishing enantiomorphs), the occupied Wyckoff positions, and the species that occupy them.

We can take the weaker geometrical representation one step further, down to:

* A **chemical formula**, or “formula,” represents the structure only in terms of composition of the elements. The “Normalized Formula” gives the composition only in relative terms; any other unit cell representation can be condensed into a formula.

Going in another direction, one can strip the absolute meaning of the chemical species and classify occupation only in terms of equivalent and non-equivalent species. The word that corresponds to Structure that we adopt for this is: “Pattern”. Hence:

* A **crystal pattern** has precise geometry but only an abstract representation that indicates which sites are occupied by equivalent species, i.e., an “occupation pattern” (and where the order of assignments is not regarded as relevant).

Following the above naming scheme to a less precise geometrical classification hence takes us to:

* A **crystal pattern type**, “patterntype” with alias “prototype”, is a structural representation based only on geometrical classification and occupation pattern.

And the next step is then:

* A **protopattern** is a structural representation specifying spacegroup, occupied Wyckoff positions, and their occupation pattern.

| Geometrical information | Anonymous occupation | Assigned species  |
| ----------------------- | -------------------- | ----------------- |
| None                    | `Formulapattern`     | `ChemicalFormula` |
| Wyckoff only            | `Protopattern`       | `Protostructure`  |
| Geometrical class       | `Patterntype`        | `Structuretype`   |
| Exact geometry          | `CrystalPattern`     | `Structure`       |

To cover a couple of other related names and explain how they differ:

* **Isopointal structures**: account for the spacegroup (with or without distinguishing enantiomorphic groups) and the same complete Wyckoff sequence, including repeated independent occupations but excluding the chemical identity of the occupying atoms.
* **Isoconfigurational structures**: isopointal structures further distinguished by geometrical classification (i.e., similarities in atomic coordinates and cell parameters, but sometimes with further requirements), however still described by colorless occupation.
* **AFLOW labels*: very similar to httk protopatterns, except occupations are ordered by occupying element symbol, making the AFLOW labels different for, e.g., ZrO₂ (A2B_oP12_29_2a_a) and FeS₂ (AB2_oP12_29_a_2a).
* **Extended AFLOW labels**: an extension of three digits is added to describe geometrical similarity (-001, -002, …), bringing them closer to our pattern types.
* **Decorated AFLOW labels**: AFLOW labels are sometimes used with a “decoration” to indicate chemical species, although there does not seem to be one universal endorsed format. These decorations take AFLOW labels closer to httk protostructures.
