# Serving structures as OPTIMADE, with derived fields and custom properties

`StructureEntryProvider` is the adapter that turns a plain `{id: UnitcellStructure}`
mapping into something an OPTIMADE server can query. It implements the neutral
`httk.core.EntryProvider` contract — `entry_types()`, `property_keys()`,
`records()` — and nothing about it is specific to any particular server. That is
the point of the seam: *httk-atomistic* has the data, *httk-serve* serves it,
and neither imports the other. They meet at the contract.

**What the provider derives for you.** Beyond the structural fields it reads
straight off the `UnitcellStructure` (`lattice_vectors`, `cartesian_site_positions`,
`species`, `species_at_sites`, `nsites`, `elements`, `nelements`), it computes
the standard OPTIMADE composition fields:

- `chemical_formula_reduced` — element counts divided by their gcd, elements in
  alphabetical order. Four Fe, twelve O and four Sm reduce by 4 to `FeO3Sm`.
- `chemical_formula_anonymous` — the same stoichiometry with the elements
  replaced by `A`, `B`, `C`, ... assigned by *descending* amount, so the same
  formula gives `A3BC`.
- `elements_ratios` — each element's fraction of the sites, ordered to match
  `elements`.
- `nperiodic_dimensions` and `dimension_types` — 3 and `[1, 1, 1]`, since a
  `UnitcellStructure` is a fully periodic crystal.
- `structure_features` — the OPTIMADE flag list. A site with mixed occupancy
  makes this `["disorder"]`; a species with attached particles adds
  `"site_attachments"`; a fully ordered structure gets `[]`.

**Limits of derivation.** The composition fields are only
meaningful when every species is a single, unattached element — there is no
sensible integer count of Fe in a site that is half Fe and half Ni. So for a
disordered structure the provider serves `null` for those derived fields rather
than inventing a number, while still serving `elements`, `nelements`, `nsites`,
and the honest `structure_features: ["disorder"]`. Compare the two entries in
the output below; this asymmetry is the interesting part.

**Entries with no structure.** Passing `None` for an id is legal and means "this
entry exists, but I have no structure for it". Every structural and derived
field then serves `null`. It keeps a known id queryable instead of making it
vanish.

**Custom properties.** A database usually has more to say than the standard
fields — a total energy, a provenance tag. Pass a `PropertyDefinition` per
custom name in `extra_definitions` (the name needs a registered database prefix,
hence the leading `_httk_`), and the per-entry values in `properties`. The
provider validates that every value you supply is described by the extended
definition, so a typo is an error at construction time rather than a silently
missing field at query time.

The last section runs an actual query: `adapter_from_providers` wraps the
provider for *httk-serve*'s backend, and `execute_query` runs a parsed
OPTIMADE filter against it. *httk-serve* is an optional peer distribution,
not a dependency.

```{literalinclude} ../../examples/serve_structures_as_optimade.py
:language: python
:lines: 54-
```
