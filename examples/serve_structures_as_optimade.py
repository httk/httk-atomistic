"""Serving structures as OPTIMADE, with derived fields and custom properties

`StructureEntryProvider` is the adapter that turns a plain `{id: Structure}`
mapping into something an OPTIMADE server can query. It implements the neutral
`httk.core.EntryProvider` contract — `entry_types()`, `property_keys()`,
`records()` — and nothing about it is specific to any particular server. That is
the point of the seam: *httk-atomistic* has the data, *httk-optimade* serves it,
and neither imports the other. They meet at the contract.

**What the provider derives for you.** Beyond the structural fields it reads
straight off the `Structure` (`lattice_vectors`, `cartesian_site_positions`,
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
  `Structure` is a fully periodic crystal.
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
provider for *httk-optimade*'s backend, and `execute_query` runs a parsed
OPTIMADE filter against it. *httk-optimade* is an optional peer distribution,
not a dependency.
"""

from httk.core import PropertyDefinition

# pyright: reportMissingImports=false
# httk-optimade is an optional peer distribution, so it is not guaranteed to be
# installed where this repository's type checks run; see HTTK_EXAMPLE_REQUIRES below.
from httk.optimade import adapter_from_providers, parse_optimade_filter
from httk.optimade.backend import execute_query

from httk.atomistic import Species, Structure, StructureEntryProvider

#: This example needs the optional peer distribution *httk-optimade* on the path.
HTTK_EXAMPLE_REQUIRES = ["httk.optimade"]

#: An ordered SmFeO3-like cell: 4 Fe, 12 O, 4 Sm on twenty sites.
FE = Species(name="Fe", chemical_symbols=("Fe",), concentration=(1.0,))
OXYGEN = Species(name="O", chemical_symbols=("O",), concentration=(1.0,))
SM = Species(name="Sm", chemical_symbols=("Sm",), concentration=(1.0,))
#: A single site shared 50/50 between Fe and Ni -- a substitutional alloy.
FE_NI = Species(name="M", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))


def ordered_smfeo3() -> Structure:
    cell = [[5.6, 0.0, 0.0], [0.0, 7.6, 0.0], [0.0, 0.0, 5.3]]
    sites = [[0.01 * index, 0.0, 0.0] for index in range(20)]
    return Structure(cell, sites, [FE, OXYGEN, SM], ["Fe"] * 4 + ["O"] * 12 + ["Sm"] * 4)


def disordered_alloy() -> Structure:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    return Structure(cell, [[0.0, 0.0, 0.0]], [FE_NI], ["M"])


def build_provider() -> StructureEntryProvider:
    """A provider over three entries: one ordered, one disordered, and one with no structure."""
    total_energy = PropertyDefinition.from_simple(
        "_httk_total_energy",
        description="Total energy of the calculation, in eV.",
        fulltype="float",
    )
    return StructureEntryProvider(
        {
            "smfeo3": ordered_smfeo3(),
            "feni-alloy": disordered_alloy(),
            "known-but-empty": None,
        },
        extra_definitions={"_httk_total_energy": total_energy},
        properties={"smfeo3": {"_httk_total_energy": -12.5}},
    )


DERIVED_FIELDS = (
    "chemical_formula_reduced",
    "chemical_formula_anonymous",
    "chemical_formula_descriptive",
    "elements",
    "elements_ratios",
    "nelements",
    "nsites",
    "nperiodic_dimensions",
    "dimension_types",
    "structure_features",
    "_httk_total_energy",
)


def show_records(provider: StructureEntryProvider) -> None:
    """Print what the provider serves, entry by entry."""
    print("Served property names:")
    print("  ", sorted(provider.property_keys("structures")))
    print("  ('id' is served from the record key", repr(provider.property_keys("structures")["id"]) + ")")
    print()

    records = {record["__id"]: record for record in provider.records("structures")}
    for entry_id in ("smfeo3", "feni-alloy", "known-but-empty"):
        record = records[entry_id]
        print(f"{entry_id}:")
        for field in DERIVED_FIELDS:
            print(f"  {field:<28} {record[field]!r}")
        print()


def show_definition(provider: StructureEntryProvider) -> None:
    """The entry-type definition is the schema; it describes more than the provider serves."""
    definition = provider.entry_types()["structures"]
    described = definition.properties
    served = set(provider.property_keys("structures"))
    print("Entry-type definition:")
    print("  properties described by the (extended) definition:", len(described))
    print("  properties actually served:                       ", len(served))
    print("  described but not served (a few):", sorted(set(described) - served)[:5])
    print("  the custom property is described:", "_httk_total_energy" in described)
    print("  ...with description:", described["_httk_total_energy"].description)
    print()


def run_queries(provider: StructureEntryProvider) -> None:
    """Hand the provider to httk-optimade and run real OPTIMADE filters against it."""
    adapter = adapter_from_providers([provider])
    print("Queries through httk-optimade:")
    print("  entry types served:", adapter.schema.all_entries)

    def query(filter_string: str | None, fields: list[str]) -> list[dict[str, object]]:
        filter_ast = parse_optimade_filter(filter_string) if filter_string else None
        results = execute_query(adapter, ["structures"], ["id", *fields], [], 100, 0, filter_ast)
        return [result.values for result in results]

    everything = query(None, [])
    print("  all ids:", sorted(str(row["id"]) for row in everything))

    for filter_string, fields in (
        ('elements HAS "Sm"', ["chemical_formula_reduced", "chemical_formula_anonymous", "elements_ratios"]),
        ("nelements = 2", ["structure_features", "elements"]),
        ("nsites > 10", ["nsites", "nperiodic_dimensions"]),
    ):
        rows = query(filter_string, fields)
        print(f"  {filter_string!r} ->")
        for row in rows:
            print("    ", {key: row[key] for key in ("id", *fields)})


def main() -> None:
    provider = build_provider()
    show_records(provider)
    show_definition(provider)
    run_queries(provider)


if __name__ == "__main__":
    main()
