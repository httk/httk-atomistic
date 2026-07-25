# Vendored anyterial property definitions

This directory holds the authoritative, supported copies of the **non-OPTIMADE-standard**
property definitions that *httk-atomistic* serves. They describe two things OPTIMADE does
not standardise: *which setting* a crystal structure is written in, together with the change
of basis from the International Tables standard setting to it; and how precisely the
structure's numbers were stated by whatever source they came from.

They are not written here. They are published definitions taken verbatim from
[schemas.anyterial.se](https://schemas.anyterial.se), and vendoring them rather than
paraphrasing them is deliberate: a client that follows a property's `$id` reaches the
authoritative schema instead of a local restatement that could drift from it.

| Served as | Definition | `$id` |
| --- | --- | --- |
| `_httk_setting_it_nc` | `setting_it_nc.json` | `…/properties/spacegroups/setting_it_nc` |
| `_httk_hall_entry` | `hall_entry.json` | `…/properties/spacegroups/hall_entry` |
| `_httk_is_reference_setting` | `is_reference_setting.json` | `…/properties/spacegroups/is_reference_setting` |
| `_httk_crystal_system` | `crystal_system.json` | `…/properties/pointgroups/crystal_system` |
| `_httk_centring_type` | `centring_type.json` | `…/properties/spacegroups/centring_type` |
| `_httk_setting_transform` | `affine_transformation.json` | `…/properties/symmetry/affine_transformation` |
| `_httk_coordinate_precision` | `fractional_coordinate_precision.json` | `…/properties/core/fractional_coordinate_precision` |
| `_httk_basis_precision` | `length_precision.json` | `…/properties/core/length_precision` |

all under `https://schemas.anyterial.se/defs/v0.1/properties/`.

## The name and the definition are two different things

OPTIMADE requires a database-specific property to carry a registered prefix in its **name**,
which is why these are served as `_httk_…`. It does not require the *definition* to be
locally authored, and these are not: each document is byte-identical to the published one,
`$id` included. So the prefix says "this database chose to serve this", while the `$id` says
"and this is what it means, as defined elsewhere".

One visible consequence: a definition's own `x-optimade-definition.name` is the unprefixed
name it was published under (`setting_it_nc`), not the prefixed name httk serves it as.
That is correct and is left alone — rewriting a published document to match a local naming
choice would defeat the point of pointing at it.

They are loaded at runtime by `httk.atomistic.anyterial_definitions()` (packaged through
`pyproject.toml`'s package-data entry `"httk.atomistic" = [..., "anyterial_defs/*"]`) and
merged into the served entry-type definition by
`httk.atomistic.StructureEntryProvider`.

## What is *not* here

The standard OPTIMADE symmetry properties — `space_group_it_number`, the H-M and Hall
symbols, `space_group_symmetry_operations_xyz`, `wyckoff_positions`,
`fractional_site_positions`, `site_coordinate_span` — need no definition from here. They are
part of the OPTIMADE standard and are already described by the vendored
`../optimade_defs/structures.json`; they were simply never served before.

Nor is there anything standard about precision: the only `standard_uncertainty` anywhere
in that file is CODATA metadata inside a unit definition, describing a physical constant
rather than a queryable property.

## Provenance

Source repository: <https://github.com/httk/anyterial-schemas> (the
`anyterial-schemas-source` checkout), generated with the Materials-Consortia
`optimade-property-tools`. Definitions are the `v0.1` set.

Copied from `output/defs/v0.1/properties/` without modification.

The two precision definitions were authored for this purpose; their source YAML lives at
`src/defs/v0.1/properties/core/{fractional_coordinate_precision,length_precision}.yaml`
in that repository and is built with `make schemas`.

## License

MIT, © 2024 Open Databases Integration for Materials Design; see the adjacent
[`LICENSE`](./LICENSE). This differs from the httk source code, which is AGPL — see the
repository root — and from the CC BY 4.0 symmetry *datasets* under `../data/`.

## Refreshing

Copy the wanted files from a checkout's `output/defs/v0.1/properties/<section>/<name>.json`
into this directory, keeping the basename. After a refresh, review the diff, re-run
`make test` (`tests/test_symmetry_entries.py` asserts each definition still carries an
`schemas.anyterial.se` `$id`), and re-commit only intended version changes.
