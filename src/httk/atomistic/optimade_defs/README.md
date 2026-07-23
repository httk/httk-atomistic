# Vendored OPTIMADE `structures` entry-type definition

This directory holds the authoritative, supported copy of the OPTIMADE standard
`structures` *entry-type definition* document that *httk-atomistic* serves. The
JSON file embeds the complete property definitions for the `structures` entry
type (their canonical `$id`s, types, units, requirements, and descriptions).

The checked-in file is the source of truth: httk-atomistic supports exactly this
version. It is loaded at runtime via
`httk.core.load_entry_type_definition("httk.atomistic", "structures")` (packaged
through `pyproject.toml`'s package-data entry
`"httk.atomistic" = [..., "optimade_defs/*"]`) and served by
`httk.atomistic.StructureEntryProvider`.

`structures.json` mixes v1.2 property `$id`s (the 25 unchanged properties) with
five v1.3-native ones (`fractional_site_positions`, `wyckoff_positions`,
`site_coordinate_span`, `site_coordinate_span_description`, `optimization_type`).

## Provenance

Source repository: <https://github.com/Materials-Consortia/schemas>

Fetched from:

| File | Version | Source URL |
| --- | --- | --- |
| `structures.json` | v1.3 (30 properties) | <https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures.json> |

(The `references`, `files`, and `calculations` standards are vendored by
*httk-core*, not here.)

## License

This definition is distributed by the Materials-Consortia under the MIT License;
see the adjacent [`LICENSE`](./LICENSE) file, fetched from
<https://raw.githubusercontent.com/Materials-Consortia/schemas/master/LICENSE>.

## Refreshing

Run `make optimade-defs` from the repository root to re-fetch this file (and the
`LICENSE`). This is the only source task that uses the network; ordinary builds
and tests read the committed copy offline. After a refresh, review the diff and
re-commit only intended version changes.
