# Vendored atomistic schemas

This package contains the authoritative, supported schemas served by
*httk-atomistic*, registered by `httk.registry.schemas.atomistic` in httk-core's
IRI registries.

## OPTIMADE `structures` entry-type

`structures.json` is the complete OPTIMADE v1.3 `structures` entry-type
definition: its properties include their canonical `$id`s, types, units,
requirements, and descriptions. It mixes v1.2 property `$id`s with the v1.3
properties `fractional_site_positions`, `wyckoff_positions`,
`site_coordinate_span`, `site_coordinate_span_description`, and
`optimization_type`.

The checked-in file is the source of truth. It is loaded with
`httk.core.load_entry_type_definition` and served by
`httk.atomistic.StructureEntryProvider`. The `references`, `files`, and
`calculations` standard entry types are vendored by *httk-core*, not here.
The JSON and license files are packaged through the
`httk.registry.schemas.atomistic` package-data entry in `pyproject.toml`.

## Published httk property definitions

The remaining nine JSON files are the authoritative, supported copies of the
non-OPTIMADE-standard property definitions that *httk-atomistic* serves. They
describe which setting a crystal structure is written in, the change of basis
from the International Tables standard setting to it, how precisely the
structure's numbers were stated, and the magnetic moments of its sites.

They are published definitions taken verbatim from
[schemas.httk.org](https://schemas.httk.org), not local paraphrases. A client
following a property's `$id` therefore reaches the authoritative schema.

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
| `_httk_site_moments` | `site_moments.json` | `…/properties/magnetism/site_moments` |

All are under `https://schemas.httk.org/defs/v0.1/properties/`.

### Served name and definition identity

OPTIMADE requires a database-specific property to carry a registered prefix in
its **name**, which is why these are served as `_httk_…`. The definition itself
does not need to be restated locally: each document is byte-identical to the
published one, including `$id`. The prefix says “this database chose to serve
this”, while the `$id` says “this is what it means, as defined elsewhere”.

The definition's own `x-optimade-definition.name` remains the unprefixed name
it was published under (`setting_it_nc`), not the prefixed name httk serves it
as. The returned `PropertyDefinition.name` is nevertheless the served name;
rewriting the published payload would defeat the point of pointing at it.

The definitions are loaded at runtime by
`httk.atomistic.entries.symmetry.setting_definitions()`,
`httk.atomistic.entries.precision.precision_definitions()`, and
`httk.atomistic.entries.moments.moment_definitions()`, then merged into the served entry-type
definition by `httk.atomistic.StructureEntryProvider`. They are packaged through
the same `httk.registry.schemas.atomistic` package-data entry.

## What is not here

The standard OPTIMADE symmetry properties — `space_group_it_number`, the H-M
and Hall symbols, `space_group_symmetry_operations_xyz`, `wyckoff_positions`,
`fractional_site_positions`, and `site_coordinate_span` — need no property
definition of their own. They are already described by `structures.json`.

Nor is there anything standard about precision: the only
`standard_uncertainty` in that file is CODATA metadata inside a unit definition,
describing a physical constant rather than a queryable property.

## Provenance

The OPTIMADE source repository is
<https://github.com/Materials-Consortia/schemas>.

| File | Version | Source URL |
| --- | --- | --- |
| `structures.json` | v1.3 (30 properties) | <https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures.json> |

The published httk schemas are at <https://github.com/httk/schemas>, served from
<https://schemas.httk.org>. Their sources are
<https://github.com/httk/schemas-sources>; they are built with the
Materials-Consortia `optimade-property-tools`. These definitions are the v0.1
set, copied from `defs/v0.1/properties/` without modification.

The two precision definitions were authored for this purpose. Their source YAML
lives at
`src/defs/v0.1/properties/core/{fractional_coordinate_precision,length_precision}.yaml`
in the sources repository and is built with `make schemas`.

## License

The OPTIMADE definition is distributed under the MIT License; see
[`LICENSE`](./LICENSE), fetched from
<https://raw.githubusercontent.com/Materials-Consortia/schemas/master/LICENSE>.
The httk property schemas are also MIT licensed; see
[`LICENSE.httk`](./LICENSE.httk). This differs from the httk source code, which
is AGPL, and from the CC BY 4.0 symmetry datasets under `httk.atomistic/data/`.

## Refreshing

Run `make optimade-defs` from the repository root to re-fetch `structures.json`
and the OPTIMADE `LICENSE`, and `make httk-defs` to re-fetch the nine httk
property definitions from <https://schemas.httk.org> (the `HTTK_DEFS` list in
the Makefile names them as `<section>/<name>`; basenames are kept). These are
the only source tasks that use the network; ordinary builds and tests read the
committed copies offline. After a refresh, run `make test`
(`tests/test_symmetry_entries.py` asserts that each definition still carries a
`schemas.httk.org` `$id`), review the diff, and re-commit only intended version
changes.
