# Vendored crystallographic symmetry datasets

This directory holds the authoritative, supported copies of the symmetry data that
*httk-atomistic* uses to build and expand asymmetric-unit structure representations
(`httk.atomistic.ASUStructure`). The checked-in files are the source of truth:
httk-atomistic supports exactly these versions.

They are loaded at runtime through `httk.core.DatasetLoader` via the accessors in
`httk.atomistic.data` (packaged through `pyproject.toml`'s package-data entry
`"httk.atomistic" = [..., "data/*"]`). Nothing is read at import time; the first lookup
triggers the load.

| File | Size | Contents |
| --- | --- | --- |
| `symmetry_basics.json.gz` | 621 KB | 527 space-group **setting** records (230 of them reference settings) + the 32 point groups |
| `spacegroup_setting_transforms.json.gz` | 8 KB | change-of-basis operation from each setting to its IT standard setting, for all 527 |
| `spacegroup_subgroups.json.gz` | 316 KB | per-IT-number Bärnighausen subgroup transformations and continuous-normalizer bases, for all 230 |
| `affine_normalizer_cosets.json.gz` | 41 KB | affine-normalizer cosets for all 527 Hall entries |

## What is in `symmetry_basics.json.gz`

One record per space-group *setting*, not per space-group *type* — so all 527 settings,
of which exactly 230 carry `is_reference_setting: true`. Each record is **self-contained
in its own setting**: its `symops`, its `wyckoff` table, and its asymmetric-unit region
`asu`/`asu_str` are expressed in that setting's own coordinates. SG 15 Wyckoff letter `e`
is `0,y,1/4` in the reference setting `15:b1` but `1/4,0,z` in `15:c1`.

All numeric values are exact rational **strings** (`"1/8"`, `"-1"`), never floats, so
everything downstream stays exact.

Two properties this package relies on, both verified in `tests/test_symmetry_data.py`:

- `len(wyckoff[i].orbit) == multiplicity` for every entry, with centering translations
  already folded in and the orbit already deduplicated. Expansion is therefore a plain
  affine loop with no coincidence testing and no tolerance.
- `sum(hasfreedom) == rank(orbit[k].matrix)` for every orbit member, with the non-free
  columns identically zero. So `hasfreedom` alone selects the free parameters, and the
  `first_orbit` strings never need parsing.

Two field-choice traps, both of which fail *silently*:

- Use `symops` and `orbit`, not `symops_mod_centering` and `orbit_mod_centering`. The
  latter are the centering-factored forms; mixing the two in a set comparison misreports
  every centred group.
- `orbit[0]` follows `first_orbit`, which differs from `first_orbit_ita` in 180 of the
  3440 Wyckoff entries. Both describe the same orbit, but only `first_orbit_ita` matches
  what International Tables prints.

## What is in `spacegroup_setting_transforms.json.gz`

One record per Hall entry, covering all 527 settings, each holding an exact rational
`affine_transformation` with a `matrix` `M` and `vector` `v`.

**The direction is the opposite of what the upstream field name suggests.** The pair maps
IT-standard coordinates *into* the setting's own coordinates:

```
x_own = M @ x_std + v                 # column-vector form
f_own = f_std @ M.T + v               # httk row-vector form
f_std = (f_own - v) @ inv(M).T        # reverse
B_own = inv(M).T @ B_std              # cell basis rows
```

`det M == 1` for 520 settings and `det M == 3` for the seven rhombohedral-axes settings
(IT numbers 146, 148, 155, 160, 161, 166, 167), where the standard hexagonal cell has
three times the volume of the rhombohedral one. `inv(M)` therefore has thirds for those,
and no code may assume the reverse transform is integral.

Note also that the IT standard setting is **not** spglib's default setting for the 24
space groups with two origin choices (48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130,
133, 134, 137, 138, 141, 142, 201, 203, 222, 224, 227, 228); they agree for the other 206.

## Provenance

Source repository: <https://github.com/httk/data-generator>

| File | Version | Provenance |
| --- | --- | --- |
| `symmetry_basics.json.gz` | 0.1.0 | data-generators commit `de1f495b9e9231c8223cb20423f0d8b69b376a55`, copied byte-for-byte |
| `spacegroup_setting_transforms.json.gz` | 0.1.0 | subset of the same commit's `transformations_hm_entry` dataset |
| `spacegroup_subgroups.json.gz` | 0.1.0 | subset of the `transformations_std` dataset in the data-generators checkout used for this refresh |
| `affine_normalizer_cosets.json.gz` | 0.1.0 | `affine_normalizer_cosets` dataset in the data-generators checkout used for this refresh, copied byte-for-byte |

The transforms file is a **derived subset**, not an upstream artifact. Upstream
`transformations_hm_entry.json.gz` is 5.2 MB compressed but 133 MB decompressed and takes
about four seconds to parse, of which httk needs only the per-setting
`hall_to_it_std_transform` record. `tools/vendor_symmetry_data.py` extracts that one field
into a document of the same JSON-LD shape, carrying the source document's `@context`,
`creator`, `dcterms:license`, and `prov:wasGeneratedBy` header forward unchanged so the
attribution chain is unbroken. No values are altered. The `baernighausen` and
`continuous_normalizer` sections are now vendored in `spacegroup_subgroups.json.gz`.
The remaining sections of the `transformations_std.json.gz` source
(`same_space_group_affine_images_std`,
`isomorphic_subgroups`, `backward_lift_criteria`, `euclidean_normalizer`, and the full
`orthogonal_affine_normalizer`/`affine_normalizer` sections) remain deliberately
un-vendored: they are unused by the runtime and are 40+ MB raw.

## License

All four datasets are distributed under the Creative Commons Attribution 4.0 International
License (CC BY 4.0); see the adjacent [`LICENSE`](./LICENSE) for the required attribution.
This differs from the httk source code, which is AGPL — see the repository root.

## Refreshing

```sh
make symmetry-data DATA_GENERATORS=/path/to/data-generators
```

This copies `symmetry_basics.json.gz` and `affine_normalizer_cosets.json.gz`, and regenerates
the two slices. It is offline:
it reads a local data-generators checkout rather than the network, unlike `make
optimade-defs`. The slice is written with a fixed gzip mtime so the output is
byte-reproducible. After a refresh, review the diff and re-commit only intended version
changes — and re-run `make test`, since `tests/test_symmetry_data.py` asserts the record
counts and structural invariants documented above.
