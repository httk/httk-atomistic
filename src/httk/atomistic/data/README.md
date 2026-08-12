# Vendored crystallographic symmetry datasets

This directory holds the authoritative, supported copies of the symmetry data that
*httk-atomistic* uses to build and expand asymmetric-unit structure representations
(`httk.atomistic.ASUStructure`). The checked-in files are the source of truth:
httk-atomistic supports exactly these versions.

They are the five canonical per-concern datasets published by upstream
data-generators, stored as byte-for-byte `.json.gz` artifacts of the upstream JSON-LD
publications. `httk.core.DatasetLoader` reads them through the ordinary lazy dataset
accessors; the complete document is parsed on first lookup. Upstream also publishes
`.sqlar` twins for large-dataset/lazy-access use, but the compressed JSON artifacts are
appropriately sized for these five vendored datasets.

The files are loaded through the accessors in `httk.atomistic.data` and included in
the wheel and sdist as package data.

| File | Size | Contents |
| --- | --- | --- |
| `symmetry_basics.json.gz` | 621 KB | 527 space-group **setting** records (230 reference settings) + 32 point groups |
| `spacegroup_setting_transforms.json.gz` | 13 KB | change-of-basis operation from each setting to its IT standard setting, for all 527 |
| `baernighausen_std.json.gz` | 316 KB | per-IT-number Bärnighausen subgroup transformations, for all 230 |
| `continuous_euclidean_normalizer_std.json.gz` | 3 KB | per-IT-number continuous-normalizer bases, for all 230 |
| `affine_normalizer_cosets.json.gz` | 41 KB | affine-normalizer cosets for all 527 Hall entries |

The split-affine records map parent standard-setting coordinates directly to child
standard-setting coordinates, as pinned by `tests/test_subgroups.py`.
For an entry `affine_transformation` with matrix `M`, the basis convention is
`B_child = M.T @ B_parent`. The full entry affine map is child-to-parent,
`f_parent = f_child @ M.T + v`; the inverse matrix is the parent-to-child coordinate
basis change used by the split operations.

## What is in `symmetry_basics.json.gz`

One record per space-group *setting*, not per space-group *type* — 527 settings, of
which exactly 230 carry `is_reference_setting: true`. Each record is **self-contained
in its own setting**: its `symops`, its `wyckoff` table, and its asymmetric-unit region
`asu`/`asu_str` are expressed in that setting's own coordinates. SG 15 Wyckoff letter `e`
is `0,y,1/4` in the reference setting `15:b1` but `1/4,0,z` in `15:c1`.

All numeric values are exact rational **strings** (`"1/8"`, `"-1"`), never floats, so
everything downstream stays exact.

Two properties this package relies on, both verified in `tests/test_symmetry_data.py`:

- `len(wyckoff[i].orbit) == multiplicity` for every entry, with centering translations
  already folded in and the orbit already deduplicated. Expansion is therefore a plain
  affine loop with no coincidence testing and no tolerance.
- `sum(hasfreedom) == rank(orbit[k].matrix)` for every orbit member, with non-free columns
  identically zero. So `hasfreedom` alone selects the free parameters, and the
  `first_orbit` strings never need parsing.

Two field-choice traps, both of which fail **silently**:

- Use `symops` and `orbit`, not `symops_mod_centering` and `orbit_mod_centering`. The
  latter are the centering-factored forms; mixing the two in a set comparison misreports
  every centred group.
- `orbit[0]` follows `first_orbit`, which differs from `first_orbit_ita` in 180 of the
  3440 Wyckoff entries. Both describe the same orbit, but only `first_orbit_ita` matches
  what International Tables prints.

## Setting transforms and subgroup datasets

`spacegroup_setting_transforms.json.gz` has one record per Hall entry, covering all 527
settings, each holding an exact rational `affine_transformation` with a `matrix` `M`
and `vector` `v`.

**The direction is the opposite of what the upstream field name suggests.** The pair maps
IT-standard coordinates *into* the setting's own coordinates:

```
x_own = M @ x_std + v                 # column-vector form
f_own = f_std @ M.T + v               # httk row-vector form
f_std = (f_own - v) @ inv(M).T        # reverse
B_own = inv(M).T @ B_std              # cell basis rows
```

`baernighausen_std.json.gz` and `continuous_euclidean_normalizer_std.json.gz` are
separate canonical datasets keyed by IT number. `spacegroup_subgroup_record()` composes
their two sections into the compatibility record used by the symmetry code; it does not
copy or normalize their exact values.

## Provenance and license

Source repository: <https://github.com/httk/data-generator>. Each file preserves its
upstream JSON-LD attribution and provenance header; all five are version `0.1.0` and
licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
See the adjacent [`LICENSE`](./LICENSE) for the required attribution. This differs from
the httk source code, which is AGPL — see the repository root.

The old vendored JSON slices are replaced by the canonical upstream per-concern
artifacts; no derived documents are generated here.

## Refreshing

```sh
make symmetry-data DATA_GENERATORS=/path/to/data-generators
```

This copies the five canonical `.json.gz` files from the local data-generators checkout's
`data/` directory, byte-for-byte, and is offline. After a refresh, review the diff and
re-run `tests/test_symmetry_data.py`, which asserts the record counts and structural
invariants documented above.
