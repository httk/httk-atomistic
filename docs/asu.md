# Asymmetric units

A crystal is mostly repetition. `ASUStructure` represents the part that is not:
a space group, one site per symmetry orbit, and the values of that site's free
parameters. Expanding it produces the full unit cell; recognizing a full cell
produces it back.

```python
from httk.atomistic import UnitcellStructureView
from httk.core import load

asu = load("nacl.cif")                     # a CIF's native, declared symmetry
structure = UnitcellStructureView(asu)     # the full cell, exactly
```

`ASUStructure` is part of `StructureLike`, so it can be passed anywhere a
structure is accepted, and expansion is exact and lazy — reading the space
group never generates the cell.

## Canonicalization

Two crystals that are the same up to origin, cell-basis choice, site order, or
setting have many `ASUStructure` descriptions. `canonicalize` searches upward
from a declared exact `ASUStructure`, then chooses the terminal by anonymous
Wyckoff occupation before chemical assignment:

```python
from httk.atomistic import canonicalize

result = canonicalize(asu)          # declared exact ASU, highest exact lift
canonical = result.asu
```

An exact P1 supercell — any multiplicity, diagonal or sheared — is collapsed to
its unique primitive description before the search, so `canonicalize` returns the
same answer whichever cell you hand it. That collapse fires only on exact
rational invariance; a *noisy* supercell whose copies merely nearly coincide is
snapped instead by `canonical_asu` below, within its tolerance.

The returned `LiftResult` also carries the child-first subgroup path, final-hop
origin shift, and largest accepted residual from the upward search. The ASU is
authoritative because state normal forms and the terminal canonicalization are
not encoded in that route metadata. Use `canonicalize_legacy` when continuing a
workflow pinned to the earlier metric-first convention.

The 11 enantiomorphic space-group pairs (76/78, 91/95, 92/96, 144/145, 151/153,
152/154, 169/170, 171/172, 178/179, 180/181, 212/213) describe the same crystal
in two mirror-image handednesses. By default `canonicalize` and `canonical_asu`
**preserve chirality**: they keep the recognized group, so a genuinely chiral
crystal retains its handedness and the two partners stay distinct. Pass
`preserve_chirality=False` to instead normalize a result in the higher-numbered
member to its lower-numbered partner through an exact improper transformation
(fractional coordinates `f → (−f) mod 1` with the cell basis unchanged — the
Cartesian inversion), so the pair maps to one canonical representative and the two
partners' canonical labels coincide. (One edge case of that normalize path: a
*hand-built exact* ASU already described in a left-handed cell of an enantiomorphic
group hits the pre-existing chirality-preserving orientation ceiling — its
left-handed cell is kept — so it still yields a distinct mirror representative;
recognition-path inputs, whose spglib cells are right-handed, are unaffected.) The
standalone `normalize_chirality` applies that same exact map to an already-preserved
result, so a caller need not re-canonicalize to get both forms. The four
classification-family canonicalizers expose the same policy directly. The
`canonicalize` and `canonical_asu` entry points reject site moments;
`normalize_chirality` returns a moment-bearing input unchanged because an axial
moment does not transform trivially under an improper map. The explicit-target
functions `canonicalize_full`, `list_representations`, and `rerepresent` honor
their target group exactly and are unaffected.

For *measured* input, use `canonical_asu`. It first puts the unit cell in a
deterministic anonymous P1 frame, recognizes symmetry within a Cartesian
tolerance with spglib, and canonicalizes the accepted model exactly:

```python
from httk.atomistic import canonical_asu
from httk.core import load

asu = canonical_asu(load("measured.cif"))   # noisy input, canonical answer
```

It sweeps tolerance multiples from loosest to tightest and accepts the first
recognized model whose expanded atoms admit a bijective same-species fit within
the base tolerance. It always works from coordinates, so it can raise a declared
symmetry the geometry supports or lower one it does not support at that
tolerance.

By default (`lift=False`) it returns the canonical representative of the
*recognized* symmetry and does not hunt for pseudosymmetry above what recognition
found. Its exact terminal removes the tabulated representational freedom and is
deterministic for supported rational crystallographic metrics. A non-rational
metric whose Cartesian factor needs unsupported nested radicals can retain its
input global rotation, as detailed in {doc}`canonicalization`. The default is
suited to bulk sweeps; `lift=True` additionally runs the exact upward search for
higher symmetry the recognition missed. That search can be slow — minutes and
beyond for low-symmetry, many-atom cells.

The anonymous geometry is fixed before a final chemical tie is resolved. Use
`canonical_asu_protostructure_assignments` to retain every tied assignment of the
original species to that geometry. The explicit
`canonical_asu_protostructure` name remains an alias for the scalar convention.
`canonical_asu_legacy` selects the earlier metric-first convention.

Only recognition and its fit test use floating point. Free Wyckoff parameters
come from exact row-Hermite chart projection of the recognized coordinates, not
from a metric least-squares fit. Boundary decisions can still vary with the
floating-point platform or spglib build. See {doc}`canonicalization` for the
numbered wrapper and terminal algorithms, exact ordering keys, finite bounds,
table hashes, and tested spglib version.

## From the command line

The same operations are available as `httk symmetry`, taking a structure file
(CIF, POSCAR) and printing a human-readable report; `-o` saves the result.

```console
$ httk symmetry info nacl.cif other.cif                    # inspect one or more structures
$ httk symmetry info --recognize measured.poscar           # also recognize symmetry from the geometry
$ httk symmetry canonicalize -o out.cif nacl.cif           # save one canonical form
$ httk symmetry canonicalize --exact --out-dir canonical/ *.cif  # save a batch by input basename
$ httk symmetry representations --target 166 nacl.cif      # list distinct forms in a related group
```

`canonicalize` defaults to the tolerant `canonical_asu` path (`--lift` searches
upward for higher pseudosymmetry); `--exact` runs the exact `canonicalize` on
input that already carries declared symmetry. Both keep the recognized group by
default (preserving chirality); `--normalize-chirality` maps an enantiomorphic pair
to its lower-numbered member. `rerepresent --target N` re-expresses
one crystal in a reachable group. Every subcommand accepts `--tolerance X` (a
Cartesian distance) and reports operator errors — a missing spglib, an unrelated
target — to stderr with a nonzero exit.

The full guide, {doc}`details/asu`, covers what an asymmetric unit holds,
arbitrary and untabulated settings, the exactness contract of expansion,
tolerance-bearing recognition, round-tripping, reading CIFs, serving symmetry
over OPTIMADE, and the symmetry tables.
