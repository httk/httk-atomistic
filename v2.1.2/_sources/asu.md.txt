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
setting have many `ASUStructure` descriptions. `canonicalize` collapses that
freedom exactly: given an `ASUStructure` it returns the single deterministic,
highest-symmetry representative, using only exact rational arithmetic.

```python
from httk.atomistic import canonicalize

result = canonicalize(asu)          # exact input, exact answer
canonical = result.asu
```

An exact P1 supercell — any multiplicity, diagonal or sheared — is collapsed to
its unique primitive description before the search, so `canonicalize` returns the
same answer whichever cell you hand it. That collapse fires only on exact
rational invariance; a *noisy* supercell whose copies merely nearly coincide is
snapped instead by `canonical_asu` below, within its tolerance.

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
result, so a caller need not re-canonicalize to get both forms — and this is exactly
how canonical prototypes and protostructures, which deliberately ignore chirality,
are derived from `canonical_asu(preserve_chirality=False)`. Magnetic structures (any
site carrying a moment) are never flipped — an axial moment does not transform
trivially under an improper map — and are left in their own group regardless. The
explicit-target functions `canonicalize_full`, `list_representations`, and
`rerepresent` honor their target group exactly and are unaffected.

For *measured* input — coordinates carrying noise — use `canonical_asu`, the
one-liner that recognizes the symmetry within a tolerance (with spglib) and then
canonicalizes the result exactly:

```python
from httk.atomistic import canonical_asu
from httk.core import load

asu = canonical_asu(load("measured.cif"))   # noisy input, canonical answer
```

It sweeps recognition over a few tolerance multiples and keeps the
highest-symmetry model whose atoms still sit within the base tolerance of the
input, so a boundary flip can be rescued without accepting extra noise. It always
works from the coordinates, so it can raise a declared symmetry the geometry
supports — or lower one it does not, at the derived tolerance.

By default (`lift=False`) it returns the canonical representative of the
*recognized* symmetry: fully deterministic, all representational freedom removed,
and cheap — the cost is essentially recognition. It does **not** hunt for
pseudosymmetry above what recognition found. Pass `lift=True` to additionally run
the exact upward search for higher symmetry the recognition missed; that is exact
too but can be slow — minutes and beyond for low-symmetry, many-atom cells. Use
the default for bulk sweeps over many structures; use `lift=True` when you are
specifically hunting the maximal (pseudo)symmetry of one crystal.

Only the recognition step is floating-point: which symmetry is *accepted* near a
tolerance boundary can vary across platforms or spglib builds, but the exact
canonicalization erases spglib's representational freedom, so *how* an accepted
symmetry is represented never does. Free-parameter values are least-squares fits
of the measured coordinates: two noisy measurements of the same crystal reach the
same Wyckoff choices but slightly different rational parameter values.

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
