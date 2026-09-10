# Subgroups and pathfinding

Space groups form a lattice of group–subgroup relations (the Bärnighausen tree).
*httk-atomistic* tabulates that graph and can move an exact asymmetric unit along
it — descending into a subgroup, lifting into a supergroup, or aligning two
crystals in a shared setting — all in exact rational arithmetic.

The tabulated relations need only an IT number:

```python
from httk.atomistic import maximal_subgroups, minimal_supergroups, subgroup_closure

maximal_subgroups(225)          # (139, 166, 202, 209, 216, 221, 224) — one hop down from Fm-3m
minimal_supergroups(166)        # one hop up
len(subgroup_closure(225))      # every group reachable downward, transitively
```

## Representing one crystal in a related group

Given a recognized `ASUStructure`, `subgroup_representation` re-expresses it
exactly in a subgroup's IT standard setting, and `rerepresent` moves it to any
reachable subgroup *or* supergroup setting:

```python
from httk.atomistic import recognize_asu, subgroup_representation, rerepresent, UnitcellStructure

structure = UnitcellStructure(
    cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    species=[
        {"name": "Cs", "chemical_symbols": ["Cs"], "concentration": [1.0]},
        {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
    ],
    species_at_sites=["Cs", "Cl"],
)
asu = recognize_asu(structure)                     # Pm-3m (221)
in_subgroup = subgroup_representation(asu, 166)     # result object; .asu is the exact ASU in R-3m's setting
same = rerepresent(asu, 166)                        # returns the ASU directly
```

A group's standard setting can still hold one crystal several ways (the
normalizer orbit). `list_representations(asu, target)` enumerates every distinct
one; `canonicalize_full(asu, target)` returns the single canonically-least
representative in that target group's standard setting.

## Lifting to higher symmetry

`lift_candidates` returns the one-hop parent lifts of an ASU; `backward_lift`
returns every exact or tolerance-accepted lift into one named supergroup; and
`highest_symmetry` runs the full breadth-first search and returns the terminal
lifts. By default it returns one representative per reached maximal group; pass
`all_paths=True` to enumerate the distinct *routes* to them (keyed on the visited
set), which matters when several descent paths reach the same group differently.

```console
>>> from httk.atomistic import highest_symmetry
>>> tops = highest_symmetry(asu)                 # highest (pseudo)symmetry reachable upward
>>> tops = highest_symmetry(asu, all_paths=True) # every distinct route, not just endpoints
```

## Aligning two crystals

Two different functions bring *two* structures together — they are distinct
operations, not aliases:

- `represent_like(structure, reference)` aligns one structure to a *reference*'s
  group and setting via a normalizer-coset search — the two must already share a
  group. Root-exported.
- `common_subgroup_representation(first, second)` puts two structures into their
  highest *common* subgroup, a shared basis in which both are expressible even
  when their groups differ. It lives in `httk.atomistic.symmetry.paths` (import
  `from httk.atomistic.symmetry.paths import common_subgroup_representation`), not
  the root namespace.

Once two asymmetric units are aligned in a common group, `interpolate_structures`
builds an exact, symmetry-preserving linear path between them, returned as a
`StructurePath` of frames:

```console
>>> from httk.atomistic import interpolate_structures
>>> path = interpolate_structures(start, end, steps=5)   # start/end aligned in a shared group
>>> len(path.frames)
```

The same operations are available from the command line as `httk symmetry` — see
{doc}`asu`.
