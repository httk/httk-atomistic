# Canonicalization algorithms and reproducibility

This page specifies the canonicalization conventions used by *httk-atomistic*.
It is intended to be detailed enough for an independent implementation to
produce the same exact outputs when it uses the pinned tables and the same
tolerant-recognition environment.

## Public operations

`canonicalize` is the everyday operation. Its structure result is always an
`ASUStructure`. `canonical_asu` is the **same callable**, with the same options.
An ASU input does not silently select a different search from a full-cell input.

| Operation | Symmetry model | Result |
| --- | --- | --- |
| `canonicalize(structure)` | anonymous framing and spglib recognition | `ASUStructure` |
| `canonicalize(asu, symmetry="declared")` | supplied exact model; no recognition or upward search | `ASUStructure` |
| `canonicalize(classification)` | supplied classification and any carried example | matching classification value |
| `search_supergroups(asu)` | explicit upward Bärnighausen exploration | `SupergroupSearchResult` |
| `canonical_asu_legacy(structure)` | historical measured pipeline | legacy-convention `ASUStructure` |
| `canonicalize_legacy(asu)` | historical upward search and terminal | `LiftResult` |

```python
from httk.atomistic import canonicalize, search_supergroups

canonical = canonicalize(raw_structure)
declared = canonicalize(declared_asu, symmetry="declared")
search = search_supergroups(declared_asu, timeout=30, max_states=1000)
if search.complete and search.candidates:
    higher_symmetry = search.candidates[0].asu
```

Generic dispatch recognizes the `BareProtostructureBackend`, `BarePrototypeBackend`,
`ProtostructureBackend` and `PrototypeBackend` families, including their views.
It delegates to the corresponding explicit classification routine described below.
Strings, files and ordinary structure inputs follow structure recognition; they
are never guessed to be classification labels. Classification operations use
available occupations and transform refined examples, preserving discriminators.
`tolerance` and nondefault `factors` are rejected for classification and declared
ASU operations: those options belong to structure recognition.

The explicit `canonical_asu_protostructure` and assignment-family routines retain
the historical experimental signatures, including `lift`; their default result
uses the same measured convention. They are not identity aliases for the unified
API. For new upward-search calls use `search_supergroups`. `highest_symmetry`
retains its advanced legacy search contract. `canonicalize_full` belongs to
explicit-target representation/alignment and is not a more thorough default.

All exact ordering below is ascending Python tuple ordering unless stated
otherwise. Fractional coordinates are reduced componentwise modulo one into
`[0, 1)`. Matrices act on row cell bases as stated in each formula.

## Declared-symmetry canonicalization

`canonicalize(asu, symmetry="declared")` performs these steps:

1. Validate options and enter the cooperative deadline scope described below.
2. Require an already materialized `ASUStructure`. An ASU view with a retained
   or resolved exact ASU can be materialized without recognition; a view requiring
   recognition raises `TypeError`. No automatic spglib fallback is allowed.
3. Require full periodicity and reject moments, assemblies and molecular inputs.
4. Expand the declared ASU exactly to establish the anonymous frame and tied
   species assignments, using shared-core steps 1–9. Preserve the declared
   higher-group Wyckoff model; for P1 use the entire anonymous frame.
5. Apply the shared exact core below to every tied assignment. This includes
   exact translation/same-group cell reduction, not general upward exploration.
6. Choose by the bare assigned key (species class names included) first, then the
   anonymous geometry, restore the compatible chemical assignments, then choose
   the scalar chemical minimum. Restore metadata, and record the precision from
   the source by the derivation rule described below.
7. Check the deadline and any reported bounded-work skips before returning the
   ASU. On a limit, raise `CanonicalizationLimitError` without a partial answer.

## Explicit upward search

`search_supergroups` uses the following search; its terminal normalization is
shared with declared canonicalization. Its intermediate state keys still use
species names. The measured pipeline's anonymous invariance results do not
establish chemical-substitution covariance of these search states.

1. Require an `ASUStructure`, full three-dimensional periodicity, no site
   moments, no assemblies, and a non-molecular structure.
2. Transform the declared setting to the International Tables standard setting.
3. For P1, find every exact atom-set self-translation and collapse the generated
   translation lattice to a primitive cell. For other groups, repeatedly use
   exact same-group isomorphic lifts as described in step 11 of the shared core.
4. For groups 1 and 2, apply exact rational Niggli reduction. Then quotient the
   state by the search-state normal form: special-position demotion, continuous
   and discrete normalizer translations, finite affine-normalizer cosets, site
   representatives, and group-preserving handedness normalization. During the
   search this quotient retains only affine cosets whose `compatible_systems`
   annotation contains the state's current crystal system; P1 instead uses its
   exact rational Gram stabilizer.
5. Set the accepted Cartesian tolerance to the explicit float `tolerance`, or
   derive it from the normalized state's stated precision. Initialize a FIFO
   queue with `(state, path=(), shift=(0,0,0), residual=0)`.
6. Key a visited state by `(IT number, structure signature)`. The signature is
   the sorted tuple `(species name, Wyckoff letter, complete sorted exact orbit)`
   for every site followed by the nine exact Gram entries. `search_supergroups`
   adds the accumulated path to this key only when `all_paths=True`.
7. Pop the oldest state. Enumerate distinct minimal parent IT numbers in
   ascending numerical order. Within each parent, enumerate subgroup transforms in table order, then
   exact candidates by their canonical result key. Direct lifts are tried first.
   Normalizer retries are used for that parent only if direct lifting finds
   none. Conventional-cell rechoice is attempted only if the state has no lift
   into any parent.
8. A lift is retained only when its exact descent reproduces the child, or its
   tolerant descent admits the prescribed same-species Cartesian match. That
   fallback requires equal bases and counts, scans child points in their current
   order, and removes the first remaining same-species partner within tolerance;
   it is a deterministic greedy check. A modular-solver branch-cap failure skips
   that parent and emits a symmetry warning; it does not abort the remaining
   parents.
9. Normalize each lifted ASU as a search state. Append its child-first
   `SubgroupTransform` tuple to `path`, retain the current hop's `shift`, and set
   `residual=max(previous_residual, hop_residual)`. Enqueue the first occurrence
   of its visited key. Raise after 10,000 distinct visited states.
10. A state whose parent evaluation finished with no lift is terminal. Apply the
    shared anonymous exact core directly, using the requested chirality policy.
    Restore source metadata and record precision by the derivation rule; carry its path, shift and
    residual unchanged. Check the deadline before retaining the terminal.
11. Deduplicate completed terminals by `(IT number, structure signature, path)`.
    Sort by `(-number of symmetry operations, IT number, residual,
    canonical_result_key)`; expose all candidates, not only the first.
12. Return the candidates, the count of discovered states (including queued
    states), and explicit completion status. An interrupted current state or
    queued frontier is never substituted for a completed terminal. No terminal
    may have finished yet, so an incomplete result can contain no candidates.

The `canonical_result_key` referenced in steps 7 and 11 is exactly

```text
(
    site_key(result.asu),
    tuple(result.shift.to_fractions()),
    result.asu Gram entries in row-major order,
    tuple((transform.index, transform.subgroup_type) for transform in result.path),
)
```

Here `site_key` is the sorted tuple `(species name, canonical Wyckoff letter,
canonical exact parameter tuple)` from shared-core step 25, and
`tuple(result.shift)` contains its three exact fractions.

`LiftResult.asu` is authoritative. `path` records child-first tabulated hops;
`shift` records the final hop's continuous-normalizer shift in that parent
frame; `residual` is the largest accepted fractional residual. Unrecorded state
normal forms and the final exact terminal mean that `path` plus `shift` is not a
recipe for reconstructing `asu`.

## Default structure canonicalization

`canonicalize(structure)` (equivalently `canonical_asu`) enters its deadline
scope and follows these steps. The measured path contains one intentionally
tolerant boundary. Everything after recognition is exact.

1. Present the input as a unit-cell view. Reject an empty occupied-site set,
   site moments, assemblies, molecular structures, or less than full
   periodicity.
2. Derive the base Cartesian tolerance as twice the structure's Cartesian
   precision, with the recognition fallback of `1e-3`. Cap it strictly below
   half the smallest confirmed nearest-image intersite distance using
   `nextafter((d_min/2)*(1-1e-12), 0)`. Candidate differences are formed exactly,
   while the exhaustive closest-lattice-image search is binary64. An explicit
   `tolerance` is converted to a Python float.
3. Group sites by source `Species.name`. Replace these groups temporarily by
   `Species("__httk_anonymous_NNNN", ("X",), (1,))`; the width is
   `max(4, digits(number_of_classes-1))`. Preserve every tied map back to the
   original complete `Species` objects.
4. Expand exactly into P1, collapse exact translations, make the basis right
   handed, reduce the rational part of the metric by the proxy Niggli procedure,
   and enumerate its exact integral Gram automorphisms.
5. For each right-handed automorphism, use every atom belonging to a tied
   least-populated species class as an origin. Build each class block as
   `(population, sorted wrapped coordinates)`, sort blocks lexicographically,
   and rank first by `(full exact Gram, anonymous blocks)`. Coincident anonymous
   blocks are ambiguous and raise `ValueError`.
6. Assign the synthetic class names only after that rank. Canonically orient
   candidates and rank remaining ties by `(full exact Gram,
   (synthetic name, parameters) sites, Cartesian basis)`. Keep all exact source
   assignments tied at the winning frame and order assignments by Python
   `repr`.
7. Sort the caller's `factors` by descending `float(factor)`. For the default
   `(1/5, 1, 5)`, try recognition symprecs `5*base`, `base`, then `base/5`.
   The first candidate whose expanded model fits the input within the *base*
   tolerance wins. This early exit assumes that looser spglib tolerances do not
   produce fewer symmetry operations.
8. Recognition uses anonymous type integers and the deterministic P1 basis,
   origin, and site order. If every factor fails or the winner is P1, repeat the
   same sweep with that P1 site's order reversed. Replace the first winner only
   if the alternate has strictly more symmetry operations.
9. Validate a recognized model by equal same-species counts and a deterministic
   bipartite match. Candidate edges are sorted by `(squared distance, model
   index)` and input roots by `(candidate count, input index)`; deterministic
   augmenting paths require a distinct model site for every input site.
10. If no candidate fits, raise with the attempted symprecs. Otherwise send
    the recognized exact ASU directly to the shared core. There is no upward
    search and no `lift` option on this entry point.
11. Reframe the recognized ASU anonymously again. P1 uses the complete frame;
    higher groups retain their recognized Wyckoff representation and use the P1
    expansion only to establish anonymous class identities.
12. For `canonical_asu_protostructure_assignments`, run the exact core for every
    tied assignment with synthetic names in the discrete key. Retain every member
    tied by the anonymous geometry key; restore all compatible outer assignments
    and exact-deduplicate the family. Chemistry is consulted only after the
    anonymous geometry has been fixed.
13. For the scalar `canonical_asu` and `canonicalize`, restore the original species
    onto the recognized ASU and run the declared path (steps 1–7 above): the
    discrete key carries the species class names, as in `canonical_bare_protostructure`,
    so the result's label is the canonical bare label as a string (for sites
    declared on their own Wyckoff letter; a site declared on a more general letter
    than its point is demoted by step 14 and not by the bare canonicalizer). It
    need not be a member of the step 12 family.
14. Restore metadata as described below. No recognition or upward-search pass is
    run after the exact terminal. Check the deadline and bounded-work status
    before returning; an incomplete minimum raises rather than returning a value.

The recognition fit uses componentwise minimum-image wrapping and Python binary64
arithmetic. It is a conservative upper bound for a skew cell outside the
near-coincident regime. Python's `round` uses ties to even. Reproducing a boundary
decision also requires the same `math.fsum`, `math.ulp`, `math.nextafter`, and
binary64 behavior. This specification does not claim identical recognition on
all spglib builds or floating-point platforms.

## Shared anonymous exact core

The following 35 steps specify the shared terminal used by detected and
declared canonicalization and by classification values carrying representatives. Its input is a recognized exact ASU; it does no
recognition and no upward symmetry search.

1. **Validate.** Require an exact `ASUStructure`, full periodicity, no moments,
   no assemblies, and a non-molecular structure.
2. **Make the anonymous P1 frame.** Expand the unit cell, partition by species
   name, assign temporary classes, and retain every tied original assignment as
   described in measured-wrapper steps 3–6.
3. **Collapse P1 translations.** From the least-populated class, form candidate
   translations from one anchor to every member. Keep a translation only if it
   maps every class's exact wrapped-coordinate `Counter` onto itself. Generate
   the finer lattice from these translations and `Z^3`, compute an integer
   echelon basis, set `B' = S B`, and set `c' = c S^-1`. Every surviving site
   must occur exactly `1/|det S|` times; deduplicate it and divide charge by that
   multiplicity.
4. **Choose handedness.** If the P1 basis is left handed, apply the basis change
   `-I` and its inverse coordinate change.
5. **Proxy-reduce.** Apply the Niggli algorithm below to the rational coefficient
   of the exact Gram matrix, carrying the same integral transforms into the full
   exact basis. The loop limit is 10,000 steps.
6. **Enumerate metric ties.** Enumerate all integral `U` satisfying
   `U G U^T = G` and `det(U)=+1` for the proxy Gram. Bounds come from an exact
   rational LDL decomposition. The corresponding point map is `U^-T`.
7. **Choose origins and classes.** For every operation and every atom in every
   tied least-populated class, subtract that atom as origin, form and sort the
   anonymous blocks, and reject indistinguishable coincident class blocks.
8. **Rank the P1 frame.** Minimize `(full Gram row-major, anonymous blocks)`.
   Assign synthetic identifiers after this comparison. Canonically orient and
   minimize `(full Gram, ordered synthetic sites, basis row-major)`; preserve
   all source assignments at the minimum.
9. **Propagate precision.** Require the total basis change relative to the
   original to be rational. Multiply cell precision by the maximum absolute row
   sum of the change, and coordinate precision by the maximum absolute column
   sum of its inverse.
   Steps 3–9 construct the auxiliary P1 frame. For an original P1 input, use
   that frame as the terminal input. For every higher group, retain the original
   recognized cell, setting and Wyckoff sites, replacing only species names
   according to each tied frame assignment. Run steps 10–30 for each such input;
   steps 31–35 choose and restore the tied results. The terminal is told which
   original species each synthetic name stands for, so its discrete key (step 20)
   can carry their species class names.
10. **Standardize.** Convert the recognized group to its IT standard setting.
    The stored transform maps standard coordinates to the input setting as
    `f_own = f_std M^T + v`; hence
    `f_std = (f_own-v) M^-T`, `B_own = M^-T B_std`, and
    `B_std = M^T B_own`. Multiply cell precision by the maximum absolute row sum
    of `M^T`, coordinate precision by the maximum absolute column sum of
    `(M^T)^-1`, and charge by `abs(det M)`.
11. **Collapse declared supercells.** In P1 repeat step 3. In P-1, when exact
    pseudo-translations exist, collapse the P1 expansion as in step 3 and read it
    back as P-1, one site per inversion orbit: the inversion keeps the finer
    lattice, and the collapse then does not depend on the input cell (the
    tabulated self-lifts below reach only some sublattice orientations). The bare
    label keeps the declared supercell, so the refined label then differs from it.
    In group 3 or higher, count exact pseudo-translations outside the group's own centring lattice.
    Repeatedly consider the concatenated vendored same-IT transforms, remove
    duplicates, keep indices greater than one that divide
    `pseudo_translation_count+1`, and stable-sort by ascending index while
    preserving table order at equal index. Try each with tolerance zero; take
    the minimum result of the first transform that lands. Tabulated indices are
    at most 9. If none lands, warn and retain the current supercell.
12. **Enter P-1 once.** For group 2, proxy-Niggli-reduce and force the
    transform proper. That cell is the only entry: the P-1 candidates of step 22
    depend on the lattice, not on its basis. Other groups also have one entry.
13. **Reduce triclinic lattices.** For groups 1 and 2, apply exact full Niggli
    reduction when the Gram is rational. A non-rational Gram that the reducer
    cannot handle remains in its proxy-selected entry.
14. **Demote sites.** Expand each orbit and identify the most specific exact
    Wyckoff position that contains it. Replace a site that lies on a special
    position. Repeat after a chirality flip. Then re-describe a left-handed cell
    as the same crystal in a right-handed one: by an improper group operation,
    else (a Sohncke group) an improper normalizer action, else (an
    enantiomorphic group) the inversion `f -> -f` with the basis negated, into
    the partner group. The same re-description reads a left-handed declared
    representative, or ASU, as the crystal wherever its letters become a label.
15. **Apply chirality policy.** With `preserve_chirality=False`, if the group is
    the higher member of an enantiomorphic pair, map every point as
    `f -> -f mod 1`, keep the basis, re-identify complete orbits in the lower
    partner, and continue there. In a Sohncke group that is its own
    enantiomorphic partner (only proper operations, e.g. P1, P2₁, I4₁, P4₂32,
    I4₁32), the enantiomer (the same fractional coordinates and group in the
    negated basis) is canonicalized as well, with its own anonymous frame, and
    the candidates of both hands compete in steps 31–33. The enantiomer keeps the
    group whether or not the inversion through the origin does (I4₁, I4₁22 and
    F4₁32 need an origin shift for it). The enantiomer also competes in a group
    with improper operations when the cell metric does not keep every operation
    exactly (`Wᵀ G W ≠ G`), as for an ASU recognized from measured data, which keeps
    its measured cell: there the improper operations cannot identify the hands.
    The test is exact. A cell built from parameters keeps the right angles and
    120° angles its group fixes exactly, and a rhombohedral-axes cell whose angle
    has no exact cosine (or whose exact cosines give an irrational `cz²`, e.g. 72°
    or 75°) is built in its symmetric orientation with exactly equal edges and
    angles, so among the repository fixtures the test fires only for cod/1001844
    (P1 as stored, recognized in a higher group), where the mirrored hand's
    representative wins.
    Other groups remain. Moments are already excluded.
16. **Build representative operations.** Start with identity. For P1, append all
    exact integral automorphisms of the full rational Gram. For other groups, append every affine
    normalizer coset in vendored order. Do not filter by the lift search's
    compatible crystal systems.
17. **Enumerate discrete translations.** Solve exactly for all translation
    classes `t` such that `(I-W)t` belongs to the full group translation lattice
    for every point operation `W`; the lattice includes centring translations.
    Integer-diagonalize the constraint matrix, enumerate the Cartesian product
    of fractions `k/d_i`, omit zero diagonal directions because they are
    continuous, map back, wrap, deduplicate, and sort. Identity is included.
18. **Enumerate group point maps.** Take one space-group operation for every
    distinct point matrix, preferring identity for the identity matrix. Cross
    these maps with every representative and discrete translation.
19. **Compile each Wyckoff action.** For one source position, visit target
    positions in their actual `spacegroup.wyckoff` order, which is most-specific
    order `(free_count, multiplicity, letter)`. Skip unequal free count or
    multiplicity. Visit the target position's branches in stored table order.
    Derive the parameter map by the chart equations below and accept the first
    chart whose *complete orbit family* matches exactly coefficient by coefficient
    modulo lattice translations. Failure to prove any action raises.
20. **Make the discrete key.** For each class, sort its mapped Wyckoff letters
    and sum their conventional multiplicities. The anonymous half is the sorted
    tuple of `(-multiplicity, letter tuple)` class keys. The assignment half is
    the sorted tuple `(-multiplicity, letter tuple, class name)`: the species
    class name of the original species for `canonical_asu`, `canonicalize` and the refined
    protostructure, so the pair is exactly the bare canonicalizer's assigned key;
    the synthetic class name for `canonical_asu_protostructure_assignments` and
    the anonymous refined prototype. This is the label's class order
    (`class_order_key`). In groups 2 to 15 the candidates are also taken after
    every kept cell change `σ` (see the classification section). Compare the pair
    `(anonymous, assignment)`, and retain the complete minimum tier. In groups 3
    to 15 every `σ` reaching it contributes; P-1 is step 22. In a
    Sohncke group only proper descriptions (`σ·action` with total determinant
    +1) compete, whatever `preserve_chirality` is: the two hands of a group that
    is its own partner are compared in step 15 instead.
21. **Keep right-handed descriptions.** The candidate basis change is the group
    point-map inverse transpose multiplied by the normalizer operation's inverse
    transpose. Candidates with a left-handed basis change are dropped: outside the
    Sohncke groups an improper group point map gives each a right-handed twin with
    the same keys, and in a Sohncke group the descriptions are proper.
22. **Reduce the cell and choose the metric.** In groups 3 to 15 move every
    surviving candidate by each cell change of `K` (the cell changes that keep
    every letter and operation) to its least `(a·a, c·c, a·c)`, the reduction
    key (see the classification section); in a Sohncke group only the cell changes
    of `K` with determinant +1. In P-1 the letters of an action `(M, t)`
    depend only on `(M mod 2, 2t mod 2)`, so a cell `P B` keeps the least key
    exactly when every row of `P` lies in a prescribed class of `L/2L`, with
    the matching half translations. Selling-reduce the exact Gram (surds
    included) to an obtuse superbase; the least vectors of each class then
    follow combinatorially from its conorms, without evaluating a norm. Keep every
    admissible class triple with the least `(a·a, b·b, c·c)`, every triple of
    signed class minimizers with determinant +1 (in a degenerate lattice three
    minimizers of independent classes can have determinant 3), and every
    admissible half translation. The reduction key is those three norms, then
    whether the Gram fails the exact Niggli conditions. Then compare
    `(reduction key, full Gram row-major)` and retain the least. Candidates that differ by a
    lattice translation give the same result and are evaluated once.
23. **Apply each distinct site action.** Apply `p' = p A + q mod 1`, transform
    the basis, and canonicalize translations and orbit representatives once per
    distinct site operation. For the action's basis change
    `R = operation.matrix^-T`, multiply cell precision by the maximum absolute
    row sum of `R` and coordinate precision by the maximum absolute column sum
    of `R^-1`. Cache compilation results by exact inputs; the 100,000-entry
    cache affects performance only.
24. **Normalize continuous origins.** Read the vendored continuous Euclidean
    normalizer basis and take every coordinate axis with a nonzero component as
    movable. Candidate shifts are identity plus, for every expanded orbit point,
    the shift that cancels all movable components of that point. Sort and
    deduplicate candidates; identity wins an equal key. P1 evaluates translated
    site keys directly. Groups 4, 29, and 33 use their proven one-axis prefix
    shortcut; all others apply each shift exactly and compare full keys.
25. **Choose orbit representatives.** Expand an orbit and select its
    lexicographically least wrapped point. Recover parameters in the same
    nondegenerate position; if it has degenerated, identify the most-specific
    position instead. Sort sites by `(species name, Wyckoff letter, exact
    parameters)`.
26. **Make the geometry key.** Compare
    `(full metric row-major, site key, basis row-major)` and keep the least exact
    candidate. Materializing its selected basis uses the rational change
    `R = B_new B_current^-1`; multiply cell precision by the maximum absolute
    row sum of `R` and leave coordinate precision unchanged.
27. **Choose exact Cartesian orientation.** Construct the lower-triangular
    Cholesky factor of a rational Gram. Accept a square root if it is rational,
    is `sqrt(r)*q` for integer `r` from 2 through 48 and rational `q`, or requires
    general factorization with `(numerator*denominator).bit_length() <= 48`.
28. **Use the orientation fallback.** For a non-rational Gram, try rational cell
    lengths with stored `CellParams` angles and accept only exact Gram equality.
    Otherwise retain the input global Cartesian rotation. This fallback is
    idempotent but cannot give a universal rotation-invariance guarantee for
    unsupported nested-radical metrics.
29. **Preserve handedness.** The protostructure terminal's cell is right-handed
    (steps 14 and 21). (The legacy terminal first inverts a left-handed cell
    jointly when inversion preserves the group, and uses a fixed left-handed
    Cartesian factor only where that map is invalid.) Never silently change the
    physical chirality.
30. **Select among entries.** Compare
    `(discrete key at identity, reduction key, metric, site key, basis)` across
    the tied P-1 entries from step 12 and the enantiomer entries of step 15.
31. **Select anonymous geometry.** Across tied source assignments compare
    the named discrete key of step 20 (empty for the anonymous family), then
    `(IT number, Hall entry, metric, sorted(population,
    sorted(Wyckoff letter, parameters)) by class, basis)`. The second
    key contains no original chemical name.
32. **Restore assignments.** Map synthetic names back to complete original
    `Species` values. Deduplicate by
    `(spacegroup, basis, site key, species tuple, coordinate precision, charge)`.
33. **Choose the scalar chemical tie.** Order the remaining family by
    `(site key, tuple(repr(species)))`. `repr(Species)` begins with `name`,
    `chemical_symbols`, and `concentration`, then includes non-`None` fields in
    this order: `mass`, `original_name`, `attached`, `nattached`,
    `concentration_precision`, `charges`, `spins`, `labels`. Its concrete form
    is `Species(name={name!r}, chemical_symbols={symbols!r},
    concentration={concentration!r}, ...)`, using Python `repr` for every value.
34. **Restore metadata.** Preserve charge and the complete species tuple,
    including unused species, plus composition, descriptive and Hill formulas,
    optimization type, immutable ID, and last-modified value. Scale an explicit
    composition by `output expanded count / input expanded count`. Do not copy a
    computed content ID.
    After restoring each assignment, record the precision from the source. A
    valid derivation is an exact `T` of either determinant sign whose rows lie in
    the source translation lattice (`Z^3` plus the exact self-translations of its
    expanded unit cell, centrings included), with `T G_src T^T = G_res`, under
    which the source's expanded sites, mapped by `x' = x T^-1` and wrapped, equal
    the result's up to one common translation, species included; a result cell
    `|det T|` times larger takes every image of each site under the source
    lattice. Rows are enumerated exactly in a proxy-reduced source basis with the
    Cauchy bound `t_j^2 <= (G_res)_kk (G_src^-1)_jj`, surd Grams included. Of the valid derivations take the least
    `(row sum of T, column sum of T^-1)`, and record the source cell precision
    times that row sum and the source coordinate precision times that column
    sum. Apply this rule at the exact-family boundary, the outer measured-family
    boundary and the right-handed re-description of step 14. The recorded bounds
    then depend on the source and the result, not on the internal path: an
    unchanged result keeps the source bounds, and a crystal and its mirror image
    record the same. Where no valid derivation exists (a tolerance-snapped
    recognition, a lifted crystal) the propagated conservative path bounds are
    kept.
35. **Return.** `canonical_asu` and `canonicalize` restore the original species
    onto the recognized ASU and run steps 10–34 with species class names in the
    discrete key, returning the one result; its label equals the canonical bare
    label as a string for sites declared on their own Wyckoff letter (step 14
    demotes a site declared on a more general letter; the bare canonicalizer does
    not). `canonical_asu_protostructure_assignments` runs them with
    synthetic names and returns the exact-deduplicated tied family of the
    anonymous convention, of which the canonical ASU need not be a member.

Exact comparisons of `SurdScalar` and `SurdVector` values are comparisons of
their real algebraic values. An implementation must use exact sign refinement;
ordering coefficient tuples or floating approximations can choose a different
minimum.

### Proxy Niggli pseudocode

For a rational proxy Gram `G`, define

```text
A = G00; B = G11; C = G22
xi = 2*G12; eta = 2*G02; zeta = 2*G01
T = I
```

At each iteration, take the first applicable branch below, set `T = U*T` and
`G = U*G*U^T`, recompute the six values, and restart. `sign(x)` is `+1` when
`x > 0` and `-1` otherwise.

```text
if A>B or (A==B and abs(xi)>abs(eta)):
    U = [[0,-1,0],[-1,0,0],[0,0,-1]]
elif B>C or (B==C and abs(eta)>abs(zeta)):
    U = [[-1,0,0],[0,0,-1],[0,-1,0]]
else:
    if xi*eta*zeta > 0:
        U_sign = diag(sign(xi), sign(eta), sign(zeta))
    else:
        i=j=k=1; zero = none
        if xi>0: i=-1; elif xi==0: zero=i
        if eta>0: j=-1; elif eta==0: zero=j
        if zeta>0: k=-1; elif zeta==0: zero=k
        if i*j*k < 0: set the last recorded zero sign to -1
        U_sign = diag(i,j,k)

    if U_sign is not I:
        U = U_sign
    elif abs(xi)>B or (xi==B and 2*eta<zeta) or (xi==-B and zeta<0):
        U = [[1,0,0],[0,1,0],[0,-sign(xi),1]]
    elif abs(eta)>A or (eta==A and 2*xi<zeta) or (eta==-A and zeta<0):
        U = [[1,0,0],[0,1,0],[-sign(eta),0,1]]
    elif abs(zeta)>A or (zeta==A and 2*xi<eta) or (zeta==-A and eta<0):
        U = [[1,0,0],[-sign(zeta),1,0],[0,0,1]]
    else:
        total = xi+eta+zeta+A+B
        if total<0 or (total==0 and 2*(A+eta)+zeta>0):
            U = [[1,0,0],[0,1,0],[1,1,1]]
        else:
            stop
```

The sign phase records the *last* zero among `xi`, `eta`, and `zeta`. Its exact
branch order is part of the convention.

### Wyckoff chart projection

A branch is the affine family `x(p) = p C + b`, where row `p` has one entry per
free parameter, `C[p, coordinate]` is the branch operation's coefficient at the
corresponding free coordinate, and `b` is its constant. Under an affine
normalizer `(M,t)`:

```text
C'[p,j] = sum_k M[j,k] C[p,k]
b'[j]   = sum_k M[j,k] b[k] + t[j]
```

For a target branch with row-Hermite chart data `(U, P^-1)`, compute

```text
D = C' U^T
d = (b' - b_target) U^T
```

Let `r` be the common free-parameter count. Reject the chart if any
`D[:, r:3]` entry is nonzero or any `d[r:3]` entry is nonintegral. Otherwise

```text
A = D[:, :r] (P^-1)^T
q = d[:r] (P^-1)^T mod 1
p' = p A + q mod 1
```

Finally compare the complete transformed source orbit-family multiset with the
complete target-family multiset after composing `(A,q)`. Coefficients must agree
exactly and constants agree modulo integers. This chart projection is also how
recognition reads back the free parameters of each orbit's group-averaged point:
it is not a metric least-squares fit.

## Classification family canonicalizers

The four functions first materialize their corresponding classification view.
That conversion keeps the existing View contract: passing an already classified
value or an exact supported projection is exact, while a raw ordinary structure
accepted by a bare view crosses the documented tolerant spglib-recognition
boundary. Once materialized, the canonicalization steps below do not recognize
symmetry or search upward:

```python
from httk.atomistic import (
    canonical_bare_prototype,
    canonical_bare_protostructure,
    canonical_prototype,
    canonical_protostructure,
)

bare_assigned = canonical_bare_protostructure(value)
bare_anonymous = canonical_bare_prototype(value)
refined_assigned = canonical_protostructure(refined_value)
refined_anonymous = canonical_prototype(refined_value)

assert canonical_bare_protostructure(bare_assigned) == bare_assigned
assert canonical_bare_prototype(bare_anonymous) == bare_anonymous
assert canonical_protostructure(refined_assigned) == refined_assigned
assert canonical_prototype(refined_anonymous) == refined_anonymous
```

The bare minimization optionally moves a higher enantiomorphic group to its lower
partner and enumerates only finite exact letter-changing actions: identity,
vendored affine-normalizer cosets, exact discrete normalizer translations, and
group-preserving inversion. It uses only the materialized group and occupations;
the bare value carries no cell or coordinates.
Each class is keyed by the label's class order, `class_order_key`:
`(-summed conventional multiplicity, sorted letter tuple)`. The assigned key is
`(sorted anonymous class keys, sorted(class key + species class name))`; the anonymous
key is the sorted tuple of class keys. Because the label renders classes in the
same order, the chosen description is the one whose label is least in that
order; for spinel (227, Mg `a`, Al `d`, O `e`) this selects the origin-shifted
description Mg `b`, Al `c`, labelled `A4B2C_cF56_227_e_c_b:O-Al-Mg`.

The finite action order is identity followed by affine-normalizer cosets in
stored order; each is crossed, as the outer loop, with sorted discrete
translations. If inversion preserves the target group, prepend inversion to
each member of that complete list and append those images. Deduplicate by first
occurrence. When normalizing chirality, compose each target-group action after
the source-to-lower-partner inversion. With `preserve_chirality=True` (the
default) a Sohncke group uses only its proper actions, so a crystal and its mirror
image can get different labels: in P4₂32 H on `i` with P on `b` gives
`A3B_cP16_208_i_b:H-P`, and with P on `c` `A3B_cP16_208_i_c:H-P`, each the mirror
image of the other. With `preserve_chirality=False` the improper actions are used
too, and both give `A3B_cP16_208_i_b:H-P`. Python's first minimum wins an equal key.

In the monoclinic groups (3 to 15) the cell in the plane perpendicular to `b` is
a convention, and changes between its valid cells can exchange special positions;
the tabulated set contains no such shears. There the action set is extended to be
closed in *signature*, the Wyckoff letter map together with the relabelling of the
operations (shears have infinite order, so no finite set of actions is closed under
composition). The generators are the cell changes `[[a,0,b],[0,1,0],[c,0,d]]`
with entries in -2..2 and determinant ±1 that map the group onto itself. A
breadth-first search over their products keeps one cell change `σ` per signature
not yet reached, giving a set `S` that starts with the identity; the action set is
`S · Tab`, every kept `σ` composed with one tabulated action per coset of the
group, in that order. In P-1 the tabulated set is not closed and is replaced:
`S` is built the same way from the elementary integer matrices (identity plus one
off-diagonal unit), and `Tab` is the eight half translations, giving one action
for each of the 1344 signatures (the affine maps of the eight inversion centres).
The sets are built once per group and cached; their letter maps are composed from
the proven maps of the generators and the tabulated actions. For example, in
P2₁/c the occupations `a, b`, `a, d` and `b, c` all give `AB_mP4_14_a_b`, while
`a, c` gives `AB_mP4_14_a_c`.

The refined terminal takes the same kept cell changes, so its least discrete key,
and hence its label, is the canonical bare one; its discrete key carries the species
names, so the refined and bare labels agree as strings, species order included, for sites
declared on their own Wyckoff letter (a site declared on a more general letter than its point
is demoted by the refined terminal only). In
groups 3 to 15 every `σ` reaching that key contributes, and the cell of each
description is reduced within its class: `K`, the cell changes whose signature is
the identity, contains every cell change congruent to the identity modulo 4, so it
is known by its 8 to 96 residues modulo 4. Each candidate's ac block is moved by
every `X ∈ K` that minimises `(a·a, c·c, a·c)`, found exactly by a level-4
enumeration in a Lagrange-reduced basis (first rows with `|y| ≤ 4`, second rows
`±w + j·u` per admissible `j` modulo 4). That minimum over all classes is the
reduction key; ties go on to the full metric and the site key. No
tier prefers a `σ` or the input's own description, so the representative's label,
cell and sites do not depend on the input cell. The cell need not be the
conventional reduced cell: keeping the letters can force a longer axis, e.g.
P2₁/c with Cl on `a` and Na on `b` needs `a' = a + c`. In P-1 the stored cell has
the lexicographically least `(a·a, b·b, c·c)` among the cells that keep the
canonical letters, then the Niggli Gram where it is among them (step 22). With
every atom on the general position every cell keeps the letters, so the lengths
are the successive minima and the stored Gram is the Niggli Gram, for a surd
metric too. Atoms on inversion centres can force another cell, the most reduced
by this lexicographic rule but not necessarily reduced in the usual sense, since
the shortest vector can come last: in the cell (5, 6.083, 7.348 Å) with Na on `a`
and Cl on `d`, the canonical letters `a, b` need Na→Cl to be `c/2`, so `c` is the
5 Å vector and the stored cell is (6.083, 7.348, 5.000 Å).

In a Sohncke group the refined terminal uses only proper descriptions of a
right-handed entry, so its label keeps the hand as the bare one does and its
representative is the crystal itself in a right-handed cell. For example, P4₂32
with H on `i` and P on `c` (structreading/208) gives `A3B_cP16_208_i_c:H-P`, and
its mirror image `A3B_cP16_208_i_b:H-P`; with `preserve_chirality=False` both give
`A3B_cP16_208_i_b:H-P`, with the right-handed representative of the hand that
reaches it (for a pair group, of the lower partner).

A refined value with no representative canonicalizes its bare base and retains
its discriminator unchanged. A refined value with a representative rebuilds an
exact ASU, runs the shared terminal, rebuilds the representative, and retains
the discriminator. A retained `WyckoffSite.representative` annotation may be
discarded only when it is exactly one member of the orbit generated by the
stored free parameters. Otherwise canonicalization raises rather than treating
an imprecise annotation as exact data.

An anonymous refined representative (a `Prototype`) is canonicalized with
synthetic class names in its discrete key, so input class names never matter, and
relabelled after geometry is canonical. Occupied classes are ordered by the label's
class order `(-summed conventional multiplicity, sorted Wyckoff-letter tuple)`, then
by their sorted `(letter, exact-parameter-tuple)` rows, and receive `A`, `B`, ...
in that order, so the representative's class names match its label. Unused dummy
species retain their count and receive the trailing canonical labels in their
existing order.

A refined geometry terminal can demote a site onto a special position or collapse
an exact P1 or same-group supercell. Those geometry-dependent changes are
unavailable to the bare routine, so canonicalization and projection do not
generally commute. Canonicalize the projected value whenever a canonical bare
result is required.

## Bonding-net canonicalization

`canonical_protonettype` and `canonical_protonetstructure` canonicalize a bare
prototype or bare protostructure together with a symmetric covalent bonding net
(see {doc}`prototypes` for the notation). Like the bare routines they work from
the materialized value alone, its bonds and shapes, and use no coordinates. The functions take
`preserve_chirality` and `timeout`; with `preserve_chirality=False` the net is
first transferred into the lower enantiomorphic group by the same inversion the
bare routines use, and canonicalized with that group's own tables. With
`preserve_chirality=True` a Sohncke group compares only proper descriptions (a
monoclinic cell change times a coset action of determinant +1), so the net keeps
the hand as the bare label does.

The `Protonettype` and `Protonetstructure` constructors, and `from_asu`, run the
same search restricted to the equivalent descriptions with the input's own bare
key, with `preserve_chirality=True` and the default timeout: the *anonymous*
bare description (the anonymous bare key) is kept and only the net is
canonicalized. When the input's bare key is already the canonical one, the
restricted search is the full search and the constructor's value equals the
canonicalizer's. At the assigned level the D3 tie-break may move species between
the Wyckoff letters of equivalent descriptions with that key:
`Protonetstructure(225, [("a", Na), ("b", Cl)])` gives `AB_cF8_225_a_b__:Cl-Na`,
whereas `BareProtostructure(...)` keeps any assigned description.

### Definitions

- **Element.** `(k, n1, n2, n3)`: apply operation `k`, then add the lattice
  translation `n` (conventional cell), so `x -> W_k x + w_k + n`. Operation `n`
  (1-based in labels) is the `n`-th branch of the general Wyckoff position of the
  standard setting, with the identity first.
- **Reference atom.** The reference atom of a site is its first-listed Wyckoff
  coordinate evaluated at the site's free parameters, not reduced modulo one and
  not moved into any asymmetric unit.
- **Row.** `(i, j, g)`: the reference atom of site `i` is bonded to `g` applied to
  the reference atom of site `j`. A row stands for the whole symmetry orbit of
  that bond, and a net is a set of rows. A row may be written from either end;
  reversing it replaces `g` by its inverse.
- **Code of a row.** Let `H_i` and `H_j` be the site-symmetry groups of the two
  reference atoms. Every element of the double coset `H_i g H_j` names the same
  orbit, and for a bond inside one orbit (`i == j`) the inverse double coset does
  too. The code is the member with the smallest key
  `(n1² + n2² + n3², k, n1, n2, n3)`.
- **Gauge.** Replacing the reference atom of site `s` by `a_s` times the old one
  turns every row `(i, j, g)` into `(i, j, a_i g a_j⁻¹)`. The new atom must stay on
  the first-listed coordinate, so `a_s` normalizes `H_s`: any element for a
  general position, none beyond the site symmetry for a parameter-free one.
- **Equivalent descriptions.** The finite same-group action domain of the bare
  canonicalizers, reduced to one representative per coset of the space group and
  per continuous origin shift. An action moves a row to
  `(i, j, b_i (ν g ν⁻¹) b_j⁻¹)`, where `b_i` is a group element that puts the
  moved reference atom back on the first-listed coordinate of its (possibly
  different) Wyckoff letter.
- **Site numbers.** Sites are numbered 1 to N in label order: classes in
  `class_order_key` order and, within a class, by Wyckoff letter. Sites sharing
  class and letter are interchangeable, and so are classes tied in count and
  letters.
- **Bond count.** The number `c(s)` of distinct atoms bonded to the reference atom
  of site `s`: a row `(a, b, g)` reaches the atoms `h g H_b` (`h` in `H_a`) from
  `a` and `h g⁻¹ H_a` (`h` in `H_b`) from `b`, each coset counted once. It is a
  site invariant: no description, gauge or numbering changes it.
- **Shape.** One symbol per site with two or more bonds, carried with the site
  (see {doc}`prototypes`); also a site invariant.
- **Listing order.** A site is *top* when no bonded neighbour has a larger bond
  count. Sites are listed by larger count, then top sites, then smaller number.
  Each row is written from the end listed first (its *listing site*), and the
  label order of the rows is listing position of the listing site, then of the
  target, then code.

### Canonical net

Over every equivalent description that yields the canonical bare key, every
allowed numbering and every gauge, the canonical net minimizes, in this order:

1. the bare key;
2. the sorted list of site pairs `(min, max)` of the numbers, one per row
   (the numbering pattern);
3. the listed pairs `(listing site, target)` in label order;
4. the list of row code keys in label order, each the least member of its double
   coset in the listed direction;
5. the shapes by site number, compared as strings (`""` for none).

For a numbering, the listing order follows from the bond counts and the numbers
alone, so steps 3 and 5 cost one pass over the rows, and step 3 prunes a
numbering before its exact code search. The codes are minimised exactly in label
order by running the reference-atom search of step 6 below with each site's
listing position in place of its number. Shapes never choose reference atoms: for
a fixed description and numbering they are fixed, so step 5 only decides between
numberings tied on steps 1 to 4, the automorphic ones that map sites of different
shapes onto the same numbers. The net text is a function of steps 1 to 5.

The result does not depend on the description. The bond counts, and so the
listing order of a numbering, are site invariants, and shapes travel with their
sites; so two descriptions related by a cell change, a coset of equivalent
descriptions, the chirality transfer, a gauge or a permutation of sites and rows
give equal keys at every step for corresponding numberings, and the existing
exactness arguments for the numbering and reference-atom searches carry over
unchanged. The cell choices from the net (step 2 of the algorithm) read only the
sites and rows, and are unchanged.

The bonds and codes do not depend on coordinates, which is the point: choosing
reference atoms by position (unit cell, asymmetric unit, reduced parameters)
would change the string when an atom crosses a boundary in another compound of
the same prototype. The shapes do depend on coordinates, through the baseline
detector that `from_asu` applies to sites without a supplied shape; the
canonicalizers themselves read the shapes the value carries. For the assigned
level the anonymous canonical net is found first; only if several realisations
give the identical net string is the tie broken by the species class name tuple
in class order, smallest first. The assigned value's bare part can then differ
from `canonical_bare_protostructure`, which breaks ties between tied classes by
class name alone.

### Algorithm

All arithmetic is exact: integers in the search, `Fraction` only while the
per-space-group tables are built.

1. **Tables.** Per space group, cached: integer multiplication and inverse tables
   for elements; per Wyckoff letter the site-symmetry group `H`, the lattice of
   translations along its flat and the classes of gauge moves; and the cosets of equivalent descriptions, each compiled to an
   operation relabelling, an integer matrix acting on translations, and the
   target letter and `b` for every letter. Everything is computed at one generic
   exact rational probe point, so it does not depend on the structure.
2. **Cells from the net (groups 3 to 15).** The changes between valid ac cells
   form an infinite group `Γ`, which no finite action set reaches, so the search
   first chooses cells from the net. It places every atom at the average of its
   bonded neighbours (exactly, over the free parameters, each row weighted by the
   number of its bonds in the cell) and forms the Gram matrix `G` of the bond
   vectors; a cell change `X` turns its ac block into `X G Xᵀ`. Its rank decides
   the cells: rank 2, every cell of `Γ` reducing it in the order
   `(G11, G22, G12)`, ties kept; rank 1, those cells modulo shears along the
   periodic direction by a fixed per-group step `d` of 1, 2 or 4 (the
   denominator of the points that 2-fold axes, 2₁ axes and inversion centres pin
   a component to); only periodic along b, one cell per class of `Γ` modulo
   `d`; finite molecules only, the identity. Each set is taken modulo `±I`, and
   the b-periodic case runs only the tabulated cosets, since its cells already
   cover the cell changes of `S`.
3. **Cosets.** Each cell, then each coset, transforms the rows and the sites;
   pairs whose bare key is not the minimal one are skipped.
4. **Class order.** The classes follow `class_order_key`; every permutation of
   classes tied in count and letters is tried.
5. **Numbering.** A branch and bound over the numbers 1, 2, ... When a site gets
   its number, its still unnumbered neighbours must take the next free numbers of
   their own blocks, since anything else lengthens the pair list; only their order
   among themselves is branched. The search returns the minimal pair list and
   every numbering that realises it.
6. **Reference atoms, exactly.** A reference atom is written as a fixed
   operation, a fixed lattice translation and a part linear in a vector `v` of
   integer unknowns: `a_s = (k_s, c_s + M_s v)`. A general position has three
   unknowns (any lattice translation) and one start per operation. A special
   position has one unknown per independent lattice translation along its flat
   and one start per remaining class of gauge moves, a finite list. A general
   position bonded to an atom already placed takes no unknowns: it is one of the
   atoms actually bonded, `a_j = h a_i g` with `h` in `H_i`. Rows are visited in
   pair order while a set of states is kept. For a row the element
   `a_i g a_j⁻¹` has a fixed operation and the translation `b + M v`. For each
   member `h1 h2` of the double coset the translation becomes a coset of a lattice
   in Z³, whose smallest vector under `(m·m, m1, m2, m3)` is found exactly: the
   lattice is brought to column echelon form and every vector no longer than the
   best so far is enumerated pivot row by pivot row. The row's code is the
   smallest key over all members. Fixing it is a linear condition on `v`; it is
   solved, substituted into every reference atom of the state, and the state is
   normalised to a unique form so that equal states merge. Every state that
   reaches the row minimum survives, and unknowns that are never constrained do
   not affect any code. No search range is used anywhere in this step; the only
   finite lookup is the classes of gauge moves of a special position, collected
   from a ±2 cell neighbourhood.
7. **Components and ties.** Connected components of the site graph are searched
   separately, since a row's code depends only on the reference atoms of its own
   component; the merged result is the same as one joint search, and the state
   sets never multiply. Within a group of tied classes, the classes that appear in
   no row are interchangeable, so they are placed in sorted order among the
   positions the bonded classes leave free. Neither change alters the result.
8. **Result.** The smallest `(bare key, pair list, listed pairs, code keys,
   shapes)` is kept together with all realisations `(cell, coset, numbering,
   gauge)` of it. Results are memoised on the oriented rows reduced to their codes
   and the shapes.

The per-group tables are the one-off cost; a net search on a typical structure
takes a fraction of the corresponding bare or protostructure canonicalization,
and its time does not depend on how far the input reference atoms are from the
best ones.

### Limits and accepted failure modes

- **Exactness is argued, not machine-checked.** The minimum over reference
  atoms and the numbering search are both exact, so for a given set of equivalent
  descriptions the result is the true minimum by construction. This is an
  argument; it is supported by scrambling tests (random coset, random gauge
  elements, shuffled sites and rows, reversed rows) but not verified by machine.
  A net for which two descriptions give different strings is a defect to report.
- **Low symmetry.** The canonical net is description-independent only as far as
  the searched descriptions reach every change between two valid descriptions.
  In the monoclinic groups the cells chosen from the net (step 2) do, so no
  monoclinic net, molecular or periodic in any direction, depends on the cell.
  In P-1 the action set is closed in signature (see the classification family
  canonicalizers above), so labels and nets that are not periodic in any
  direction (molecular crystals) do not depend on the cell, but nets with any
  periodic component remain canonical only relative to the cell that httk's
  standardisation chooses: their codes contain lattice translations, which the
  shears change (D7). P1 is excluded from the guarantee: it has no special
  positions, so its labels are unaffected, but its periodic nets stay canonical
  only relative to the cell.
- **Numbering blow-up.** The numbering search is exact but exponential in the
  worst case, with many interchangeable, similarly connected sites. The whole
  call runs under the cooperative `timeout` (default 120 seconds), the bare part
  first and the net search with what remains. No partial net is returned: with
  the default `on_limit="undetermined"` the result is an undetermined value whose
  label keeps every determined part and has `X` for the rest
  (`<bare label>__X`, see [undetermined labels](prototypes.md#undetermined-labels));
  `on_limit="raise"` raises `CanonicalizationLimitError`.
- **Translation range.** A label can write a translation only in -5..+4 per axis.
  Input rows and bonds take any integer translation, and gauge moves absorb
  unreduced free parameters; a canonical net that still needs a longer
  translation (a genuinely long bond) raises `ValueError` (D4).
- **Reference atoms are not geometry.** The canonical string does not store the
  reference atoms. Laying a net back onto coordinates requires the realisation's
  gauge, which the label does not carry.

## Limits and accepted failure modes

- Exact algorithms require rational fractional data and supported exact surds.
  Only the orientation fallback may retain an input global Cartesian rotation.
- Niggli reduction is capped at 10,000 steps.
- The upward lift solver permits 200,000 exact/product branches, 4,096 noisy
  fallback branches, and 20,000 Fourier-Motzkin inequalities. A saturated
  parent can be skipped with a warning, so the returned terminal can then be
  below the true maximum.
- `highest_symmetry` permits 10,000 visited states. `all_paths=True` consumes
  this budget faster because paths participate in the visited key.
- Exact same-group supercell collapse uses table indices through 9 and can
  retain a supercell with a warning if no tabulated self-lift lands.
- Metric automorphisms and discrete translation classes are exact finite
  enumerations with no heuristic candidate cap. Continuous normalizer freedom
  is reduced using the vendored axis-aligned basis convention.
- The measured default and declared canonicalization do not run the breadth-first
  search. Upward-search limits concern `search_supergroups` and explicit legacy
  or experimental calls that request that search.
- The benchmark corpus exercises the measured anonymous recognition path. It
  does not establish full coverage of the upward BFS or prove species covariance
  for its name-bearing intermediate states.

## Limits, completion and migration

`canonicalize`, `search_supergroups` and the six classification-family
canonicalizers default to `timeout=120.0` seconds; `None` disables the deadline.
The classification-family canonicalizers return an undetermined label on a limit
unless called with `on_limit="raise"` (see {doc}`prototypes`); `canonicalize`
always raises. A positive finite value is required. The clock is
monotonic. A scoped `ContextVar` isolates concurrent callers and restores the
previous scope even after exceptions; nested scopes cannot extend an enclosing
deadline. Checkpoints cover recognition sweeps and fitting, anonymous origins,
normalizer candidates, translation reductions, solver products/recursion, search
states and parent/cell choices. A result is checked again before return.

This is a **cooperative deadline**, not hard preemption: an individual native
spglib call or uninterrupted arithmetic operation may overrun it. Process-level
limits such as `httk memguard` remain useful for batch workloads. A deadline
changes success versus failure, never which partial minimum becomes canonical.
`CanonicalizationLimitError` is a `RuntimeError`, so normal recognition retries
that catch `ValueError` cannot swallow exhaustion.

`SupergroupSearchResult` carries `candidates`, `complete`, `reasons` and
`states_visited`. Reasons can include `deadline_exceeded`, `state_limit_exceeded`,
`modular_solver_branch_cap`, `noisy_solver_branch_cap` and
`fourier_motzkin_limit`. More than one reason can be recorded. Modular/noisy
branch skips continue exploring other branches but set `complete=False`. An
escaping Fourier–Motzkin limit stops the search and retains previously completed
terminals. Completion refers
to the implemented finite tables and retry conventions, not proof of a global
maximum over all possible interpretations. Partial candidates are exploratory,
not canonical identities. The CLI returns a failure status for incomplete search
and refuses to save its first candidate as a canonical structure.

Legacy calls create no deadline scope, retain their original return types and
numerical conventions, and retain the old fixed state cap. `highest_symmetry`
returns all legacy terminals; `canonicalize_legacy` returns its first. Those
legacy terminals use the historical metric-first terminal and optional chirality
normalization rather than the shared anonymous terminal used by
`search_supergroups`. `canonical_asu_legacy` preserves the earlier measured
recognition/terminal behavior and its optional upward-search flag.

Migration: replace `canonicalize(asu).asu` with `canonicalize(asu)` for ordinary
canonicalization, or use `symmetry="declared"` to retain the supplied model.
Code that needs search routes must call `search_supergroups` and inspect
completion before selecting a candidate. Replace `canonical_asu(..., lift=True)`
with explicit recognition followed by `search_supergroups(recognized_asu)`;
remove redundant `lift=False`. Imports of `canonicalize` belong to
`httk.atomistic`, `httk.atomistic.symmetry` or `symmetry.canonical`, not `lift`.
Identity-pinned workflows can keep using the explicit legacy functions.

## Reproducibility pins

Independent exact reproduction requires these packaged version `0.1.0` gzip
members byte for byte:

| File | SHA-256 | Generator commit |
| --- | --- | --- |
| `symmetry_basics.json.gz` | `98ce57f5c0a592705c0168c4378a307428d8b65915930af2850d0978cbeaa908` | `66bd7246630c580689bd7e31fa04641146564d5d` |
| `spacegroup_setting_transforms.json.gz` | `e0f26de161d8f1c25aa0eaefbaaa4b8361ecdf55f183a3b67e39ef8af2cca55a` | `3eee0dc59f9bce6f8a704ac06be6d887e6550acc` |
| `baernighausen_std.json.gz` | `998ebbb04ba166fb53425d8ad191b555cdc507c4a08d74e8b88501f7b37cede1` | `3eee0dc59f9bce6f8a704ac06be6d887e6550acc` |
| `continuous_euclidean_normalizer_std.json.gz` | `dc8a1dfb16b685d72dec7751df1876daf7a19321c251eb86b97944cd00bbf721` | `3eee0dc59f9bce6f8a704ac06be6d887e6550acc` |
| `affine_normalizer_cosets.json.gz` | `d75d4cc8ae0a4975e6547ce570bff17abc41ef0cda7c1ad50cc0cf2b7c1ead44` | `d658b5161de1fb6ec20480079d2d007311c59662` |
| `isomorphic_subgroups_std.json.gz` | `0e9942d43828935c5bc43ab0fa89b5eb163fc9e387da3b7e51846eaffe875efe` | `3eee0dc59f9bce6f8a704ac06be6d887e6550acc` |

The basics contain 527 Hall settings, 230 standard settings, and 3,440 Wyckoff
positions. Use the full stored symmetry operations and full stored orbits;
`mod_centering` subsets are not interchangeable. Preserve JSON array order. In
180 positions the first stored orbit branch differs from the ITA display branch,
so replacing `orbit[0]` with `first_orbit_ita` changes results.

Measured recognition was pinned against spglib 2.7.0. Reproduce its dataset and
build as well as Python's conversion policy:

- call `spglib.get_symmetry_dataset(cell, symprec=tolerance)` with every other
  argument left at the spglib build's default. Here `cell` is exactly
  `(recognition_view.cell.basis.to_floats(),
  recognition_view.sites.reduced_coords.to_floats(), types)`, and `types`
  assigns `1, 2, ...` by the sorted unique species names. For `canonical_asu`,
  this recognition view is the deterministic anonymous P1 frame;
- consume only `number`, `transformation_matrix`, `origin_shift`, `wyckoffs`,
  and `equivalent_atoms` from that dataset. The implementation does not call
  `standardize_cell`, does not pass `no_idealize`, and does not use the dataset's
  idealized standard lattice or positions;
- retain that recognition view's basis, normalized coordinates, species, and
  precision. Use the recovered transformation and origin to project those
  coordinates onto exact vendored Wyckoff charts. The dataset letters and
  equivalence classes are placement/grouping hints; if hinted reconstruction
  raises `ValueError`, repeat exhaustive exact-table placement. There is no
  spglib coordinate averaging or substitution of its standardized geometry;
- recover spglib transformation-matrix entries with
  `Fraction(float_value).limit_denominator(48)` and require absolute error at
  most `1e-12`;
- when an arbitrary origin is not such a small rational, convert
  `float_value` through `f"{value:.12f}"`; reject non-finite entries;
- bridge spglib's origin choice explicitly to the IT standard choice for groups
  48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138, 141,
  142, 201, 203, 222, 224, 227, and 228;
- when recognition retained spglib's found transform, try it before identity,
  then fall back to exhaustive setting handling after any `ValueError`;
- visit candidate Wyckoff positions in `(free_count, multiplicity, letter)`
  order, branches in table order, and setting-lattice cosets in stored order.

The 2.7.0 pin characterizes the tested recognition environment; it does not turn
float tolerance boundaries into a platform-independent exact contract.

Bonding-net canonical strings additionally depend on these conventions:

- operation `n` is the `n`-th branch of the general Wyckoff position of the
  standard setting, identity first, not the International Tables order and not
  `Spacegroup.symmetry_operations` order (group 34 lists the identity last there);
- a row's code is the smallest element of its double coset under the key
  `(n1² + n2² + n3², k, n1, n2, n3)`, with `k` the operation index, taken in
  the listed direction (from the end that comes first in the listing order);
- shapes are compared as strings in site-number order, after the codes; the
  defaults table and the baseline detector are part of the nets version;
- the per-group tables are computed at one exact rational probe point
  (`(13/97, 29/101, 41/103)`), never a float or a structure's own parameters;
- the set of equivalent descriptions is closed in signature (Wyckoff letter map
  and operation relabelling) modulo the group for every space group, and closed
  as actions modulo the group outside groups 2 to 15, pinned by
  `test_actions_close_in_signature`.

## Identity and migration convention

Canonicalization can change computed content IDs because the selected canonical
representation is identity-bearing. Existing stored IDs remain readable and are
never rewritten automatically. A builder must not resume into one output while
mixing canonical conventions. Start a fresh build for the current convention,
or use `canonical_asu_legacy` and `canonicalize_legacy` consistently when
continuing a legacy build.

The explicit protostructure names remain available for callers that already use
them. New callers can use `canonicalize` (or its identical alias
`canonical_asu`) for measured structures, and pass `symmetry="declared"`
to either name for declared exact ASUs.
