# CIF and mCIF reading in detail

This page describes the complete path from CIF text to httk structure objects, including
the deliberately bounded repairs used for imperfect files in large crystallographic
databases. The short practical introduction is {doc}`../cif`; the lower-level parsing
example is {doc}`../examples/parse_cif`.

The central policy is:

- retain the source text and exact numerical meaning whenever the model has a channel for it;
- repair only common deviations with a deterministic interpretation;
- warn whenever repair changes or omits source data;
- reject cases where choosing a repair would choose chemistry, symmetry, or a coordinate
  setting on the caller's behalf.

## The reading pipeline

Reading is split into layers so syntax, neutral file data, and atomistic interpretation do
not become one inseparable parser.

1. `httk.core.load()` selects the registered reader from the filename. `.cif`, `.mcif`, and
   recognized compressed forms such as `.cif.gz`, `.cif.bz2`, and `.cif.xz` dispatch to the
   CIF stack in `httk.atomistic.io.cif`.
2. `read_cif()` tokenizes CIF text into normalized data blocks. It does not construct domain
   objects.
3. `read_cif_asus()` or `read_mcif_asus()` selects structural blocks and creates a neutral
   payload made only of mappings, sequences, strings, and numbers. Blocks with atom sites
   that cannot be interpreted are retained in `unparsed` with their errors.
4. The atomistic format adapter converts an ordinary CIF payload into an `ASUStructure`.
   A commensurate mCIF becomes a `SymopsStructure`; an incommensurate mCIF becomes a
   `ModulatedStructure` descriptor.
5. A view performs any further representation work lazily. `UnitcellStructureView` expands
   declared operations. `ASUStructureView` recognizes symmetry only when the source did not
   already provide a native asymmetric-unit representation.

Pass `raw=True` to stop after step 3:

```python
from httk.core import load

payload = load("example.cif", raw=True, repair=True)
block = payload["blocks"][0]

print(payload["format"])                 # "cif"
print(block["cell_parameters_exact"])   # exact central tokens
print(block["positions_exact"])
print(block["symbols"])
print(payload["unparsed"])
```

The source block name is carried into the atomistic bridge in strict and repair modes so
errors identify the actual `data_` block.

## Strict mode and repair mode

`load(path)` is strict by default. `load(path, repair=True)` enables the repairs below.
The `httk symmetry info` command is intentionally the other way around: it enables repair
by default for diagnostic work, accepts any registered structure input format, and provides
`--no-repair` for strict reading.

Repair is not a general “make this file work” switch. It is a fixed set of transformations:

| Input issue | Strict mode | `repair=True` |
| --- | --- | --- |
| Invalid UTF-8 in a path input | `UnicodeDecodeError` | Retry the complete file as Latin-1 and warn |
| Malformed auxiliary loop with unequal column lengths | Reject | Drop the loop and warn |
| Malformed protected structural loop | Reject | Reject; structural data is never discarded |
| Missing symmetry-operation loop with a valid Hall symbol | Reject | Generate the exact setting's operations and warn |
| Unknown declared Hall/IT symmetry but usable operations | Reject the contradictory declaration | Ignore the declaration, identify the operations, and warn when the documented repair path applies |
| Invalid modern Wyckoff metadata | Reject | Ignore the metadata, use coordinates, and warn |
| One occupancy within `[-0.1, 1.1]` outside `[0, 1]` | Reject | Clamp to the nearest boundary and warn |
| Larger occupancy violation | Reject | Reject |

The Latin-1 retry applies only when the reader owns a filesystem path and can reopen it.
An already-open text stream has already made its decoding choice.

Malformed-loop repair is restricted to auxiliary data. Atom positions, cell parameters,
symmetry operations, occupancies, magnetic moments, and other protected structural families
are not candidates for removal.

## Exact numbers and source precision

CIF decimal text is not first converted to binary floating point. A coordinate written as
`0.3333` enters the atomistic layer as the exact rational `3333/10000`. A writer-provided
`_httk_*_exact` companion tag takes precedence over its rounded standard display value.

Parenthesized standard uncertainties are retained as precision information. The weaker of
the final written digit and the stated uncertainty controls the tolerance; `5.6402(3)` is
therefore treated as precise to `0.0003`, not `0.0001`.

A plain decimal literal written with zero or one decimal place makes no decimal-precision
claim. In crystallographic tables such literals commonly spell exact special values (for
example `0`, `0.5`, `1.0`, or the signed `-0.5`) rather than measurements known only to the
final written digit. Two or more decimal places are significant: `0.25` claims a decimal step
of `0.01`, and `0.50` likewise claims `0.01`. Exponent notation is always an explicit
precision statement, so the core decimal-precision rule applies regardless of how many
mantissa digits are written: `1.2e-3` claims a step of `0.0001`.

Coordinate precision is converted to a Cartesian distance using the cell. A projected
positional uncertainty from 0.1 up to (but not including) one ångström is reported at DEBUG
when the Wyckoff match is unambiguous or trusted from the CIF. It is reported at WARNING
when the Wyckoff letter is missing or untrusted and multiple compatible Wyckoff positions
remain. A projected uncertainty of one ångström or more is a hard safety error because it can
make many unrelated positions plausible. A caller who has inspected the source may opt in explicitly:

```python
structure = load("coarse.cif", allow_large_cif_uncertainty=True)
```

Recognition tolerances are capped strictly below half the nearest-site separation. The cap
includes a small numerical margin because the later squared-distance calculation rebuilds
the same Cartesian distance through matrix arithmetic. This prevents two distinct sites on
opposite sides of a special position from both being snapped onto it.

Cell lengths and angles are retained as their exact CIF parameters. For angles outside the
exact surd trigonometric set, the Cartesian basis remains the cell backend's documented
deterministic rational approximation; the original parameter backend remains recoverable.

## Dummy sites and attached hydrogens

An atom-site row whose `_atom_site_calc_flag` is `dum` declares an atom with no meaningful
coordinates. The row is omitted from the structure's represented sites but retained as a
`Species` definition with its stated occupancy. Its species name appears in
`structure.implicit_atoms`, and `structure_features` consequently contains
`"implicit_atoms"`. Unlike an occupancy probability on a represented site, this value can
be an aggregate count greater than one.

A positive `_atom_site_attached_hydrogens` value is represented on that row's `Species` as
`attached=("H",)` and the matching `nattached` count. Zero or an unknown value adds no
attachment; any represented attachment sets the `"site_attachments"` structure feature.

Writing applies the reverse mapping: unused single-constituent species are emitted as `dum`
rows with coordinates `-1 -1 -1`, and a positive single hydrogen attachment is emitted through
`_atom_site_attached_hydrogens`. Attachment forms without an exact CIF representation remain
serialization errors.

## Symmetry operations and settings

For an ordinary CIF, the operation list is the authoritative description of the file's
coordinate setting. Coordinate expressions accept both lowercase and uppercase conventional
variables, so `x,y,z` and `+X,+Y,+Z` have the same meaning. Coefficients and translations are
parsed exactly.

The normalized operation set is compared exactly with the tabulated space-group settings.
A declared Hall symbol, International Tables number, or recognized Hermann–Mauguin symbol is
a claim checked against those operations; it does not override them silently.

In strict mode, httk does not invent operations when the loop is absent. With `repair=True`,
a valid Hall symbol is the one safe exception: it uniquely identifies the axes, origin, and
setting, so httk generates that setting's tabulated operations and warns. An IT number or
Hermann–Mauguin symbol alone is not used for this repair because it can leave several settings
or origins possible. Likewise, if an operation set matches no tabulated setting, httk does not
search the infinitely many possible affine normalizer transforms. The caller must correct the
source or supply the intended `SettingTransform` explicitly. See {doc}`asu` for the setting model.

Volume-changing setting transforms are handled through all of their lattice cosets. During
recognition, the matched coset is retained, and orbit membership is required to be a
one-to-one match. Duplicate sites in one coset cannot stand in for a missing site in another.

## Wyckoff declarations and coordinate snapping

Modern `_atom_site_site_symmetry_multiplicity` and
`_atom_site_site_symmetry_order` declarations participate in selecting the Wyckoff position.
Under repair, a declaration inconsistent with the coordinates is ignored with a warning and
the position is derived from the coordinates.

The deprecated `_atom_site_symmetry_multiplicity` name is ambiguous in historical files: it
has been used both for International Tables multiplicity and for site-symmetry order. httk
does not interpret it. When it is the only multiplicity-like tag, the reader emits a
debug-level note; when modern metadata is also present, the deprecated field is
ignored silently.

Snapping changes only fixed Wyckoff components. Free parameters retain the exact rational
written by the source unless the caller explicitly requests denominator limiting.

## Atom-type symbols, charges, isotopes, and pseudo-sites

The CIF core dictionary's conventional atom-type symbols are recognized, including their
listed integral oxidation states. Real files use more charge spellings than that list. httk
accepts both magnitude-before-sign and sign-before-magnitude forms whenever removing the
charge leaves an element:

| CIF spelling | Chemical symbol | Charge |
| --- | --- | ---: |
| `Fe4+` | `Fe` | +4 |
| `Fe+3` | `Fe` | +3 |
| `O2-` or `O-2` | `O` | −2 |
| `Cl-` | `Cl` | −1 |
| `Ti0` | `Ti` | 0 |

The historical dictionary spelling `TL` is normalized to the element `Tl`.

Special nonstandard and isotope symbols are represented as follows:

| CIF symbol | `chemical_symbols` entry | Species label | Default mass |
| --- | --- | --- | ---: |
| `D` | `H` | `D` | unstated |
| `T` | `H` | `T` | unstated |
| `X` | `X` | none | unstated |
| `Vac`, `Va`, `vacancy` | `vacancy` | none | 0 |
| `M`, `R`, `LP`, `Lp`, or another unknown token | `X` | source token without its charge spelling | unstated |

The isotopes `D` and `T` carry no invented default mass; the species label already records
the isotope, so a mass is set only when the file states one. An `_atom_type_mass`,
`_atom_type.mass`, `_atom_type_atomic_mass`, or `_atom_type.atomic_mass` table supplies that
mass and is aligned back to every site using the atom-type symbol. This channel is preserved
for ordinary CIF and mCIF, and a write does not emit an `_atom_type_mass` loop the source did
not have. Conflicting or misaligned type/mass loops are rejected.

An unrecognized symbol is readable even in strict mode. It produces one warning per distinct
token and becomes the OPTIMADE non-chemical species `X` with the source token in the aligned
species label. This preserves information without claiming, for example, that `M`, `R`, or
`FeNi` names an element.

When `_atom_site_type_symbol` is absent, which the core dictionary allows, httk infers an
element from the leading element run of `_atom_site_label` and warns, because inferring
chemistry from a label is a guess. `MgM1` can therefore become magnesium. A label such as
`?` has no inferable identity: in strict mode it is rejected rather than guessed, and only
under `repair=True` is it mapped to the non-chemical `X` with a warning.

## Partial occupancy and ordinary-CIF disorder

Occupancy becomes the constituent concentration of a `Species`; it is never used merely as
a Boolean site-presence flag.

- A partially occupied single row gains an explicit `vacancy` constituent for the missing
  fraction.
- Rows whose complete symmetry orbits coincide are combined into one mixed species. Element
  symbols, concentrations, uncertainties, charges, source labels, and masses remain aligned.
- Identical duplicate rows are deduplicated.
- Orbits that overlap only partly are rejected because they cannot describe one shared site.
- A co-located concentration total above one that lies outside its stated-precision interval
  is normalized without repair when the excess is no larger than `1/1000` and every constituent
  has a stated precision, with a DEBUG diagnostic (a total within its stated precision is kept
  unchanged, as before). An
  excess no larger than `1/10` is rescaled with a warning under `repair=True`, or rejected with
  a `repair=True` remedy hint otherwise. Larger excesses are rejected for an ordinary CIF,
  including in repair mode. Choosing which constituent to change is a chemistry decision.

The individual-value repair is deliberately narrow: values in `[-0.1, 0)` clamp to zero and
values in `(1, 1.1]` clamp to one, with a warning. Values outside that band remain errors.

## Magnetic CIF

A commensurate mCIF is represented natively as `SymopsStructure`: the listed rows, exact
spatial or magnetic operations, centering operations, time-reversal flags, species, and
site moments are retained before expansion. Decimal moments are converted to exact rationals
before any operation is applied. Their componentwise decimal steps, ESDs, and
`_atom_site_moment.symmform` strings remain aligned with the listed sites. Both Cartesian and
crystal-axis moment bases are supported.

Native expansion applies every operation and transforms axial moments with the operation
determinant and time reversal. It preserves independent co-located source rows, which is
necessary for magnetic disorder. When stabilizer operations map one source row onto itself,
the reader derives their exact invariant moment subspace. A source central value already in
that subspace is unchanged. Otherwise, a componentwise weighted projection uses half of each
last-decimal step and any explicit ESD; the projection is accepted only when every component
remains within its source claim. The expanded view then carries the exact invariant result,
while `listed_site_moments` retains the literal source values. An incompatibility outside the
source claims remains an error. Recognized linear `symmform` declarations are checked against
the reconciled moment. The declared operations remain authoritative: a contradictory
`symmform` is retained as source metadata and warned about rather than rotating or deleting a
moment that the operations support.

Incommensurate structural or magnetic modulation produces a `ModulatedStructure`. The data is
identified and retained, but ordinary unit-cell/ASU expansion is not currently implemented.

### Spatial reporting of an mCIF

`httk symmetry info` reports ordinary crystallographic symmetry. For a `SymopsStructure` it
therefore builds a moment-free spatial projection rather than treating antiferromagnetic
moments as chemical symmetry breaking:

1. Apply the spatial part of every operation to each listed source row independently.
2. Cluster only images of that same row within a bounded Cartesian tolerance. The implicit
   tolerance follows source precision but is clamped to the interval 0.002–0.05 Å.
3. Replace each rounded cluster by its periodic circular mean, avoiding an origin-boundary
   bias near fractional zero.
4. Compare complete row orbits through a deterministic bipartite perfect matching. Merely
   overlapping or non-bijective orbits are rejected.
5. Align matching constituent orbits and circular-average their positions independently of
   source-row order.
6. Combine their disorder species, then recognize the ordinary spatial ASU.

This *report-only projection* applies the same occupancy tiers as ordinary CIF loading: a
total outside its stated-precision interval with a rounding-level excess no larger than
`1/1000` is normalized without repair when all constituents have stated precisions, with a
DEBUG diagnostic; an excess no larger than `1/10`
is normalized under `repair=True` with a warning and otherwise rejected with a `repair=True`
remedy hint; larger excesses are rejected. It may also omit a mass channel declared for only
some constituents under repair. These projections do not mutate the native magnetic structure.

## Writing and round-trip guarantees

The ordinary CIF writer emits one atom-site row per non-vacancy constituent of a mixed
species. It preserves:

- disorder concentrations and their precision;
- source atom-site labels;
- integral species charges, including a valid retained source spelling such as `P+5`;
- `D`, `T`, `X`, vacancy, and arbitrary pseudo-site labels;
- atom-type masses;
- exact rational cell parameters, coordinates, and occupancies through standard display
  values, optionally with non-standard `_httk_*_exact` companion tags (opt-in; see below).

Read → write → read is tested across the committed disorder CIF corpus. The writer rejects
state for which ordinary CIF has no exact implemented channel: fractional species charges,
species spins or attachments, assemblies, a net structure charge, or an independently
declared composition.

A cell with no exact CIF form — an irrational length or angle, or a rational six-parameter set
that would rebuild a different oriented basis — is written by default: asking to save into CIF
renders whatever the format can hold. `httk.core.save(structure, path, format="cif")` writes the
cell parameters as rounded decimals (twelve significant digits). This is lossy for the
non-representable part — the arbitrary orientation of the basis is not recovered on read-back —
and only the cell parameters are rounded; the fractional coordinates are already exact rationals
and are written unchanged. An already-exactly-representable structure is written identically
whether or not the default applies.

The `_httk_*_exact` companion tags — a non-standard httk extension carrying exact rational tokens
such as `1/3` — are **off by default**, so a saved CIF holds only standard columns and interoperates
cleanly. Callers who want a lossless httk round-trip pass `exact_companions=True`
(`save(structure, path, format="cif", exact_companions=True)`) to emit them.

Callers who need the strict "exact or nothing" guarantee pass `approximate=False`
(`save(structure, path, format="cif", approximate=False)`), which refuses a non-representable cell
instead of rounding it.

The current writer is an ordinary structural CIF writer, not a magnetic-CIF serializer. It
does not provide a magnetic-moment round-trip guarantee; do not use an ordinary `.cif` save
as archival output for native mCIF moments.

## `httk symmetry info`

The command loads through the general structure-reader registry, so CIF, mCIF, POSCAR, and
other registered structure formats share one reporting path:

```console
httk symmetry info structure.cif
httk symmetry info POSCAR
httk symmetry info magnetic.mcif
```

Output uses decimal floats by default. Use `--exact` for rational values. Repairs are enabled
by default for this diagnostic command; use `--no-repair` to reproduce strict loading.
`--recognize` additionally recognizes symmetry from the projected geometry. For mCIF it uses
the same moment-free, disorder-combined spatial projection rather than expanding the raw
magnetic rows as if they were ordinary sites.

## Failures that remain deliberate

The following conditions need source correction or an explicit caller decision:

- missing symmetry operations without a valid Hall symbol and explicit repair;
- operations that identify no tabulated setting without a supplied transform;
- an atom site with neither a type symbol nor an inferable label;
- projected positional uncertainty at least one ångström without explicit opt-in;
- partly overlapping disorder orbits;
- ordinary-CIF co-located occupancies above one;
- gross individual occupancy violations beyond the repair band;
- conflicting atom-type masses;
- magnetic stabilizer constraints incompatible with the componentwise source resolution and
  ESDs;
- incommensurate modulation when an ordinary expanded unit cell is requested.

The current sampled COD instances are recorded in the workspace's `DATA/cod_issues.md`.

## Lower-level entry points

| Function | Role |
| --- | --- |
| `read_cif()` | Parse text into normalized raw blocks and a header |
| `read_cif_asus()` | Produce the neutral ordinary-CIF payload |
| `read_mcif_asus()` | Produce the neutral magnetic-CIF payload |
| `asu_structure_from_cif()` | Convert one neutral ordinary block into an `ASUStructure` |
| `symops_structures_from_mcif()` | Convert neutral mCIF blocks into native magnetic or modulated structures |
| `write_cif()` | Write low-level CIF blocks |
| `httk.core.load()` / `save()` | Registered high-level I/O |

Use the neutral payload when building a format tool that should not take on atomistic
interpretation. Use `load()` when the desired result is a structure model.
