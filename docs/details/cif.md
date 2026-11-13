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
| Malformed auxiliary loop with unequal column lengths | Reject; structure loading drops a malformed `_geom_bond_` loop and warns | Drop the loop and warn |
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

Wyckoff matching uses each site's own stated uncertainty. A site whose coordinate tokens
make no precision claim is matched with only the cell-precision floor, so another site's
large ESD cannot loosen its match. The structure's shared coordinate precision remains
the coarsest claim across sites; it is a conservative bound on the entire site list.
A declared Wyckoff letter is matched with the half-digit rounding bound first. Only when
no position satisfies it does matching fall back to a one-last-digit bound, which admits
truncated decimals such as `0.6666` for `2/3`; undeclared matching keeps the half-digit
bound.

Recognition tolerances are capped strictly below half the nearest-site separation. The cap
includes a small numerical margin because the later squared-distance calculation rebuilds
the same Cartesian distance through matrix arithmetic. This prevents two distinct sites on
opposite sides of a special position from both being snapped onto it.

Cell lengths and angles are retained as their exact CIF parameters. For angles outside the
exact surd trigonometric set, the Cartesian basis remains the cell backend's documented
deterministic rational approximation, in which the rational cosines and sines (0, ±1/2, ±1) are
not approximated, so right angles stay exact; the original parameter backend remains
recoverable.

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

## Bond loops

A block with a `_geom_bond_` loop always loads as a structure with that bond network, its
`bonds` (see {doc}`asu`); there is no option to ask for it. A block without the loop gives
`bonds=None` (no network known), and a loop without rows gives `()`. Only
`_geom_bond_atom_site_label_1`, `_geom_bond_atom_site_label_2`,
`_geom_bond_site_symmetry_1`, `_geom_bond_site_symmetry_2` and `_geom_bond_distance` are
read; the other columns are ignored.

- **Symmetry codes.** `n_klm` is operation `n` of the block's own operation list followed
  by the translation `(k-5, l-5, m-5)`; `n` alone has no translation; `.`, `?` or a missing
  column is the atom as listed. `n` names an operation by its id
  (`_space_group_symop_id`, `_space_group_symop.id` or `_symmetry_equiv_pos_site_id`) when
  the file gives ids, and by its 1-based position in the list otherwise. Any other code needs
  the block's own operations: it is unreadable when the block lists none (repair may generate
  them from a Hall symbol) or gives ids that do not name them one to one.
- **Labels.** A label matches an atom-site label exactly, else the one label equal to it
  ignoring case (DDLm types `_atom_site.label` as a case-insensitive code).
- **Atoms are located through the reader's own changes.** A label names its atom-site row,
  and the row names the site the reader made from it after snapping to a special position,
  combining co-located disorder and removing repeated orbits. The bonded atom is that site's
  orbit point nearest the coordinate the row wrote (found in floating point among the exact
  orbit points), so a row written outside the unit cell keeps its translation, and every row
  of a disordered site maps to the combined site, whichever orbit member it was written at.
- **Distances are checked.** Each bond's distance, computed from the loaded structure, must
  equal `_geom_bond_distance` within the precision the file states, which includes the
  precision of the coordinates it is computed from. The tolerance is `d_tol + δ₁ + δ₂`:
  `d_tol` is three standard uncertainties, but never less than one unit in the last written
  digit, and that unit alone when no uncertainty is given; `δₖ` is `(|a| + |b| + |c|)·pₖ`, with
  `pₖ` one unit in the last written digit of the coarsest coordinate of bond end `k`'s
  atom-site row; a coordinate with at most one decimal place, or an exact one, states no
  precision and contributes nothing. A constrained distance, such as a riding hydrogen's
  `0.9300` without an uncertainty, is thereby accepted when the rounded coordinates explain
  the difference. A distance of `?` or `.`, or a loop without the distance column, is not
  checked.
- **An unreadable loop never stops loading.** An unknown code form, an operation the block
  does not define, a resulting operation outside the group, a label that names no row,
  several rows or a dummy row, a bond of an atom to itself, a distance that is not a finite
  decimal number, a distance mismatch, or bond columns of different lengths log one warning
  naming the block, the row (its labels, codes and distance) and the reason; the structure
  loads with `bonds=None`. A bond loop whose value count does not fit its columns is dropped
  with a warning even without `repair=True`. A loop whose rows hold only `?` and `.` carries
  no information and gives `bonds=None` without a warning.

A CIF bond list is the refinement program's choice, not a chemical model: it typically
lists bonds by distance between atoms the author asked about, often without the bonds to
hydrogen. The bonds are taken as the file gives them and become the `~` bonds of the
protonet labels ({doc}`../prototypes`); CIF has no shape item, so the shapes in those labels
come from the baseline detector (see {doc}`asu`). Missing bonds are not completed, but a network with at
least one bond that leaves Wyckoff sites without a bond logs one warning naming the block and
counting those sites per element, such as `17 H sites have no bond in the listed loop`. Sites
are counted as Wyckoff sites (orbits), not atoms. A site is named by its elements other than
vacancy, sorted and joined with `/` for a mixed-occupancy site (`Co/Fe`), or by its species
name when no element remains. An empty loop, `()`, gives no warning, and the warning never stops
a load.

```python
from httk.core import load
from httk.atomistic import ProtonetstructureView, canonical_protonetstructure

structure = load("example.cif")
print(structure.bonds)
print(canonical_protonetstructure(structure).label)
print(ProtonetstructureView(structure).label)
```

Bond types and orders, `_chemical_conn_bond_` loops, and writing a network back to CIF are
not supported.

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

The writer never changes a value. Every coordinate and cell parameter is written exactly as
stored (terminating values as their exact decimal, others at sixteen significant digits), so a
relaxed structure's float noise such as `0.3549999999999969` is written with all its digits; a
coordinate is only padded with zeros to the width its precision resolves. For clean output,
rationalize first, for example `save(structure.rationalized(), path)` for a CONTCAR loaded with
`precision=5e-4` (see [Rationalizing explicitly](precision.md#rationalizing-explicitly)). Either way the precision
read back is the coarsest of the written digits, so a file whose every coordinate carries float
noise reads back at about `1e-16`; pass a `tolerance=` to {py:func}`~httk.atomistic.recognize_asu`
when recognizing the symmetry of such a file.

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

## Lossless reading and `CIFStructure`

`load()` builds a structure and keeps only what the structure model has a channel for. To keep
everything a CIF file says, read it losslessly instead. A {py:class}`~httk.atomistic.CIFStructure`
holds one data block exactly as written: its data items, loops, comments and save frames in
written order, its data names in their written case, its numbers as written, the file's header
comment lines and the comments before the `data_` line. Every data block is one, also a block
without atoms or a block that is not a valid structure. It is immutable, and two blocks with the
same content are equal.

```python
import httk.core
from httk.atomistic import ASUStructureView, CIFStructure, read_cif_structures

cs = CIFStructure.from_cif("example.cif")      # a file with exactly one data block
blocks = read_cif_structures("several.cif")    # one CIFStructure per data block, in order

print(cs.value("_journal_name_full"))          # lookup ignores case
print(cs.numbers("_cell_length_a", "_cell_length_b", "_cell_length_c"))
asu = ASUStructureView(cs).resolve()           # equals httk.core.load("example.cif")
httk.core.save(blocks, "copy.cif")             # an equivalent file
```

`load()` is unchanged: a lossless block is always asked for explicitly. The dialect is
`"mcif"` for a `.mcif` path or stream name (also compressed) and `"cif"` otherwise, or set with
`dialect=`.
A `"cif"` block is CIF1, or full CIF2 (lists, tables, triple quotes) when line 1 is the CIF2
magic code `#\#CIF_2.0`; an `"mcif"` block uses the magnetic CIF reader's syntax, CIF1 plus
lists, whatever its header says. The lower-level form is `read_cif(source, lossless=True)`,
which returns the syntax-only {py:class}`~httk.atomistic.io.cif.CIFBlock` objects that
`write_cif` writes back.

**Values.** A value is a string, a text field (its raw lines between the `;` delimiters), a
number, a gap (`?` unknown or `.` inapplicable), or a CIF2 list or table. A quoted `'1.5'`, `'?'`
or `'.'` is a string, and the writer quotes whatever would otherwise read as something else. A
number is a {py:class}`~httk.core.DecimalScalar`: its exact decimal value, its standard
uncertainty and its spelling, so `0.50`, `0.5`, `1.2345(6)`, `12(1)`, `1.5e-3`, `007` and
`+1.0` all come back as written. `numbers(...)` gives cell parameters, coordinates,
occupancies, displacement parameters or bond distances as a
{py:class}`~httk.core.DecimalVector`; `FracVector(...)` of it is exact.

**The content rule.** This is the one definitive statement of what a block's content is; equal
content means equal blocks and equal content ids. A block's content is its name, its dialect,
the file's header, its leading comments, and its items in order: data names, values and
comments in their places. Two files are equivalent when they read to the same blocks in the
same order. Whitespace and quoting are not
content: whitespace and line breaks outside text fields and strings, quote style, line ends,
the encoding, a byte-order mark, trailing comment whitespace and blank lines do not matter,
except that these four are content:

1. **A comment's line placement:** a comment inline after a token (`_x 1 # c`) differs from
   the same comment on its own line.
2. **Quotes that change a value's kind:** `'1.5'` is a string and `1.5` a number; `'?'` and
   `'.'` are strings and `?` and `.` gaps.
3. **Where the file header ends:** a blank line before or among the file's leading comment
   lines ends the header (an empty header when the file starts with a blank line) and makes the
   later comments the first block's leading comments.
4. **Runs of spaces inside a pragmatically joined value** (D1 below): `P  21/c` differs from
   `P 21/c`.

Inside a CIF2 list or table a comment's placement is not content: the comment moves to just
after the value (D5 below).

Writing, reading and writing again gives identical text.

**Conversion.** {py:meth}`~httk.atomistic.CIFStructure.resolve`, or constructing a structure
view such as `ASUStructureView(cs)` or `UnitcellStructureView(cs)`, gives what `load()` gives
for the block: an `ASUStructure` with its bond network for `"cif"`, a `SymopsStructure` or
`ModulatedStructure` for `"mcif"`. `resolve(repair=True)` and
`resolve(allow_large_cif_uncertainty=True)` correspond to the `load()` options. A block that
is valid CIF but not a valid structure (no cell, an occupancy above one, operations that
contradict the declared group, labels repeated at one position, an element that cannot be
inferred, a coordinate that is not a number) reads and round-trips; converting it raises the
error `load()` raises. Two deliberate differences: a data name the structure reader consumes
(the cell, symmetry, atom-site and magnetic names; not the bond-loop names, so a bond loop never
stops a conversion) may occur only once, also across save frames, where `load()` silently keeps
the last occurrence; and in a `"cif"` block, a CIF2 list or table as the value of a name the
structure reader reads (bond-loop names included) raises. Repeated names the structure reader
does not consume are kept and ignored.

**Saving.** `httk.core.save(cs, "x.cif")` and `save([cs, ...], "x.cif")` write the blocks
losslessly, with the header once at the top; a `.mcif` destination needs `"mcif"` blocks and a
`.cif` one `"cif"` blocks. `header=` is refused, `max_line_length=` wraps loop rows, and
`approximate` and `exact_companions` have no effect. Blocks written together must carry the
same header and dialect, and a block other than the last cannot end with an own-line comment
(it would read back as the next block's leading comment).

### Storing and serving CIF blocks

{py:class}`~httk.atomistic.CIFStructureRecord` (registered as `atomistic-cif-structure`, entry
family `structures`) stores a `CIFStructure`: its block name, dialect, a `canonical_form`
version, and the header, leading comments and items as one canonical JSON text. Equal blocks
have equal content ids: files that are equivalent under the content rule above give the same
id, and any difference that rule counts as content gives a different one. The stored text
cannot be queried by
data name. Each block of a file is an independent entry: keep the order and ids of a file's
blocks yourself, and identical blocks deduplicate. MongoDB's 16 MB document limit bounds one
block.

`store.save(cs)` stores a block. `httk.core.coerce(record, CIFStructure)` turns a fetched record
back into the block; the record also converts like any structure source, so
`ASUStructureView(record)` builds the structure, and `httk.core.save(records, "x.cif")` writes
the blocks back. A record built under another `canonical_form` is refused with a request to
rebuild the database. When serving over OPTIMADE, a CIF block is never converted: the in-memory
provider and every store backend serve the same row, with the record's `id`, `immutable_id`
and `last_modified`, `structure_features` `[]` and every other structure property null.
`last_modified` filters and sorts; `structure_features` filters as an empty list.

### What lossless reading normalises or rejects

These are the deviations from a round trip, and the inputs where it differs from `load()`:

- **D1, joined values** (under `pragmatic`, the default). A bare value followed on its line by
  more words is joined into one string as the fast reader joins it, up to the next data name
  or keyword and cut at `#` (the rest is an inline comment). It is written back quoted. A join
  that cannot be quoted in the syntax raises.
- **D2.** Lossless reading has no other `pragmatic` behaviour: a Unicode minus stays a string
  and a blank first text-field line stays as written.
- **D3.** `repair` does not drop a malformed loop; a loop whose values do not fill its rows
  raises.
- **D4.** Under `repair`, a file that is not UTF-8 is decoded as Latin-1 with the warning
  `load()` gives. Output is UTF-8 with `\n` line ends; a byte-order mark is dropped.
- **D5.** A comment inside a CIF2 list or table moves to just after the value.
- **D6.** The header is written once per file.
- **D7.** A number-shaped token whose exponent is outside `Decimal`'s range is a string; it is
  written bare and reads back as the same string.
- **D8.** Inputs `load()` accepts but lossless reading rejects or reads differently: a value
  without a data name, a data name without a value (also before `loop_` or a block code), `stop_`
  or `global_` as a value, an unterminated text field, content before the first `data_`, an
  unwritable string such as a D9 value holding both `' ` and `" `, and a D1 cut inside a quote
  raise; `ab#c` is one word (the fast reader cuts it at `#`), and a quote directly before `]`
  in an mCIF list, an upper-case keyword after a joined value and a multi-line mCIF list are
  read correctly where the fast reader does not. For such inputs conversion follows the correct
  reading, so it can differ from `load()`.
- **D9, an unterminated quote** (under `pragmatic`, CIF1 and mCIF only). A quote left open at
  the end of its line reads as the rest of the line, opening quote included, as the fast reader
  reads it; the value round-trips, its quoting does not. Otherwise it raises.

Fast-reader faults that were fixed (they affected `load()`, not lossless reading; on the
repository fixtures and the sampled COD and MAGNDATA files no loaded structure changed):

- **FP1.** A comment in a loop header corrupted the loop. A comment (a `#` that begins a token,
  outside quoted strings and text fields, to the end of the line) and a blank line are now
  ignored on a `loop_` line and among the loop's names, as everywhere else.
- **FP2.** A loop name followed by a value, or two names, on one line became one column name.
  Header lines are now split into tokens: every data name is a column, and the first token that
  is not a data name starts the loop's values.
- **FP3.** Splitting a joined value was quadratic in the run of spaces that follows it; it is
  now linear. The data name, `data_` or `loop_` that ends a joined value on its line is now read
  instead of being dropped, and an empty block code there (`data_` alone) raises "empty CIF
  data block name". Accepted limit: many items on one line still take time quadratic in the
  line's length, as each item copies the rest of the line (about half a second for 20 000
  items on one line).
- **FP4.** In a loop, a data name on the same line after a quoted value or a
  closing `;` was read as a value. It now ends the loop, also in a loop that structure reading
  skips, so a structural item there (such as `_cell_length_a`) is kept.

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
| `read_cif(..., lossless=True)` | Parse text into lossless `CIFBlock` objects |
| `CIFStructure.from_cif()` / `read_cif_structures()` | Read lossless, structure-like blocks |
| `read_cif_asus()` | Produce the neutral ordinary-CIF payload |
| `read_mcif_asus()` | Produce the neutral magnetic-CIF payload |
| `asu_structure_from_cif()` | Convert one neutral ordinary block into an `ASUStructure` |
| `symops_structures_from_mcif()` | Convert neutral mCIF blocks into native magnetic or modulated structures |
| `write_cif()` | Write low-level CIF blocks, or lossless `CIFBlock` objects |
| `httk.core.load()` / `save()` | Registered high-level I/O |

Use the neutral payload when building a format tool that should not take on atomistic
interpretation. Use `load()` when the desired result is a structure model.
