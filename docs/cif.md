# Reading and writing CIF files

*httk-atomistic* ships the CIF/mCIF parser, reader and writer stack under
`httk.atomistic.io.cif`, and registers its readers with *httk-core* through
`httk.registry.io.atomistic`. Importing `httk.core` therefore discovers the
`.cif` loader, so `httk.core.load` can dispatch a CIF file. `.cif`, `.mcif` and
their compressed forms (`.cif.gz`, `.cif.bz2`) all dispatch here.

Plain `load(path)` returns the native `ASUStructure` — a structure held as its
asymmetric unit, expanding to a full cell on demand (see {doc}`structures` and
{doc}`asu`). Pass `raw=True` to get the neutral parsed CIF payload instead, when
you want the file's contents without interpreting them as a structure:

```python
import httk.core  # discovery registers the ".cif" loader

payload = httk.core.load("structure.cif", raw=True)
block = payload["blocks"][0]           # one neutral asymmetric-unit mapping per structural block
print(payload["header"])               # the file's leading comment lines
print(block["symbols"])                # e.g. ["Na", "Cl"]
print(block["cell_parameters_exact"])  # ("a", "b", "c", "alpha", "beta", "gamma") as verbatim tokens
```

## The neutral CIF payload

The neutral payload is a mapping with `format` `"cif"` (`"mcif"` for magnetic
CIFs), a `blocks` list holding one asymmetric-unit mapping per structural block,
`unparsed` reasons for blocks that have atom sites but cannot be interpreted, and
the verbatim `header`. Numeric values are kept as strings; where the file carries
`_httk_*_exact` companion tags those exact tokens are preferred, so no precision
is lost at the I/O layer.

Two conveniences smooth over real-world files:

- **Inferred element symbols.** `_atom_site_type_symbol` is optional in the CIF
  core dictionary. When it is absent, each site's element is inferred from the
  leading element run of its `_atom_site_label` (`"MgM1"` → `Mg`), with a
  `RuntimeWarning`. A label whose prefix names no element is not guessed at: its
  block cannot be interpreted, so `load` omits it from `blocks` and records the
  reason in `unparsed` (the underlying parser raises a `ValueError` that the
  loader catches per block).
- **Repair.** Passing `repair=True` enables a bounded set of warning-emitting
  repairs and stamps `repair=True` on neutral payloads. The low-level reader drops
  malformed *auxiliary* loops whose column counts do not line up and retries legacy
  non-UTF-8 path inputs as Latin-1. During `load`, the structure adapter additionally
  ignores invalid declared Wyckoff metadata in favor of the coordinates and clamps
  an individual refined occupancy no more than `0.05` outside `[0, 1]` to the nearest
  boundary. Larger violations remain errors. Strict loading rejects each of those cases.

### Partial occupancy and disorder

Site occupancy is represented without discarding chemistry. When several atom-site rows
generate exactly the same symmetry orbit, the reader combines their elements, occupancies,
charges, and source labels into one mixed `Species`. When a site's total occupancy is below
one, the remaining fraction is represented by an explicit `"vacancy"` constituent. A
co-located total above one outside its stated precision remains invalid for an ordinary CIF
even in repair mode; no constituent is silently dropped. For the moment-free spatial report
of an mCIF, repair mode may instead normalize a co-located mixture whose total is at most
`1.05` and omit
a mass channel declared for only some constituents. Both lossy projections emit warnings and
leave the native magnetic structure unchanged.

Orbits that only partly overlap remain invalid, because they do not describe one shared
crystallographic site and cannot be combined as a species composition.

The CIF writer emits one atom-site row per non-vacancy constituent, preserving occupancies,
source labels, integral charge spellings, isotope/pseudo-site labels, and declared masses.
Read→write→read is covered over the disorder fixture corpus. State without an exact CIF
channel—fractional charges, spins, attached species, assemblies, a net structure charge, or
an independently declared composition—is rejected rather than projected away.

### Atom-type symbols and isotopes

The CIF core dictionary's standard `_atom_type_symbol` values are interpreted as their
elements and optional oxidation states. Both magnitude-before-sign (`Fe3+`) and the common
sign-before-magnitude spelling (`Fe+3`) are accepted when the remaining token is an element.
The widespread isotope symbols `D` and `T` become
hydrogen constituents with species labels `D` and `T` and default masses 2.008 and 3.0160
u. An `_atom_type_mass` or `_atom_type.atomic_mass` table overrides those defaults. `X`
maps to OPTIMADE's non-chemical `"X"`; `Vac`, `Va`, and `vacancy` map to `"vacancy"` with
zero mass.

Any other CIF-valid type symbol remains readable in strict mode. The reader emits one
warning per distinct unrecognized symbol, represents its chemistry as `"X"`, and preserves
the symbol without a charge suffix in the aligned species label. This covers conventional
pseudo-sites such as `M`, `R`, `LP`, and `Lp`, as well as arbitrary values such as `dummy`
or `FeNi`, without pretending that they name chemical elements.

### Declared Wyckoff data

The modern CIF atom-site declarations `_atom_site_site_symmetry_multiplicity`
(International Tables multiplicity) and `_atom_site_site_symmetry_order` (the
site-symmetry order) are honored when identifying Wyckoff positions. The deprecated
`_atom_site_symmetry_multiplicity` tag is never parsed: if it is the only
multiplicity-like tag in a block, httk ignores it and emits one info-level note for
that block because legacy values are ambiguous between the two conventions. If a
modern declaration is present as well, the deprecated tag is ignored silently.

## Lower-level API

The runnable {doc}`/examples/parse_cif` walks the lower-level `read_cif`,
`parse_cif_float` / `parse_cif_int` and `write_cif` API in full. `read_cif`
parses a CIF into a `(data_blocks, header)` tuple of raw named blocks without
interpreting any tag's meaning; `read_cif_asus` interprets the blocks into
asymmetric-unit mappings; and `write_cif` reconstructs a CIF, including its
`loop_` sections, from that neutral form. `read_cif` and `read_cif_asus` are
importable from `httk.atomistic.io.cif`; `write_cif` lives in
`httk.atomistic.io.cif.cif_writer` and the parse helpers in
`httk.atomistic.io.cif.cif_parser`.
