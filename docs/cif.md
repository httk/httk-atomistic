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
- **Autocorrect.** Passing `autocorrect=True` to `load` (or to `read_cif` /
  `read_cif_asus`) drops a malformed *auxiliary* loop — one whose column counts do
  not line up and whose tags are not a protected structural family — warning about
  each repair instead of refusing the file, and stamps `autocorrect=True` on the
  payload. Without it, such a loop is a hard `ValueError`.

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
