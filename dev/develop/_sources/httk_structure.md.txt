# Exact structure files (`.httkstruct`)

The `httk-structure` format stores one structure losslessly: for every
`UnitcellStructure` `s`, `httk.core.load(path)` of a saved file gives a structure
equal to `s` with the same content id. It keeps everything that makes up the
structure — the exact cell (its `scale` and `unscaled_basis` factoring, irrational
`sqrt` entries included), reduced coordinates, the stated precisions, species
with occupancies, masses, charges, spins and labels, site moments, assemblies,
symmetry and composition declarations, and the source metadata. Use it to keep a
structure exactly, for example as the recorded input of a calculation; use
{doc}`cif` or {doc}`poscar` to exchange structures with other software. The format
is specific to *httk₂*.

```python
from httk.core import load, save

save(structure, "input.httkstruct")      # any httk structure; ASUs are expanded to the unit cell
same = load("input.httkstruct")          # -> UnitcellStructure, == structure
save(structure, "input.httk.json", format="httk-structure")  # any other filename
same = load("input.httk.json", format="httk-structure")
```

The loader dispatches on the `.httkstruct` extension (also compressed, as in
`.httkstruct.gz`); *httk-core* dispatches on the final suffix only, so a
`.httk.json` name would claim every `.json` file and is instead reached through
`format="httk-structure"`. Values that are not *httk₂* structures (spglib-like
tuples, ASE or pymatgen objects) are refused rather than converted silently: pass
`UnitcellStructureView(value)` to make that conversion explicit. Only the structure
is kept: a live `VASPStructure`'s POSCAR-specific state (its comment line, selective
dynamics flags and Cartesian form) is not part of the structure and is not stored.

A file is one deterministic JSON object (sorted keys, ASCII), so the same structure
state gives identical bytes. Exact numbers are strings with one canonical spelling:
rationals as `"-3"` or `"173/97"` in lowest terms, and irrational values as sums of
at most four `R*sqrt(K)` terms with squarefree radicands `K <= 1000`, e.g.
`"1/2*sqrt(3)"` or `"1-3/2*sqrt(3)"` (crystallographic cells need only `sqrt(2)`,
`sqrt(3)` and `sqrt(6)`). Floats appear only for species masses, which are floats in the
structure model too. Reading never evaluates or imports anything: the reader
rejects unknown members, unknown versions, wrong shapes, non-canonical or
oversized numbers, and malformed JSON with a `ValueError`. Its size limits keep any
accepted file cheap to load: at most 1024 characters per number (256 in the cell) or
other string, at most 192 symmetry operations of at most 64 characters each, and no
cell whose scale or determinant nearly cancels to zero. Saving refuses the same. The normative schema is
the `httk.atomistic.io.httk_structure` module docstring.
