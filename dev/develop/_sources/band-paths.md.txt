# Band-structure paths

{py:func}`~httk.atomistic.band_path` returns the recommended band-structure k-path of a crystal
in the HPKOT convention (Hinuma, Pizzi, Kumagai, Oba and Tanaka, *Comput. Mater. Sci.* **128**,
140 (2017), [doi:10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015)).
It reproduces SeeK-path 2.2.2 without depending on it; the function registers the paper with
`httk.core.credits`.

## What it returns

A {py:class}`~httk.atomistic.BandPathResult` holds:

- `structure`: the HPKOT primitive cell. The k-point coordinates refer to the reciprocal basis of
  *this* cell, so a band-structure calculation must use `result.structure`, not the input cell.
- `points`: a read-only mapping from SeeK-path's ASCII labels (`GAMMA`, `X`, `X_1`, `W_2`, ...)
  to exact fractional reciprocal coordinates.
- `path`: the ordered `(start, end)` label segments.
- `bravais_lattice` and `bravais_lattice_extended`: for example `cF` and `cF2`.
- `transform`: the integer row-convention matrix with
  `basis(structure) = transform * basis(primitive)`, where `primitive` is
  {py:func}`~httk.atomistic.primitive_cell` of the input. Its determinant is `+1`, or `-1` when the
  conventional basis was left-handed (the result basis is always right-handed). For `mC` and `oA`
  the HPKOT primitive matrices differ from those of `primitive_cell`.

## Example

```pycon
>>> from httk.atomistic import ASUStructure, WyckoffSite, band_path
>>> from httk.core import FracVector
>>> species = [
...     {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
...     {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
... ]
>>> asu = ASUStructure(
...     [["5.64", 0, 0], [0, "5.64", 0], [0, 0, "5.64"]], 225,
...     [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")],
...     species,
... )
>>> result = band_path(asu)
>>> result.bravais_lattice_extended
'cF2'
>>> result.path
(('GAMMA', 'X'), ('X', 'U'), ('K', 'GAMMA'), ('GAMMA', 'L'), ('L', 'W'), ('W', 'X'))
>>> result.points["W"].to_fractions()
[Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)]
>>> len(result.structure.sites)
2
```

## Writing a VASP KPOINTS file

`httk.atomistic.integrations.vasp.io.write_line_mode_kpoints` writes the points and path as a
VASP line-mode KPOINTS file. Use it with a calculation on `result.structure`; the decimals are a
rounded presentation of the exact points.

```pycon
>>> import io
>>> from httk.atomistic.integrations.vasp.io import write_line_mode_kpoints
>>> out = io.StringIO()
>>> write_line_mode_kpoints(out, result.points, result.path, divisions=40, comment="NaCl")
>>> print("".join(out.getvalue().splitlines(keepends=True)[:8]), end="")
NaCl
40
Line-mode
Reciprocal
  0.000000000000  0.000000000000  0.000000000000 ! GAMMA
  0.500000000000  0.000000000000  0.500000000000 ! X

  0.500000000000  0.000000000000  0.500000000000 ! X
```

## Which symmetry is used

The space group is the one the structure holds. An {py:class}`~httk.atomistic.ASUStructure` uses
its declared group as stored, without spglib. A raw structure (for example a
{py:class}`~httk.atomistic.UnitcellStructure` built from an expanded cell) has its group
recognized, which needs the optional spglib dependency. To get the detected symmetry of a
structure that declares one, pass a raw structure built from its expanded cell instead.
`tolerance` and `limit_denominator` are passed on to recognition.

## Exactness and limits

Selection rules, free parameters and k-point coordinates are evaluated exactly from the rational
metric of the conventional cell, with no idealization of the input: the result for a noisy
cell agrees with SeeK-path to within the recognition tolerance, but its lattice parameters are
the stated ones. Refused inputs:

- special-angle surd metrics (for example `beta = 135` degrees), whose Gram matrix is not rational;
- magnetic structures (any site moments), since nuclear-symmetry paths are not valid for them;
- whatever `primitive_cell` refuses (assemblies, crystal-axis moments, non-3D-periodic cells).

## Time reversal

With `time_reversal=False` and a group without inversion, every point `X` other than `GAMMA`
gains a partner `X'` at `-X`, and the path is repeated with primed endpoints. Non-magnetic
calculations, including spin-orbit coupling, keep time-reversal symmetry, so the default is
right for them. Pass `time_reversal=False` only when the calculation itself breaks it, for
example an applied magnetic field or magnetic order imposed on a structure stated without
moments.

## Compatibility with SeeK-path

When the declared space group is the full symmetry spglib detects, the extended symbol, labels,
path and points are identical to SeeK-path's, which was checked differentially over the space
groups. Lattice choices are exact and deterministic. Where
spglib's choice depends on the tolerance or the input (ties between equal-length axes,
near-degenerate monoclinic or triclinic cells, and, in space groups 6-9, the choice between two
equally short cells related by a 2-fold rotation about `b`), *httk-atomistic* may return an
equivalent but differently oriented cell; the points still refer to `result.structure`.
