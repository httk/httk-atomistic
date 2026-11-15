# Reading VASP POSCAR and CONTCAR files into a UnitcellStructure

Getting a structure off disk is the first thing most workflows do, and VASP's
POSCAR/CONTCAR is the format a great many of them meet first. `httk.core.load`
is the loading entry point:

`load(path)`
: **The loading entry point.** It picks a reader by file type, transparently
  decompresses `.bz2` and `.gz`, and returns the native `UnitcellStructure`.

**The values stay exact.** The file's numbers are read as *strings* and turned
into exact rationals, never into floats first, so a lattice row written as
`5.3982999999999999` is that exact decimal and a coordinate written as `1/3`
would be that exact third. Direct coordinates become reduced coordinates
verbatim. Cartesian coordinates are converted exactly as `cart * basis.inv()`,
and the VASP universal scaling factor cancels in that expression (it scales the
lattice vectors and the Cartesian positions alike), so the reduced coordinates
come out exact either way. The one place exactness is unavoidably lost is a
*negative* scale line, which encodes a target cell **volume**: the resulting
overall scale is a cube root, which leaves the squarefree-radical field, so it
is a deterministic rational approximation — the basis rows themselves stay
exact.

VASP-4 POSCAR files have no species line, so the elements are simply not in the
file; the loader raises `ValueError` rather than guessing.

```{literalinclude} ../../examples/load_from_poscar.py
:language: python
:lines: 27-
```
