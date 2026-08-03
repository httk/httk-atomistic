"""Reading VASP POSCAR and CONTCAR files into a UnitcellStructure

Getting a structure off disk is the first thing most workflows do, and VASP's
POSCAR/CONTCAR is the format a great many of them meet first. `httk.core.load`
is the loading entry point:

`load(path)`
: **The loading entry point.** It picks a reader by file type, transparently
  decompresses `.bz2` and `.gz`, and returns the native `UnitcellStructure`.

*httk-io* must be installed so its POSCAR reader can be registered.

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
"""

import bz2
import tempfile
from pathlib import Path

from httk.core import load

from httk.atomistic import UnitcellStructure

#: This example needs *httk-io* on the path: it registers the POSCAR reader that
#: ``httk.core.load`` dispatches to.
HTTK_EXAMPLE_REQUIRES = ["httk.io"]

POSCAR_TEXT = """SmFeO3 (VASP-5, Direct coordinates)
1.0
5.3982999999999999 0.0 0.0
0.0 5.6 0.0
0.0 0.0 7.6
Sm Fe O
1 1 2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.1 0.2 0.3
0.4 0.5 0.6
"""

CONTCAR_TEXT = """He cell
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
He
1
Direct
0.0 0.0 0.0
"""


def summarize(label: str, structure: UnitcellStructure) -> None:
    print(f"{label}:")
    print("  species          ", [species.name for species in structure.species])
    print("  species_at_sites ", structure.species_at_sites)
    print("  basis            ", structure.cell.basis.to_floats())
    print("  reduced coords   ", structure.sites.reduced_coords.to_floats())
    print("  volume           ", float(structure.cell.volume))


def load_a_plain_poscar(directory: Path) -> None:
    """The ordinary case: a POSCAR file on disk."""
    path = directory / "POSCAR"
    path.write_text(POSCAR_TEXT, encoding="utf-8")
    structure = load(str(path))
    summarize("load('POSCAR')", structure)
    # The first lattice row is float-exact from the string "5.3982999999999999" --
    # it was never rounded through a float on the way in.
    print("  first row is exact:", structure.cell.basis.to_floats()[0] == [5.3982999999999999, 0.0, 0.0])


def load_a_compressed_contcar(directory: Path) -> None:
    """A bz2-compressed CONTCAR: decompression is handled by httk.core.load, not by you."""
    path = directory / "CONTCAR.bz2"
    path.write_bytes(bz2.compress(CONTCAR_TEXT.encode("utf-8")))
    structure = load(str(path))
    summarize("load('CONTCAR.bz2')", structure)


def main() -> None:
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        load_a_plain_poscar(directory)
        print()
        load_a_compressed_contcar(directory)


if __name__ == "__main__":
    main()
