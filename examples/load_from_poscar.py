"""Reading VASP POSCAR and CONTCAR files into a Structure

Getting a structure off disk is the first thing most workflows do, and VASP's
POSCAR/CONTCAR is the format a great many of them meet first. *httk-atomistic*
gives you two entry points, and the split between them is deliberate:

`load_structure(path)`
: **The end-to-end convenience.** It calls `httk.core.load`, which picks a
  reader by file type — and transparently decompresses `.bz2` and `.gz`, so an
  archived `CONTCAR.bz2` needs no special handling — then dispatches on the
  payload's `"format"` tag to the matching structure builder. This is what you
  want in a script.

`structure_from_poscar(mapping)`
: **The bridge alone.** It takes the neutral, string-preserving mapping that
  `httk.io.read_poscar` produces (`{"format": "vasp-poscar", "cell": [...],
  "coords": [...], ...}`) and turns it into a `Structure`. Use it when the
  file has already been parsed, when the mapping came from somewhere other
  than a file, or when you want to edit the mapping before building.

Notably, `structure_from_poscar` **imports nothing from *httk-io***. It only
understands the shape of the mapping. That is what keeps parsing (*httk-io*) and
the domain model (*httk-atomistic*) decoupled: neither distribution depends on
the other, and the neutral mapping is the whole contract between them.
`load_structure` does need *httk-io* installed at run time, though — not to
import it, but because importing `httk.core` walks the `httk.handlers` namespace
package, and that is where *httk-io* registers the POSCAR reader for the
`POSCAR`/`CONTCAR` filenames. Hence the `HTTK_EXAMPLE_REQUIRES` declaration
below, which tells this repository's example smoke test to skip this script when
*httk-io* is not on the path.

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

One incompatibility worth knowing: VASP-4 POSCAR files have no species line, so
the elements are simply not in the file. `structure_from_poscar` raises
`ValueError` rather than guessing.
"""

import bz2
import tempfile
from pathlib import Path

from httk.atomistic import Structure, load_structure, structure_from_poscar

#: This example needs *httk-io* on the path: it registers the POSCAR reader that
#: ``httk.core.load`` -- and therefore ``load_structure`` -- dispatches to.
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

#: The neutral mapping shape: exactly what ``httk.io.read_poscar`` returns, and the
#: only thing ``structure_from_poscar`` knows how to read. Note the numbers are
#: *strings* -- that is what makes the conversion exact rather than float-rounded.
ALREADY_PARSED = {
    "format": "vasp-poscar",
    "comment": "rocksalt-ish",
    "scale": "1.0",
    "volume": None,
    "cell": [["4.0", "0.0", "0.0"], ["0.0", "4.0", "0.0"], ["0.0", "0.0", "4.0"]],
    "symbols": ["Na", "Cl"],
    "counts": [1, 1],
    "cartesian": False,
    "coords": [["0.0", "0.0", "0.0"], ["0.5", "0.5", "0.5"]],
    "selective_dynamics": None,
}


def summarize(label: str, structure: Structure) -> None:
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
    structure = load_structure(str(path))
    summarize("load_structure('POSCAR')", structure)
    # The first lattice row is float-exact from the string "5.3982999999999999" --
    # it was never rounded through a float on the way in.
    print("  first row is exact:", structure.cell.basis.to_floats()[0] == [5.3982999999999999, 0.0, 0.0])


def load_a_compressed_contcar(directory: Path) -> None:
    """A bz2-compressed CONTCAR: decompression is handled by httk.core.load, not by you."""
    path = directory / "CONTCAR.bz2"
    path.write_bytes(bz2.compress(CONTCAR_TEXT.encode("utf-8")))
    structure = load_structure(str(path))
    summarize("load_structure('CONTCAR.bz2')", structure)


def build_from_an_already_parsed_mapping() -> None:
    """The bridge on its own, for a mapping you already have in hand."""
    structure = structure_from_poscar(ALREADY_PARSED)
    summarize("structure_from_poscar(mapping)", structure)

    # Cartesian input is converted exactly; round-tripping recovers the input positions.
    cartesian_mapping = dict(ALREADY_PARSED, cartesian=True)
    cartesian = structure_from_poscar(cartesian_mapping)
    print("Cartesian coordinates in the same mapping:")
    print("  reduced coords   ", cartesian.sites.reduced_coords.to_floats())
    print("  back to Cartesian", cartesian.cartesian_sites().to_floats())

    # A negative scale line encodes a target volume; the cube-root scale is the one
    # deterministic approximation in the whole path.
    volume_mapping = dict(ALREADY_PARSED, scale=None, volume="512.0")
    by_volume = structure_from_poscar(volume_mapping)
    print("Volume-scaled cell (scale line was negative):")
    print("  requested volume 512.0, got", float(by_volume.cell.volume))

    # VASP-4 has no species line, so the elements are simply absent from the file.
    try:
        structure_from_poscar(dict(ALREADY_PARSED, symbols=None))
    except ValueError as error:
        print("VASP-4 input (no species line):")
        print("  ValueError:", error)


def main() -> None:
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        load_a_plain_poscar(directory)
        print()
        load_a_compressed_contcar(directory)
    print()
    build_from_an_already_parsed_mapping()


if __name__ == "__main__":
    main()
