# Integrations

*httk-atomistic* bridges to a few widely-used third-party formats and objects
through `httk.atomistic.integrations`. Each bridge is optional: it needs its own
package importable and is exercised on its own CI lane, so nothing here is pulled
in by a bare install.

## ASE and pymatgen

`ASEAtomsView` and `PymatgenStructureView` present any `StructureLike` as an
`ase.Atoms` or a pymatgen `Structure`; `unview` hands back the plain third-party
object:

```python
from httk.atomistic import UnitcellStructure
from httk.atomistic.integrations.ase import ASEAtomsView
from httk.atomistic.integrations.pymatgen import PymatgenStructureView
from httk.core import unview

structure = UnitcellStructure(
    cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    species=[
        {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
        {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
    ],
    species_at_sites=["Na", "Cl"],
)

atoms = unview(ASEAtomsView(structure))          # an ase.Atoms
pmg = unview(PymatgenStructureView(structure))   # a pymatgen Structure
```

The pymatgen bridge carries species decoration across the boundary: per-species
`charges` and `spins` map to pymatgen oxidation states / spins. A `label` maps
only on a dummy species; a label on a real element is rejected (pymatgen cannot
attach a label to an element), as are explicit masses and attached species.

## VASP

`VASPStructure` loads a POSCAR/CONTCAR lazily and round-trips it byte-for-byte,
so a structure read from VASP output and written back reproduces the original
file exactly. `VASPTrajectory` reads OUTCAR and/or XDATCAR data lazily as a
trajectory:

```console
>>> from httk.atomistic.integrations.vasp import VASPStructure, VASPTrajectory
>>> structure = VASPStructure("POSCAR")          # lazy, byte-exact round-trip
>>> trajectory = VASPTrajectory("OUTCAR")         # frames read on demand
```

These build on the low-level, string-preserving VASP readers: see {doc}`poscar`
for the neutral POSCAR/CONTCAR mapping, {doc}`vasp_outputs` for the OUTCAR,
XDATCAR, OSZICAR and POTCAR readers, and {doc}`wavecar` for the binary WAVECAR
layer.
