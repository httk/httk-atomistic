# *httk-atomistic*

This site documents specifically the *httk-atomistic* module. For the full
documentation of *httk₂* as a whole, see [docs.httk.org](https://docs.httk.org).

*httk-atomistic* is a *httk₂* module providing crystal structure representations under the namespace `httk.atomistic`. It also carries the file input/output stack for atomistic data — the CIF/mCIF reader and writer, the VASP POSCAR/CONTCAR and output-file readers, the WAVECAR binary reader/writer, and the OPTIMADE trajectory JSON Lines holding format — registering these readers with *httk-core* through `httk.registry.io.atomistic`.

```{admonition} Quick links
:class: tip

- **API reference**: {doc}`reference/index`
- **Structures**: {doc}`structures`
- **Composition and formulas**: {doc}`composition`
- **Material-information levels**: {doc}`prototypes`
- **Asymmetric units**: {doc}`asu`
- **Subgroups and pathfinding**: {doc}`subgroups`
- **Site moments (magnetism)**: {doc}`moments`
- **Integrations (ASE, pymatgen, VASP)**: {doc}`integrations`
- **Data precision**: {doc}`precision`
- **Periodicity (slabs, wires, molecules)**: {doc}`periodicity`
- **Lattice reduction**: {doc}`lattice-reduction`
- **Primitive cells**: {doc}`primitive-cells`
- **Reading and writing CIF files**: {doc}`cif`
- **Reading POSCAR/CONTCAR files**: {doc}`poscar`
- **Reading VASP output files**: {doc}`vasp_outputs`
- **Reading and writing WAVECAR files**: {doc}`wavecar`
- **Plane-wave wavefunctions**: {doc}`wavefunctions`
- **Trajectory JSON Lines**: {doc}`trajectory_jsonl`
- **Runnable examples**: {doc}`examples/index`
- **Examples notebook**: {doc}`notebooks/examples`
- **Disorder walkthrough**: {doc}`notebooks/disorder`

The topic pages above are short and practical; the ones with a full guide link
onward to it in the **Details** section of the sidebar.
````

## Install

Preferably work in a Python virtual environment, then do:
```bash
git clone https://github.com/httk/httk-atomistic
cd httk-atomistic
python -m pip install -e .
```

## Usage example

```python
from httk.atomistic import UnitcellStructure, PlainStructureView

structure = UnitcellStructure(
    cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    species=[
        {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
        {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
    ],
    species_at_sites=["Na", "Cl"],
)

# Present the same structure as an spglib-like (lattice, positions, numbers) tuple.
lattice, positions, numbers = PlainStructureView(structure)
```

```{toctree}
:maxdepth: 2
:caption: Documentation

reference/index
structures
composition
prototypes
cif
poscar
vasp_outputs
wavecar
wavefunctions
trajectory_jsonl
asu
subgroups
moments
integrations
precision
periodicity
lattice-reduction
primitive-cells
examples/index
notebooks/examples
notebooks/disorder
```

```{toctree}
:maxdepth: 1
:caption: Details

details/structures
details/asu
details/precision
details/periodicity
```
