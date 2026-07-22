# *httk-atomistic*

This site documents specifically the *httk-atomistic* module. For the full
documentation of *httk₂* as a whole, see [docs.httk.org](https://docs.httk.org).

*httk-atomistic* is a *httk₂* module providing crystal structure representations under the namespace `httk.atomistic`.

```{admonition} Quick links
:class: tip

- **API reference**: {doc}`reference/index`
- **Structure guide**: {doc}`structures`
- **Examples notebook**: {doc}`notebooks/examples`
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
from httk.atomistic import Structure, StructurePrimitiveView

structure = Structure(
    cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    species=[
        {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
        {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
    ],
    species_at_sites=["Na", "Cl"],
)

# Present the same structure as an spglib-like (lattice, positions, numbers) tuple.
lattice, positions, numbers = StructurePrimitiveView(structure)
```

```{toctree}
:maxdepth: 2
:caption: Documentation

reference/index
structures
notebooks/examples
```
