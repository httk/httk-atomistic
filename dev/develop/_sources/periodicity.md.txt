# Periodicity

Not everything worth describing is a crystal. A slab is periodic in two
directions, a nanowire in one, a molecule in none. A
{py:class}`~httk.atomistic.Cell` records which of its three directions
actually repeat, so httk can represent all of them and reject operations that
only make sense for a crystal:

```python
from httk.atomistic import Cell

cell = Cell([[3, 0, 0], [0, 3, 0], [0, 0, 1]], periodicity=(True, True, False))
cell.periodicity             # (True, True, False) — a slab
```

The default is `(True, True, True)`: a fully periodic crystal.

The full guide, {doc}`details/periodicity`, covers the frame-not-a-box rule
for aperiodic directions, what changes (wrapping, identity, volume), which
operations are refused, marking a structure as a slab, and serving
`nperiodic_dimensions`/`dimension_types` over OPTIMADE.
