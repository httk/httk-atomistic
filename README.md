# httk-atomistic

*httk-atomistic* is a [*httk₂*](https://github.com/httk/httk2) module providing crystal structure representations under the namespace `httk.atomistic`.

It also provides the atomistic file-I/O layer: reading and writing CIF/mCIF, VASP POSCAR/CONTCAR and outputs (OUTCAR, XDATCAR, OSZICAR, POTCAR, WAVECAR), and trajectory JSONL. The readers and writers register automatically on `import httk.core`, so `httk.core.load` and `httk.core.save` pick them up.
