# VASP total density of states

`httk.atomistic.integrations.vasp.io.doscar.read_doscar(path, spin_mode=...)`
reads the six-line DOSCAR header and its declared `NEDOS` total-DOS rows. The
caller states whether the total block is `nonmagnetic` or `collinear`; the
reader requires three columns for nonmagnetic data and five for collinear
data. Energy, DOS, integrated DOS and Fermi energy remain exact `Fraction`
values. Energies must increase strictly. Paths are owned and closed by the
reader; caller-owned text streams stay open.

The result has one `total` channel for nonmagnetic data, or named `up` and
`down` channels for collinear data. The reader stops after the total block and
does not parse any atom-projected sections. It does not infer spin mode from
the column count. Raw DOS samples may be negative, for example from higher-order
smearing, and the reader preserves them. Numerical routines that require a
nonnegative DOS can reject such samples. VASP documents the six-line header,
total-DOS columns, spin column layout and units as states/eV and states in the
[DOSCAR reference](https://vasp.at/wiki/DOSCAR).
