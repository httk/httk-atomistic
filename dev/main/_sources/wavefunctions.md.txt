# Plane-wave wavefunctions

`httk-atomistic` exposes VASP plane-wave coefficients as
`PlaneWaveFunctions`. The class is numpy-backed, uses zero-based spin,
k-point, and band indices, and can be built from the neutral `vasp-wavecar`
payload supplied by *httk-io*.

Install the optional dependency before using this functionality:

```bash
python -m pip install -e '.[numpy]'
```

## Loading a WAVECAR

Importing `httk.core` discovers the *httk-io* reader. `load` selects the
reader by the exact `WAVECAR` basename or by the `.wavecar` extension, then
the atomistic registration turns the payload into `PlaneWaveFunctions`:

```python
from httk.core import load

wave = load("WAVECAR")
assert wave.nspins >= 1
coefficients = wave.coefficients(0, 0, 0)
```

Use `with load("WAVECAR") as wave:` to close a file-backed source automatically; cached coefficients and metadata remain available after exit.

The WAVECAR does not store whether gamma compression used the `x` or `z` half-space. The default is `x`; load z-half gamma files with `load("WAVECAR", gamma_half="z")`.

The main metadata properties are `nspins`, `nkpts`, `nbands`, `encut`,
`cell`, `kpoints`, `eigenvalues`, `occupations`, `nplanewaves`,
`double_precision`, `is_gamma`, `gamma_half`, and `kgrid_size`.
Coefficient vectors are loaded lazily from a file-backed source and cached by
default. Use `coefficients(..., cache=False)` to avoid populating the cache;
an existing cached value is still reused.

## G-vectors and real-space waves

`gvectors(kpt=0)` returns the reciprocal-grid integer vectors selected by the
cell, k-point, and energy cutoff. Pass `gamma=True` or `gamma=False` to ask
for a particular representation, and use `gamma_half="x"` or `"z"` for the
gamma half-grid. The default representation follows the source.

`realspace_wave(spin, kpt, band)` returns a three-dimensional complex numpy
array from the selected coefficient vector. It is normalized by default;
pass `norm=False` to retain the unnormalized inverse transform.

## Selecting and converting coefficients

`select` can restrict any combination of spins, k-points, and bands. The
indices in `spins`, `kpts`, and `bands` are sequences of distinct zero-based
indices. `format=None` retains the current representation; `format="std"`
requests the standard full grid and `format="gamma"` requests a gamma-point
half-grid:

```python
subset = wave.select(spins=[0], kpts=[0], bands=[0, 1], format="std")
gamma_wave = subset.select(format="gamma", gamma_half="x")
```

Gamma compression is only valid for one gamma-point k-point. Conversion from
gamma to standard reconstructs the conjugate half. Conversion from standard
to gamma reduces through a real-space representation and therefore destroys
the original coefficient phase. That conversion is lossy and must be chosen
explicitly with `format="gamma"`; it is not an exact round trip.

## Saving and comparing waves

Save a `PlaneWaveFunctions` object through the core writer registry:

```python
from httk.core import save

save(wave, "WAVECAR.copy.wavecar")
```

`wavefunction_overlap(phi1, phi2)` returns the complex conjugate inner
product of two same-shaped arrays. `save_vesta` writes the real and imaginary
parts of a three-dimensional complex real-space wave as
`<basename>_r.vasp` and `<basename>_i.vasp`, using the supplied structure's
POSCAR representation:

```python
from httk.atomistic import save_vesta

save_vesta("wave", structure, wave.realspace_wave(0, 0, 0))
```

`save_vesta` also requires the *httk-io* package for its POSCAR and
volumetric writers.

## Deliberate scope

Non-collinear and spinor WAVECAR files are not supported. The old v1
rearrangement behavior is not part of this API; use explicit `select` calls
and explicit standard/gamma conversion instead.
