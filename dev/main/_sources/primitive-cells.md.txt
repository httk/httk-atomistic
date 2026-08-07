# Primitive cells

`primitive_cell` constructs a reproducible primitive cell from the IT standard-setting
conventional cell. It uses the fixed centring-dependent transformation convention adopted by
[spglib](https://spglib.readthedocs.io/en/latest/definition.html#transformation-to-the-primitive-cell),
so the result is tied to the standardized crystallographic basis rather than to a cell-shape
reduction algorithm.

## The transformation convention

With lattice vectors represented as columns, spglib defines

$$
B_p = B_s P_c,
$$

where $B_s$ is the standard-setting conventional basis and $B_p$ is the primitive basis. The
column-vector matrices are

$$
P_A =\begin{pmatrix}
1&0&0\\
0&\frac12&-\frac12\\
0&\frac12&\frac12
\end{pmatrix},\quad
P_C =\begin{pmatrix}
\frac12&\frac12&0\\
-\frac12&\frac12&0\\
0&0&1
\end{pmatrix},
$$

$$
P_R =\begin{pmatrix}
\frac23&-\frac13&-\frac13\\
\frac13&\frac13&-\frac23\\
\frac13&\frac13&\frac13
\end{pmatrix},\quad
P_I =\begin{pmatrix}
-\frac12&\frac12&\frac12\\
\frac12&-\frac12&\frac12\\
\frac12&\frac12&-\frac12
\end{pmatrix},
$$

$$
P_F =\begin{pmatrix}
0&\frac12&\frac12\\
\frac12&0&\frac12\\
\frac12&\frac12&0
\end{pmatrix},\qquad P_P=I_3.
$$

httk stores cell vectors as rows. It therefore applies the row-form matrix
$T_c=P_c^T$:

$$
\mathop{\rm basis}_{\rm prim}=T_c\mathop{\rm basis}_{\rm conv},\qquad
f_p=f_sT_c^{-1},
$$

with fractional coordinates normalized into $[0,1)$. The matrix has determinant $1/n$, where
$n$ is the number of centring translations: 1 for P, 2 for A, C, and I, 3 for R, and 4 for F.

## Relation to `conventional_cell`

The operation first calls {py:func}`~httk.atomistic.conventional_cell`, including its optional
recognition step. `primitive_cell` then applies the fixed table above to that exact conventional
result. It does not reduce the primitive basis by Niggli or any other cell-shape algorithm. A
primitive cell is consequently reproducible from the standard setting, while a Niggli cell is a
separate canonical lattice reduction.

All matrix and coordinate arithmetic remains exact after recognition: rational fractional
coordinates stay rational, and Cartesian basis operations retain httk's exact surd arithmetic.
Cell and coordinate precision metadata is widened by the corresponding exact matrix norms.

## Example

```pycon
>>> from httk.atomistic import ASUStructure, WyckoffSite, primitive_cell
>>> from httk.core import FracVector
>>> carbon = [{"name": "C", "chemical_symbols": ["C"], "concentration": [1.0]}]
>>> asu = ASUStructure(
...     [[5, 0, 0], [0, 5, 0], [0, 0, 5]], 229,
...     [WyckoffSite("a", FracVector.create(()), "C")], carbon,
... )
>>> result = primitive_cell(asu)
>>> result.multiplier
Fraction(1, 2)
>>> len(result.structure.sites)
1
```

To obtain a Niggli-reduced cell after this operation, use
{py:func}`~httk.atomistic.niggli_reduced` as a separate step; see
{doc}`lattice-reduction`.
