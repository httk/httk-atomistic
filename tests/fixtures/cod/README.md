# COD fixtures

Unmodified entries from the Crystallography Open Database (COD), copied byte-for-byte with
their headers intact. Each file carries COD's public-domain dedication ("All data on this site
have been placed in the public domain by the contributors."), so redistribution here is permitted.
Do not edit or trim them; the tests depend on the exact original content.

| COD ID | Used for |
| --- | --- |
| 1000008 | diopside: no `_atom_site_type_symbol` column (symbol inferred from labels); its SiO3 chains along c, a monoclinic net chosen cell-independently |
| 1008801 | deuterium with disorder (isotope kept in the label) |
| 4002923 | occupancy that loads without repair |
| 4002853 | occupancy needing `repair=True` |
| 4002910 | occupancy needing `repair=True` |
| 3000362 | W-doped VO₂ declared P1, an author deposition that the depositor released into the public domain (`_cod_depositor_comments`): four mixed V/W sites of one species class, whose class structure is recognized as P2₁/c (labels, distance bonds, protonet label and a broken bond set) |
| 1509425 | K2AgSbS4 (Pnn2): bonding-net (protonet) canonicalisation |
| 1539945 | CuCl2 (C2/m): chains along b, a monoclinic net chosen cell-independently |
| 1535669 | AlCl3 (C2/m): layers containing b, a monoclinic net chosen cell-independently |
| 1509688 | AgO (P2₁/c): a 3-periodic Ag(I)/Ag(III) oxide net whose E-only (form 2) label depended on the cell |
| 7217521 | tricyclic tetrazine C8H16N4O2 (P2₁/c): molecule on an inversion centre (`test_protonet_molecular.py`) |
| 7217843 | fluorophenyl naphthalimide C20H14FNO3 (P2₁/c): a larger molecule on a general position |
| 7218683 | Cu(C9H9BrNO)2 (P2₁/c): Cu on 2b, which the canonical protonetstructure moves to 2a |
| 7219855 | C12H10N4 with adipic acid (P2₁/c): two molecules on two inversion centres |
| 4134228 | C9H17N3O (P2₁/c): a flexible molecule, a protonet search performance regression |
| 7219077 | C18H18N2S2 (C2/c): molecule on a 2-fold axis, two of its atoms on the axis |
| 4326344 | 2-[(N,N-dimethylamino)methyl]aniline (P2₁): a protonet search performance regression |
| 4131686 | C20H16N2O2 (P-1, Z = 1): molecule on an inversion centre |
| 7217003 | triazolo-benzothiadiazine dioxide C16H14N4O2S (P-1): a protonet search performance regression |
| 7219102 | sarcosine C3H7NO2 (P2₁2₁2₁): a small chiral-group molecular crystal |
| 4120336 | ferracarborane C8H22B14Fe (P2₁/c): `distance_bonds` drops its six Fe–B bonds across the cage C (a known limitation, `test_bonding.py`) |
