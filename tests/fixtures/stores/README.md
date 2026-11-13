# Store fixtures

`a2db444_domain_networks.{sqlite,duckdb}.gz` and `a2db444_domain_networks.mongo.json.gz` (a
`bson.json_util` dump of each collection's documents and indexes) are protocol-3 stores written by
*httk-atomistic* `a2db444` (with *httk-core* `6cbb302` and *httk-store* `d4ed802`), before unit-cell
bond networks and bond sources were stored. `tests/test_bond_network_records.py` reopens each with
`upgrade=True` and checks that the upgrade is additive and that every row keeps its content id and
network.

`build_a2db444_store.py` built them from `git archive a2db444` (see its docstring), with
`entry_ids=EntryIdScheme("httk.test", "1")` and the unit-cell, fundamental-domain and
asymmetric-unit records for `StructureEntry`, saving:

- `distance_bonds(load("cod/1000008.cif"))` (diopside) and `distance_bonds(load("cod/1509688.cif"))`,
  asymmetric units with networks;
- a `FundamentalDomainStructure` of the second with its network and supplied shapes (the baseline
  symbols of all its shaped sites but the first);
- diopside without its network, as an asymmetric unit and as its unit cell.

Both COD entries are public domain (see `../cod/README.md`).
