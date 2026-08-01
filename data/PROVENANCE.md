# Sample molecule provenance

- **Approved source URL:** https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=1000&offset=0
- **Successful retrieval URL:** https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=1000&offset=0
- **Retrieval date:** 2026-07-31 UTC (HTTP response date: Fri, 31 Jul 2026 21:35:24 GMT)
- **Raw response SHA-256:** `8bd9308a97851d57f31e497102fcacc0a2a9b971b5e1ff4b932a8f40c3322252`

The response contained 1,000 molecule records. In the returned order, this sample retains the first 256 records with a nonempty `molecule_chembl_id` and a nonempty `molecule_structures.canonical_smiles` value that RDKit parses. Each retained structure was canonicalized with `Chem.MolToSmiles(molecule, canonical=True)` and written with only its molecule ID and canonical SMILES.

Documentation:

- [ChEMBL Data Web Services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
- [ChEMBL downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
- [ChEMBL data licensing information](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing)

This small sample is for demonstration only. Before production use or redistribution, recheck the current ChEMBL terms and licensing information at the official links above.
