# Data preparation

This repository publishes split IDs, model code and a trained checkpoint. It does not redistribute PDBbind structures, affinity-label tables or processed feature tensors. Obtain data and any necessary redistribution permission from the [official provider](https://pdbbind-plus.org.cn/); do not assume that a download entitlement grants redistribution rights. The current [provider terms](https://www.pdbbind-plus.org.cn/termofuse) should be checked for your intended use.

## Verified saved splits

The IDs in `splits/*.txt` were extracted in their saved order from the project's processed PyG datasets.

| Split | Complexes |
|---|---:|
| train | 18,025 |
| val | 1,000 |
| test2016 | 285 |
| test2013 | 195 |

Train/validation/test ID intersections are zero. The two test sets share 107 IDs. These checks concern exact PDB IDs, not sequence or scaffold similarity. File sizes and SHA-256 hashes of the original processed tensors are recorded in `../results/reference/manifest.json` for local identity checks.

## Inputs accepted by create_data.py

Provide the existing per-complex sequences and SMILES, affinity labels, and corresponding ligand MOL2 files:

```text
INPUT_ROOT/
  sequence/
    train_protein.csv     # columns: id,seq
    train_pocket.csv      # columns: id,seq
    val_protein.csv
    val_pocket.csv
    test2016_protein.csv
    test2016_pocket.csv
    test2013_protein.csv
    test2013_pocket.csv
  smiles/
    train_smi.csv         # columns: pdbid,smiles
    val_smi.csv
    test2016_smi.csv
    test2013_smi.csv
  pdbbind2020/
    train/mol2/XXXX_ligand.mol2
    val/mol2/XXXX_ligand.mol2
    test2016/mol2/XXXX_ligand.mol2
    test2013/mol2/XXXX_ligand.mol2
```

`--affinity-csv` supplies a CSV with a header whose first two columns are PDB ID and numerical affinity. It must use the same target scale as the original experiment. A MOL2 filename begins with its four-character PDB ID, followed by `_` or `.`, and there must be one ligand file per selected complex.

```bash
python create_data.py --input-root INPUT_ROOT --affinity-csv AFFINITY_CSV --data-root data
```

Use `--splits test2016 test2013` to prepare only the evaluation splits. Existing outputs are not overwritten; select a fresh `--data-root` to rebuild them.

## Encoding conventions

- Protein/pocket alphabet: `ABCDEFGHIKLMNOPQRSTUVWXYZ`, mapped to indices 1 through 25; zero pads the tail.
- Protein length 1,000; pocket length 63; no extra start/CLS token. Saved tensors were checked for these shapes and valid index ranges.
- SMILES length 100, with the original character dictionary and index-minus-one encoding. This intentionally retains the historical collision between padding zero and the `(` character. Changing it would require retraining.
- Ligand graph: RDKit MOL2 parsing; 108 original atom descriptors, normalized by their row sum; chemical bonds represented in both directions.
- The released preprocessor uses the supplied split ID order and fails on parsing/missing-data errors rather than dropping samples silently.

## Remaining provenance limitation

The inspected source accepts pre-extracted protein and pocket sequences; it does not contain the original PDB parsing, chain selection or pocket-residue extraction pipeline. Those choices cannot be uniquely reconstructed from the truncated integer tensors. A newly extracted sequence dataset is not guaranteed to reproduce the original tensors, even when it uses the same complex IDs. Obtain the original sequence/SMILES inputs or document and validate the extraction procedure before claiming an exact reconstruction from raw structures.
