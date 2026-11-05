# MTGPLA data preparation

Prepare PDBbind 2020 / CASF data obtained from the [official provider](https://pdbbind-plus.org.cn/) under its terms. The preprocessing command takes protein sequences, pocket sequences, ligand SMILES, ligand MOL2 structures, and affinity labels as inputs.

## Dataset splits

The ordered complex IDs are provided in `splits/*.txt`.

| Split | Complexes |
|---|---:|
| train | 18,025 |
| val | 1,000 |
| test2016 | 285 |
| test2013 | 195 |

The training and validation sets are disjoint from each other and from both test sets. CASF-2016 and CASF-2013 share 107 complexes. File sizes and SHA-256 hashes for the reference processed datasets are available in [manifest.json](../results/reference/manifest.json).

## Input layout

Prepare per-complex protein and pocket sequences before running `create_data.py`. The input directory has the following layout:

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

The affinity CSV must include a header, with PDB IDs in the first column and numerical affinity values in the second column. Use affinity values on the target scale of the experiment. Each MOL2 filename begins with its four-character PDB ID, followed by `_` or `.`, with one ligand file per selected complex.

## Preprocessing command

```bash
python create_data.py --input-root INPUT_ROOT --affinity-csv AFFINITY_CSV --data-root data
```

Use `--splits test2016 test2013` to process only the evaluation sets. Use `--split-dir` to specify an alternative directory of split IDs. The output directory must be fresh when rebuilding a dataset.

## Feature encoding

- **Protein and pocket:** the alphabet `ABCDEFGHIKLMNOPQRSTUVWXYZ` is mapped to indices 1 through 25, with zero padding. Sequence lengths are 1,000 and 63, respectively.
- **SMILES:** sequences use the `CHAR_SMI_SET` character dictionary and its index-minus-one mapping, with a maximum length of 100 and zero padding.
- **Molecular graph:** RDKit parses ligand MOL2 files into atom descriptors and molecular bonds. Each atom has 108 descriptor values normalized by their row sum; bonds are stored in both directions.
- **Sample order:** preprocessing follows the ordered IDs in each split file and checks the required inputs for each complex.

## Processed datasets

The generated PyG datasets are stored under `DATA_ROOT/processed/`:

```text
train.pt
val.pt
test2016.pt
test2013.pt
```

Use these files with the training and evaluation entry points in the [project README](../README.md). Structures, affinity-label tables, and processed tensors are prepared separately from the code repository.
