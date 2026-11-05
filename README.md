# MTGPLA

MTGPLA is a multimodal model for protein-ligand binding affinity prediction. It combines protein and pocket sequences, ligand SMILES, a ligand molecular graph, and pocket-ligand cross-attention.

This release contains the experimental model code, the checkpoint corresponding to the manuscript's main results, exact split IDs, preprocessing and evaluation tools, and reference results. The internal model class is named `DeepTTG` for compatibility with the saved checkpoint.

**Implementation details and known manuscript discrepancies are documented in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).** The trained architecture is preserved. PDBbind structures, affinity labels and processed feature tensors are not bundled. See [data preparation](data/README.md) for the required inputs and the remaining raw-data preprocessing limitation.

## Installation

Use Python 3.11 in a separate environment. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CPU or CUDA platform, then:

```bash
python -m pip install -r requirements.txt
```

The manuscript reports PyTorch 1.9.0 and PyG 2.0.1 for the original experiments. The present command-line tools are validated separately in the environment recorded under `results/reference/verification_environment.json`; this is not an export of the historical cluster environment. The current dependency file targets the modern PyG loader. CUDA toolkits and NVIDIA Apex source archives do not belong in this repository; mixed precision uses `torch.cuda.amp`.

## Data

Obtain the appropriate PDBbind 2020 / CASF data from the [official provider](https://pdbbind-plus.org.cn/) under its terms. Preserve the supplied split IDs and preprocessing conventions. Place checkpoint-compatible, trusted processed files here:

```text
data/processed/train.pt
data/processed/val.pt
data/processed/test2016.pt
data/processed/test2013.pt
```

Alternatively, prepare the sequence/SMILES CSV and MOL2 layout described in [data/README.md](data/README.md), then run:

```bash
python create_data.py --input-root /path/to/pdbbind_2016_2013 --affinity-csv /path/to/affinity_data.csv --data-root data
```

The preprocessing entry point stops on missing/invalid complexes rather than silently changing a split. The original PDB-to-sequence and pocket-extraction procedure has not been recovered, so downloading raw PDBbind alone is not yet a complete end-to-end reconstruction of the original inputs.

## Evaluate the released checkpoint

```bash
python evaluate.py --checkpoint checkpoints/mtgpla_best.pt --data-root data --device auto --batch-size 16
```

Use `--device cpu` for CPU execution. Outputs are aggregate metrics and per-complex predictions under `results/evaluation/`. Only load processed PyG pickle files from a trusted source. State-dictionary checkpoints use restricted loading where PyTorch supports it.

The original run `DeepTTG_2026-03-23 20:44:04.526124_42` selected epoch 95 by validation loss. Its archived results are:

| Split | n | R | RMSE | MAE | SD | CI |
|---|---:|---:|---:|---:|---:|---:|
| CASF-2016 | 285 | 0.820581 | 1.257964 | 0.994258 | 1.242701 | 0.809702 |
| CASF-2013 | 195 | 0.801265 | 1.353451 | 1.090783 | 1.346737 | 0.797700 |

Original full-precision results, release verification metrics and SHA-256 hashes are in `results/reference/`. The full 285/195-complex checkpoint evaluation in the recorded modern environment agrees with the manuscript after rounding to three decimals; small unrounded differences remain. These values round to the manuscript's main results. Training is stochastic; a newly trained model need not reproduce them bit for bit.

## Train

```bash
python main.py --data-root data --device cuda --seed 42 --batch-size 96 --lr 0.001 --epochs 100 --save-best-from 60
```

Default training settings preserve the original entry point: AdamW, summed MSE, seed 42, batch size 96, 100 epochs, and checkpoint selection by minimum validation loss from epoch 60 onward. Evaluation sets are logged during training but are not used for checkpoint selection. Set a smaller batch size for limited GPU memory; doing so changes the training configuration. Use `--output-dir` to choose a run folder.

## Visualizations

```bash
python plt_scatter.py --predictions-dir results/evaluation
python plt_tSNE.py --data-root data --checkpoint checkpoints/mtgpla_best.pt --split test2016
```

For training/validation scatter plots, first include those phases in `evaluate.py --splits train val test2016 test2013`. Figures are written under `results/plots/`. Stage A in the released t-SNE tool is **protein embeddings only**, matching the original plotting code; it is not a combined four-modality embedding. t-SNE layouts can vary across scikit-learn versions.

## Repository contents

- `model.py`: checkpoint-compatible model architecture, unchanged from the experimental source.
- `dataset.py`, `create_data.py`: processed-data loading and explicit input preprocessing.
- `main.py`, `evaluate.py`, `runtime.py`, `metrics.py`: training, prediction and metrics.
- `plt_scatter.py`, `plt_tSNE.py`: configurable visualization entry points.
- `checkpoints/mtgpla_best.pt`: the main experimental checkpoint (about 8.3 MiB).
- `data/splits/`: ordered train/validation/test complex IDs.
- `results/reference/`: original results, hashes, and release verification records.

The manuscript's ablation implementations were not found in the inspected project source. They are not reconstructed or presented here as verified experimental code.
