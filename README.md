# MTGPLA

**A Hybrid Multimodal Deep Learning Framework Integrating Interaction Features for Accurate Protein-Ligand Affinity Prediction**

MTGPLA predicts protein-ligand binding affinity by integrating protein sequence information, local binding-pocket context, ligand chemical sequences, molecular topology, and pocket-ligand interaction features.

## Framework

MTGPLA processes four input modalities:

- **Protein sequence:** Transformer-based encoding captures global protein sequence features.
- **Pocket sequence:** Sequence encoding represents the local binding environment.
- **Ligand SMILES:** A self-attention and CNN pathway extracts chemical sequence features.
- **Ligand molecular graph:** A four-layer Graph Isomorphism Network (GIN) encodes molecular topology using 108-dimensional atom descriptors.

A cross-attention module uses ligand sequence features as queries and pocket features as keys and values. The interaction representation is concatenated with the four modality representations and passed to a multilayer perceptron for affinity regression.

See [framework details](docs/IMPLEMENTATION_NOTES.md) for input dimensions, feature fusion, and prediction modules. The model class in the source code is named `DeepTTG`.

## Installation

Use Python 3.11 in a separate environment. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CPU or CUDA platform, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The environment used to evaluate the released checkpoint is recorded in [verification_environment.json](results/reference/verification_environment.json). GPU training supports automatic mixed precision through PyTorch.

## Data preparation

MTGPLA uses PDBbind 2020 for training and validation, with CASF-2016 and CASF-2013 as evaluation sets. Obtain the data from the [official provider](https://pdbbind-plus.org.cn/) under its terms.

The supplied complex IDs define the following splits:

| Split | Complexes |
|---|---:|
| Training | 18,025 |
| Validation | 1,000 |
| CASF-2016 | 285 |
| CASF-2013 | 195 |

Prepare protein/pocket sequence CSVs, ligand SMILES CSVs, affinity labels, and ligand MOL2 files following the [input format](data/README.md), then run:

```bash
python create_data.py --input-root /path/to/pdbbind_2016_2013 --affinity-csv /path/to/affinity_data.csv --data-root data
```

The preprocessing command takes per-complex sequences and ligand representations as inputs. It checks that the complexes listed in each split have the required inputs. Processed datasets are stored as:

```text
data/processed/train.pt
data/processed/val.pt
data/processed/test2016.pt
data/processed/test2013.pt
```

PDBbind structures, affinity-label tables, and processed feature tensors are obtained or prepared separately.

## Evaluation

Evaluate the released checkpoint on the two core test sets:

```bash
python evaluate.py --checkpoint checkpoints/mtgpla_best.pt --data-root data --device auto --batch-size 16
```

Use `--device cpu` for CPU execution. Aggregate metrics and per-complex predictions are written to `results/evaluation/`. Use trusted processed PyG datasets.

The released checkpoint was selected at epoch 95 using validation loss. Its reference performance is:

| Dataset | R | RMSE | MAE | SD | CI |
|---|---:|---:|---:|---:|---:|
| CASF-2016 | 0.821 | 1.258 | 0.994 | 1.243 | 0.810 |
| CASF-2013 | 0.801 | 1.353 | 1.091 | 1.347 | 0.798 |

Full-precision metrics, dataset/checkpoint hashes, and evaluation records are available in [results/reference](results/reference).

## Training

```bash
python main.py --data-root data --device cuda --seed 42 --batch-size 96 --lr 0.001 --epochs 100 --save-best-from 60
```

The default configuration uses AdamW, sum-reduced squared-error loss, seed 42, batch size 96, and 100 epochs. The checkpoint is selected by minimum validation loss from epoch 60 onward. Use `--output-dir` to choose a run folder and adjust `--batch-size` to fit available GPU memory.

## Visualization

Generate affinity scatter plots from the evaluation outputs:

```bash
python plt_scatter.py --predictions-dir results/evaluation
```

Visualize intermediate model representations using t-SNE:

```bash
python plt_tSNE.py --data-root data --checkpoint checkpoints/mtgpla_best.pt --split test2016
```

The t-SNE tool visualizes protein embeddings, fused features before the prediction MLP, and the output of the second MLP linear layer. Figures are saved in `results/plots/`. For training and validation scatter plots, first generate the corresponding predictions with `evaluate.py --splits train val test2016 test2013`.

## Repository structure

| File or directory | Purpose |
|---|---|
| `model.py` | MTGPLA model architecture |
| `dataset.py`, `create_data.py` | Dataset loading and feature preprocessing |
| `main.py` | Model training |
| `evaluate.py`, `runtime.py`, `metrics.py` | Prediction and performance evaluation |
| `plt_scatter.py`, `plt_tSNE.py` | Result and feature visualization |
| `checkpoints/mtgpla_best.pt` | Released model checkpoint |
| `data/splits/` | Ordered train, validation, and test IDs |
| `results/reference/` | Reference metrics, hashes, and evaluation records |
| `docs/IMPLEMENTATION_NOTES.md` | Framework details |

## Citation

If you use MTGPLA in your research, please cite the MTGPLA manuscript. Author information is provided in [CITATION.cff](CITATION.cff).
