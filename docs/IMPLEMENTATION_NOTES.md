# MTGPLA framework

MTGPLA combines sequence-based and graph-based representations to predict protein-ligand binding affinity. Its feature extraction, interaction modeling, and regression modules form a unified prediction pipeline.

## Input representations

| Modality | Representation |
|---|---|
| Protein | Integer-encoded sequence of length 1,000 |
| Binding pocket | Integer-encoded sequence of length 63 |
| Ligand SMILES | Integer-encoded chemical sequence of length 100 |
| Ligand graph | Molecular connectivity with 108-dimensional atom features |

Sequences are truncated or padded to their configured lengths. Ligand molecular graphs are constructed from MOL2 files using RDKit, with bonds represented in both directions.

## Sequence feature extraction

Protein and pocket sequences are mapped into 128-dimensional embeddings and processed by Transformer-based sequence pathways. The Transformer configuration uses four encoder layers, four attention heads, and a feed-forward dimension of 512.

The ligand SMILES pathway combines self-attention with two one-dimensional convolutional layers. This pathway provides token-level representations for interaction modeling and a pooled 128-dimensional ligand sequence feature.

## Molecular graph feature extraction

A four-layer Graph Isomorphism Network processes the ligand atom features and molecular connectivity. Its hidden representations use 96 channels. Batch normalization and ReLU activations accompany graph feature extraction, and global sum pooling produces the ligand graph representation.

## Pocket-ligand interaction modeling

Cross-attention links ligand sequence features with the binding-pocket representation:

- Queries are derived from ligand sequence features.
- Keys and values are derived from pocket features.
- Normalization and sequence pooling yield a 128-dimensional interaction feature.

This interaction pathway contributes binding-context information alongside the global sequence and molecular graph representations.

## Feature fusion and affinity prediction

| Feature | Dimensions |
|---|---:|
| Protein sequence | 128 |
| Pocket sequence | 128 |
| Ligand SMILES | 128 |
| Ligand molecular graph | 96 |
| Pocket-ligand interaction | 128 |
| Concatenated representation | 608 |

The prediction MLP maps the fused representation through layers of 608 -> 512 -> 256 -> 1. ReLU activations and dropout are applied within the regression module. The final scalar is the predicted binding affinity.

## Training and evaluation

Training uses AdamW and an MSE objective. The default configuration sets seed 42, learning rate 0.001, batch size 96, and 100 epochs. The best checkpoint is selected using validation loss from epoch 60 onward.

Evaluation reports Pearson correlation (R), root mean squared error (RMSE), mean absolute error (MAE), regression standard deviation (SD), and concordance index (CI). The `evaluate.py` entry point exports metrics and per-complex predictions.

## Feature visualization

The visualization tools produce prediction scatter plots and t-SNE projections at three stages:

1. Mean-pooled protein embeddings.
2. Fused representations entering the prediction MLP.
3. Outputs of the second MLP linear layer.

The t-SNE visualization groups complexes using the median experimental affinity of the selected split. Commands and output locations are described in the [project README](../README.md).
