# Experimental implementation and manuscript alignment

The six original source scripts in the inspected project matched the public repository after newline normalization. The selected saved run reproduces the numerical main-result table after rounding. Checkpoint tensor shapes agree with the released architecture. These facts establish the provenance of the checkpoint and code dimensions; they do not prove every historical training or preprocessing detail.

The model source `model.py` is preserved, including the following behaviors:

| Item | Experimental implementation | Difference from manuscript wording |
|---|---|---|
| Protein/pocket encoder | Both forward paths call `protein_transformer`; embeddings are also shared. The separately registered pocket encoder layer is unused. | The manuscript describes two independent, unshared encoders. |
| Positional lookup | `pos_emb` is indexed by residue token IDs. | This is not the usual positional encoding indexed by sequence positions. |
| Regression MLP | 608 -> 512 -> 256 -> 1, confirmed by checkpoint weights. | The manuscript describes 608 -> 256 -> 128 -> 1. |
| GIN epsilon | `GINConv` uses its default `train_eps=False`. | The manuscript describes a learnable epsilon. |
| SMILES interaction length | Two unpadded kernel-3 convolutions reduce length 100 to 96. Cross-attention receives these 96 tokens. | The interaction matrix is described as retaining 100 tokens. |
| Graph feature order | Ring indicator precedes aromaticity in the code; atom vectors are normalized by their row sum. | Follow the code when constructing checkpoint-compatible inputs. |
| t-SNE Stage A | Mean-pooled protein embeddings only, 128 dimensions. | It is not a combined multimodal representation. |

Do not change these model behaviors while claiming to use this trained checkpoint. Correcting the architecture to match the prose would define a different model and require retraining and fresh evaluation. The manuscript should be reconciled with the actual experiment separately.

## Release engineering changes

- Added configurable training, evaluation and plotting entry points; removed cluster-specific absolute paths and font requirements.
- Retained model parameters, forward computation, default training hyperparameters, and metric definitions.
- Updated trusted PyG dataset loading for PyTorch versions with restricted-loading defaults; state-dictionary checkpoints use restricted loading where supported.
- Changed the CI denominator check from Python identity (`is not 0`) to numeric comparison (`!= 0`).
- Replaced the inspected alternate preprocessing script's incompatible start token 27 and `data27` destination with the no-start-token 1000/63 input format found in the saved `data` tensors. The original alternate script cannot be used directly with this 26-row embedding checkpoint.
- Preserved Stage A's actual protein-only definition and removed unused intermediate t-SNE variables. The second MLP stage records the second linear layer's output before ReLU, as in the original script.

The original source is available in repository history before this release. No ablation models were fabricated from the result table. The original raw-structure-to-sequence extraction pipeline and historical dependency lockfile remain unavailable.
