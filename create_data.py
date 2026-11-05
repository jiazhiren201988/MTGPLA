import pandas as pd
import itertools
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from dataset import TestbedDataset
from tqdm import tqdm
from pathlib import Path
import argparse
CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,"~":65}
def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1
    return X

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(str(atom.GetChiralTag()),['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER','misc'])+
                    one_of_k_encoding_unk(atom.GetFormalCharge(),[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'])+
                    one_of_k_encoding_unk(atom.GetNumRadicalElectrons(),[0, 1, 2, 3, 4, 'misc'])+
                    one_of_k_encoding_unk(str(atom.GetHybridization()),['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'])+
                    [atom.IsInRing()]+
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromMol2File(smile)
    # if mol is None:
    #     mol = Chem.MolFromMol2File(smile, sanitize=False)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {letter: index + 1 for index, letter in enumerate(SEQ_VOCAB)}


def encode_sequence(sequence, length):
    result = np.zeros(length, dtype=np.int64)
    for index, letter in enumerate(sequence[:length]):
        result[index] = SEQ_DICT[letter]
    return result


def main():
    parser = argparse.ArgumentParser(description="Build checkpoint-compatible PyG datasets from sequence/SMILES CSVs and MOL2 files.")
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--affinity-csv', required=True, type=Path)
    parser.add_argument('--data-root', default=Path('data'), type=Path)
    parser.add_argument('--split-dir', default=Path('data/splits'), type=Path)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test2016', 'test2013'])
    args = parser.parse_args()
    affinity_frame = pd.read_csv(args.affinity_csv, dtype=str)
    affinities = dict(zip(affinity_frame.iloc[:, 0].astype(str), affinity_frame.iloc[:, 1].astype(float)))
    for phase in args.splits:
        destination = args.data_root / 'processed' / (phase + '.pt')
        if destination.exists():
            raise FileExistsError(f'{destination} already exists; choose a fresh --data-root.')
        ids = (args.split_dir / (phase + '.txt')).read_text().splitlines()
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {phase}')
        protein_frame = pd.read_csv(args.input_root / 'sequence' / (phase + '_protein.csv'), dtype=str)
        pocket_frame = pd.read_csv(args.input_root / 'sequence' / (phase + '_pocket.csv'), dtype=str)
        smiles_frame = pd.read_csv(args.input_root / 'smiles' / (phase + '_smi.csv'), dtype=str)
        proteins = dict(zip(protein_frame['id'].astype(str), protein_frame['seq']))
        pockets = dict(zip(pocket_frame['id'].astype(str), pocket_frame['seq']))
        smiles = dict(zip(smiles_frame['pdbid'].astype(str), smiles_frame['smiles']))
        mol2_root = args.input_root / 'pdbbind2020' / phase / 'mol2'
        mol2_files = {}
        for path in sorted(mol2_root.glob('*.mol2')):
            identifier = path.name.split('_')[0].split('.')[0]
            if identifier in mol2_files:
                raise ValueError(f'Multiple MOL2 files for {identifier}')
            mol2_files[identifier] = path
        graphs, protein_vectors, pocket_vectors, smiles_vectors = {}, {}, {}, {}
        for identifier in tqdm(ids, desc=phase):
            try:
                graph = smile_to_graph(str(mol2_files[identifier]))
                protein = encode_sequence(proteins[identifier], 1000)
                pocket = encode_sequence(pockets[identifier], 63)
                ligand = label_smiles(smiles[identifier], 100)
                _ = affinities[identifier]
                if len(graph[1][0]) != 108 or not graph[2]:
                    raise ValueError('Expected 108 atom features and at least one bond')
            except Exception as error:
                raise ValueError(f'{phase}/{identifier}: {error}') from error
            graphs[identifier] = graph
            protein_vectors[identifier], pocket_vectors[identifier], smiles_vectors[identifier] = protein, pocket, ligand
        TestbedDataset(root=str(args.data_root), dataset=phase, pro=protein_vectors,
                       poc=pocket_vectors, smi=smiles_vectors, y=affinities, smile_graph=graphs)


if __name__ == '__main__':
    main()
