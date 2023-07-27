import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

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


pro = []
for dt_name in ['metz1','metz2','metz3']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        pro += list(df['target_sequence'])
pro = set(pro)


seq_voc = "ACDEFGHIKLMNPQRSTVWXY"
L=len(seq_voc)


pro_dic={}
for k in pro:
    pro_3d=np.zeros((L,L,L))
    arr_3=[]
    kk=len(k)
    max=1
    for i in range(kk-2):
        # p_i=k[i,i+2]  str，int类型不匹配

        a=k[i]
        b=k[i+1]
        c=k[i+2]
        p_i=a+b+c
        arr_3.append(p_i)

    for a in range(L):
        for b in range(L):
            for c in range(L):
                aa = seq_voc[a]
                bb = seq_voc[b]
                cc = seq_voc[c]
                dd = aa + bb + cc
                ff = aa + cc + bb
                qq = bb + aa + cc
                uu = bb + cc + aa
                pp = cc + aa + bb
                mm = cc + bb + aa

                count = 0
                for ee in arr_3:
                    if ee == dd or ee == ff or ee == qq or ee == uu or ee == pp or ee == mm:
                        count += 1
                if(count>max):
                    max=count
                pro_3d[a, b, c] = count
    pro_dic[k] = pro_3d/max


compound_iso_smiles = []
for dt_name in ['metz1','metz2','metz3']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)



seq_sml = "BCHNOSPFIMbclnospr0123456789()[]=.+-#"
LS = len(seq_sml)

smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


zx = 1
dpro_dic = {}
for k in pro :
    kk = len(k)
    arg11 = []
    arg22 = np.zeros((L,L))
    oop = len(pro)
    print('Converting data222 to format111: {}/{}'.format(zx, oop))
    zx +=1
    for i in range(kk) :
        for j in range(kk) :
            a = k[i]
            b = k[j]
            c = a+b
            arg11.append(c)

    for x in range(L) :
        for y in range(L) :
            aa = seq_voc[x]
            bb = seq_voc[y]
            cc = aa + bb
            count = 0
            for ee in arg11 :
                if ( cc == ee ) :
                    count +=1
            arg22[x,y] = count
    dpro_dic[k] = arg22


datasets = ['metz1','metz2','metz3']

for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):

        df = pd.read_csv('data/' + dataset + '_train.csv')

        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])

        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)

        df = pd.read_csv('data/' + dataset + '_test.csv')

        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])

        test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph, pro_dic=pro_dic ,dpro_dic=dpro_dic)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,y=test_Y, smile_graph=smile_graph, pro_dic=pro_dic ,dpro_dic=dpro_dic)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')


