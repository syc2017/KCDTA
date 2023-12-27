import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

all_prots = []
datasets = ['davis','kiba']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold1 = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    # fifth fold
    train_fold = [ee for e in train_fold1[1:5] for ee in e]
    test_fold = [ee for e in train_fold1[0:1] for ee in e]
    com_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    # ligands1=ligands1.update(ligands)

    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    # proteins1 = proteins1.update(proteins)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(ligands[d])
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test','com']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[test_fold], cols[test_fold]
        elif opt == 'com':
            rows, cols = rows[com_fold], cols[com_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(test_fold))
    print('com_fold:', len(com_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))


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

# Put all the protein sequences into the array pro
pro = []
for dt_name in ['davis','kiba']:
    opts = ['train','test','com']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        pro += list(df['target_sequence'])
pro = set(pro)


seq_voc = "ACDEFGHIKLMNPQRSTVWXY"
L=len(seq_voc)

# The protein sequences were mapped into three - dimensional matrix using K-mers method
pro_dic={}
for k in pro:
    pro_3d=np.zeros((L,L,L))
    arr_3=[]
    kk=len(k)
    max=1
    for i in range(kk-2):
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

# Place all the small molecule sequences into the array compound_iso_smiles
compound_iso_smiles = []
for dt_name in ['davis','kiba']:
    opts = ['train','test','com']
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

# The protein sequences are mapped to two - dimensional matrices by Cartesian product method
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


datasets = ['davis','kiba']

for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    processed_data_file_com = 'data/processed/' + dataset + '_com.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)) or (not os.path.isfile(processed_data_file_com))):

        df = pd.read_csv('data/' + dataset + '_train.csv')

        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])

        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)

        df = pd.read_csv('data/' + dataset + '_test.csv')

        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])

        test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        df = pd.read_csv('data/' + dataset + '_com.csv')

        com_drugs, com_prots, com_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])

        com_drugs, com_prots, com_Y = np.asarray(com_drugs), np.asarray(com_prots), np.asarray(com_Y)

        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph, pro_dic=pro_dic ,dpro_dic=dpro_dic)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,y=test_Y, smile_graph=smile_graph, pro_dic=pro_dic ,dpro_dic=dpro_dic)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')

        com_data = TestbedDataset(root='data', dataset=dataset + '_com', xd=com_drugs, xt=com_prots, y=com_Y,smile_graph=smile_graph, pro_dic=pro_dic, dpro_dic=dpro_dic)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')


