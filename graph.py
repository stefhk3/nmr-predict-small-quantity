import numpy as np
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
import mendeleev
import math
from torch_geometric.data import Batch,Data
import torch
from torch import tensor

el_map={}


def atom_features_default():

  feature_getters = {}

  feature_getters["ohe atomic number"] = lambda atom:atomic_number_ohe.transform(np.array([[atom.GetAtomicNum()]]))[0] # Atomic number
  feature_getters["hyb ohe"] = lambda atom: hybridization_ohe.transform(np.array([[atom.GetHybridization().name]]))[0]
  feature_getters["valence ohe"] = lambda atom: valence_ohe.transform(np.array([[atom.GetTotalValence()]]))[0]
  feature_getters["hybridization"] = lambda atom: atom.GetHybridization()
  feature_getters["atomic radius"]= lambda atom: getMendeleevElement(atom.GetAtomicNum()).atomic_radius or 0 # Atomic radius
  feature_getters["atomic volume"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).atomic_volume # Atomic volume
  feature_getters["atomic weight"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).atomic_weight # Atomic weight
  feature_getters["dipole polarizability"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).dipole_polarizability # Dipole polarizability
  feature_getters["electron affinity"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).electron_affinity # Electron affinity
  feature_getters["electronegativity"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).en_pauling # Electronegativity
  feature_getters["electrons"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).electrons # No. of electrons
  feature_getters["neutrons"] = lambda atom: getMendeleevElement(atom.GetAtomicNum()).neutrons # No. of neutrons
  feature_getters["formal charge ohe"] = lambda atom: fc_ohe.transform(np.array([[atom.GetFormalCharge()]]))[0]
  #feature_getters["gaisteigerCharge"] = lambda atom: 0 if np.isfinite(float(atom.GetProp('_GasteigerCharge'))) else float(atom.GetProp('_GasteigerCharge'))  #partial charges
  feature_getters["chiral tag"] = lambda atom: atom.GetChiralTag()

  return feature_getters



def getNaiveBondLength(bond):
    a = getMendeleevElement(bond.GetEndAtom().GetAtomicNum()).atomic_radius or 0
    b =  getMendeleevElement(bond.GetBeginAtom().GetAtomicNum()).atomic_radius or 0

    return a/200.0 + b/200.0

def bond_feature_smart_distance_and_rdkit_type(bond):
  onehot_encoded_bondtype = onehot_encoder.transform(np.array([[bond.GetBondType()]]))[0]
  [x1, y1, z1] = list(bond.GetOwningMol().GetConformer().GetAtomPosition(bond.GetBeginAtomIdx()))
  [x2, y2, z2] = list(bond.GetOwningMol().GetConformer().GetAtomPosition(bond.GetEndAtomIdx()))
  ex_dist = getNaiveBondLength(bond)
  distance = [(math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)) - ex_dist] # Distance
  return distance+ list(onehot_encoded_bondtype)

def getMendeleevElement(nr):
  if nr not in el_map:
   el_map[nr] = mendeleev.element(nr)
  return el_map[nr]

def nmr_shift(atom):
  for key, value in atom.GetOwningMol().GetPropsAsDict().items():
    if key.startswith("Spectrum"):
      for shift in value.split('|'):
        x = shift.split(';')
        if (len(x) == 3 and x[2] == f"{atom.GetIdx()}"):
          return float(x[0])
  return float("NaN") # We use NaN for atoms we don't want to predict shifts

def bond_features_distance_only(bond):
  #onehot_encoded_bondtype = onehot_encoder.transform(np.array([[bond.GetBondType()]]))[0]
  [x1, y1, z1] = list(bond.GetOwningMol().GetConformer().GetAtomPosition(bond.GetBeginAtomIdx()))
  [x2, y2, z2] = list(bond.GetOwningMol().GetConformer().GetAtomPosition(bond.GetEndAtomIdx()))
  distance = [(math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))] # Distance
  return distance#+list(onehot_encoded_bondtype)


def flatten(l):
  ret=[]
  for el in l:
    if isinstance(el, list) or isinstance(el, np.ndarray):
      ret.extend(el)
    else:
      ret.append(el)
  return ret

def turn_to_graph (molecule, atom_feature_getters= atom_features_default().values(), bond_features=bond_features_distance_only):
  node_features = [flatten([getter(atom) for  getter in atom_feature_getters ]) for atom in molecule.GetAtoms() ]
  node_targets = [nmr_shift(atom) for atom in molecule.GetAtoms()]
  edge_features = [bond_features(bond) for bond in molecule.GetBonds()]
  edge_index = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()]
  # Bonds are not directed, so lets add the missing pair to make the graph undirected
  edge_index.extend([reversed(bond) for bond in edge_index])
  edge_features.extend(edge_features)
  # Some node_features had null values in carbon data and then the long graph compilation process was stopped.
  if any(None in sublist for sublist in node_features):
      return None

  return Data(
      x=tensor(node_features, dtype=torch.float),
      edge_index=tensor(edge_index, dtype=torch.long).t().contiguous(),
      edge_attr=tensor(edge_features, dtype=torch.float),
      y=tensor([[t] for t in node_targets], dtype=torch.float)
  )

# One hot encoding

## Bonds
bond_idxes = np.array([Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC])
bond_idxes = bond_idxes.reshape(len(bond_idxes), 1)
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
onehot_encoder.fit(bond_idxes)

## Hybridization
hybridization_idxes = np.array(list(Chem.HybridizationType.names))
hybridization_idxes = hybridization_idxes.reshape(len(hybridization_idxes), 1)
hybridization_ohe = OneHotEncoder(sparse=False)
hybridization_ohe.fit(hybridization_idxes)

## Valence
valences = np.arange(1, 8);
valences = valences.reshape(len(valences), 1)
valence_ohe = OneHotEncoder(handle_unknown="ignore",sparse=False)
valence_ohe.fit(valences)


## Formal Charge
fc = np.arange(-1, 1);
fc = fc.reshape(len(fc), 1)
fc_ohe = OneHotEncoder(handle_unknown="ignore",sparse=False)
fc_ohe.fit(fc)


## Atomic number
atomic_nums = np.array([6,1,7,8,9,17,15,11, 16])
atomic_nums = atomic_nums.reshape(len(atomic_nums), 1)
atomic_number_ohe = OneHotEncoder(handle_unknown="ignore",sparse=False)
atomic_number_ohe.fit(atomic_nums)
atomic_number_ohe.transform(np.array([[1]]))


#TODO
#solvent_ohe = np.array([1,0,0,0,0])



el_map={}
def getMendeleevElement(nr):
  if nr not in el_map:
   el_map[nr] = mendeleev.element(nr)
  return el_map[nr]

def nmr_shift(atom):
  for key, value in atom.GetOwningMol().GetPropsAsDict().items():
    if key.startswith("Spectrum"):
      for shift in value.split('|'):
        x = shift.split(';')
        if (len(x) == 3 and x[2] == f"{atom.GetIdx()}"):
          return float(x[0])
  return float("NaN") # We use NaN for atoms we don't want to predict shifts

def bond_features(bond):
  onehot_encoded_bondtype = onehot_encoder.transform(np.array([[bond.GetBondType()]]))[0]
  [x1, y1, z1] = list(bond.GetOwningMol().GetConformer().GetAtomPosition(bond.GetBeginAtomIdx()))
  [x2, y2, z2] = list(bond.GetOwningMol().GetConformer().GetAtomPosition(bond.GetEndAtomIdx()))
  distance = [(math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))] # Distance
  return distance+list(onehot_encoded_bondtype)

def atom_features(atom,  molecule=None):
  me = getMendeleevElement(atom.GetAtomicNum())

  #[x, y, z] = list(atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()))
  features = []
  #TBD: Do we need to encode molecule atom itself? Or is atomic number sufficient? One-hot encode?
  features.extend(atomic_number_ohe.transform(np.array([[atom.GetAtomicNum()]]))[0]) # Atomic number
  #features.append(atom.GetIsAromatic()) 
  features.extend(np.array([1,0,0,0,0]))
  features.extend(hybridization_ohe.transform(np.array([[atom.GetHybridization().name]]))[0])
  #features.extend(valence_ohe.transform(np.array([[atom.GetTotalValence()]]))[0])
  features.extend(fc_ohe.transform(np.array([[atom.GetFormalCharge()]]))[0]) 
  features.append(me.atomic_radius or 0) # Atomic radius
  features.append(me.atomic_volume) # Atomic volume
  features.append(me.atomic_weight) # Atomic weight
  features.append(me.covalent_radius) # Covalent radius
  features.append(me.vdw_radius) # Van der Waals radius
  features.append(me.dipole_polarizability) # Dipole polarizability
  features.append(me.electron_affinity) # Electron affinity
  features.append(me.electrophilicity()) # Electrophilicity index
  features.append(me.en_pauling) # Electronegativity
  features.append(me.electrons) # No. of electrons
  features.append(me.neutrons) # No. of neutrons
  #features.append(x) # X coordinate - TBD: Not sure this is a meaningful feature (but they had in the paper)
  #features.append(y) # Y coordinate - TBD: Not sure this is a meaningful feature (but they had in the paper)
  #features.append(z) # Z coordinate - TBD: Not sure this is a meaningful feature (but they had in the paper)
  #features.append(0 if np.isfinite(float(atom.GetProp('_GasteigerCharge'))) else float(atom.GetProp('_GasteigerCharge')))  #partial charges
  features.append(atom.GetChiralTag())
  features.append(atom.IsInRing())
  return features


def convert_to_graph(molecule, atom_feature_constructor=atom_features):
  #Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)
  node_features = [atom_feature_constructor(atom, molecule) for atom in molecule.GetAtoms()]
  node_targets = [nmr_shift(atom) for atom in molecule.GetAtoms()]
  edge_features = [bond_features(bond) for bond in molecule.GetBonds()]
  edge_index = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in molecule.GetBonds()]
  # Bonds are not directed, so lets add the missing pair to make the graph undirected
  edge_index.extend([reversed(bond) for bond in edge_index])
  edge_features.extend(edge_features)
  
  # Some node_features had null values in carbon data and then the long graph compilation process was stopped.
  if any(None in sublist for sublist in node_features):
      return None

  return Data(
      x=tensor(node_features, dtype=torch.float),
      edge_index=tensor(edge_index, dtype=torch.long).t().contiguous(),
      edge_attr=tensor(edge_features, dtype=torch.float),
      y=tensor([[t] for t in node_targets], dtype=torch.float)
  )

def scale_graph_data(latent_graph_list, scaler=None):


  if scaler:
    node_mean, node_std, edge_mean, edge_std = scaler
    print(f"Using existing scaler: {scaler}")
  else:
      #Iterate through graph list to get stacked NODE and EDGE features

    node_stack=[]
    edge_stack=[]
    for g in latent_graph_list:
      
        node_stack.append(g.x)          #Append node features
        edge_stack.append(g.edge_attr)  #Append edge features

    node_cat=torch.cat(node_stack,dim=0)
    edge_cat=torch.cat(edge_stack,dim=0)

    node_mean=node_cat.mean(dim=0)
    node_std=node_cat.std(dim=0,unbiased=False)
    edge_mean=edge_cat.mean(dim=0)
    edge_std=edge_cat.std(dim=0,unbiased=False)

  #Apply zero-mean, unit variance scaling, append scaled graph to list
  latent_graph_list_sc=[]
  for g in latent_graph_list:
      x_sc=g.x-node_mean
      x_sc/=node_std
      ea_sc=g.edge_attr-edge_mean
      ea_sc/=edge_std
      ea_sc=torch.nan_to_num(ea_sc, posinf=1.0)
      x_sc=torch.nan_to_num(x_sc, posinf=1.0)
      temp_graph=Data(x=x_sc,edge_index=g.edge_index,edge_attr=ea_sc, y=g.y)
      latent_graph_list_sc.append(temp_graph)

  scaler= (node_mean,node_std,edge_mean,edge_std)
  return latent_graph_list_sc,scaler
