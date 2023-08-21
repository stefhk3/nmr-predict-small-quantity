import graph
import model
from rdkit import Chem
import random
import torch
import math

def main():
    TRAIN_TEST_SPLIT = 0.8

    # Use CUDA by default
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    supplier3d = Chem.rdmolfiles.SDMolSupplier("nmrshiftdb2withsignals_fl.sd",True, False, True) #Flourine
    #supplier3d = Chem.rdmolfiles.SDMolSupplier("download") #Carbon
    print(f"In total there are {len(supplier3d)} molecules")

    all_data = list(supplier3d)
    random.Random(80).shuffle(all_data)
    train_data =all_data[:int(TRAIN_TEST_SPLIT * len(supplier3d))]
    test_data =all_data[int(TRAIN_TEST_SPLIT * len(supplier3d)):]

    #TODO: Use training data scaler on test data.
    train_graphs, scaler = graph.scale_graph_data([graph.convert_to_graph(molecule, atom_feature_constructor = graph.atom_features) for idx, molecule in enumerate(train_data) if molecule])
    test_graphs, scaler = graph.scale_graph_data([graph.convert_to_graph(molecule, atom_feature_constructor = graph.atom_features) for idx, molecule in enumerate(test_data) if molecule], scaler=scaler)

    #all_data = [convert_to_graph(molecule) for idx, molecule in enumerate(supplier3d) if molecule and idx not in[24,25,30,31,32]]
    print(f"Converted {len(supplier3d)} molecules to {len(train_graphs) + len(test_graphs)} graphs")
    print(f"Found {sum([sum([1 for shift in graph.y if not math.isnan(shift[0])]) for graph in train_graphs+test_graphs])} individual NMR shifts")

if __name__ == "__main__":
    main()