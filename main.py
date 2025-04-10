#!/usr/bin/env python 
"""\
This runs a training and test round of the model from "NMR shift prediction from small data quantities" by Herman Rull, Markus Fischer, and Stefan Kuhn (forthcoming).

It can be run without any parameters. To change between 13C and 19F prediction, comment lines after #Reading data
"""

import graph
import model
import util
from rdkit import Chem
import random
import torch
import math
from torch_geometric.loader import DataLoader

def main():
    TRAIN_TEST_SPLIT = 0.8

    # Use CUDA by default
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    #Reading data
    supplier3d = Chem.rdmolfiles.SDMolSupplier("nmrshiftdb2withsignals_fl.sd",True, False, True) #Flourine
    #supplier3d = Chem.rdmolfiles.SDMolSupplier("nmrshiftdb2withsignals_c.sd",True, False, True) #Carbon
    #supplier3d = Chem.rdmolfiles.SDMolSupplier("nmrshiftdb2.nmredata.sd",True, False, True) #nmredata
    print(f"In total there are {len(supplier3d)} molecules")

    all_data = list(supplier3d)
    random.Random(80).shuffle(all_data)
    train_data =all_data[:int(TRAIN_TEST_SPLIT * len(supplier3d))]
    test_data =all_data[int(TRAIN_TEST_SPLIT * len(supplier3d)):]

    train_graphs, scaler = graph.scale_graph_data([graph.convert_to_graph(molecule, "nmrshiftdb", nucleus="13C", atom_feature_constructor = graph.atom_features) for idx, molecule in enumerate(train_data) if molecule])
    test_graphs, scaler = graph.scale_graph_data([graph.convert_to_graph(molecule, "nmrshiftdb", nucleus="13C", atom_feature_constructor = graph.atom_features) for idx, molecule in enumerate(test_data) if molecule], scaler=scaler)

    print(f"Converted {len(supplier3d)} molecules to {len(train_graphs) + len(test_graphs)} graphs")
    print(f"Found {sum([sum([1 for shift in graph.y if not math.isnan(shift[0])]) for graph in train_graphs+test_graphs])} individual NMR shifts")

    #Training
    VALIDATION_SPLITS = 4
    EPOCHS = 500
    BATCH_SIZE = 128

    splits = util.chunk_into_n(train_graphs, VALIDATION_SPLITS)
    split_errors=[]
    for idx, split in enumerate(splits):
        split_train_data = []
        for s in splits:
                if s!=split:
                    split_train_data+=s

        train_loader = DataLoader(split_train_data, batch_size = BATCH_SIZE)
        test_loader = DataLoader(split, batch_size = BATCH_SIZE)
        model, optimizer, criterion = util.init_model(6,0.001,0.1)
        loss_list = []
        train_err_list = []
        test_err_list = []

        model.train()
        print(f"Split {idx+1}. Training/test split size:{len(split_train_data)}/{len(split)}")
        for epoch in range(EPOCHS):
            tloss = util.train(model, criterion, optimizer, train_loader)
            train_err = util.evaluate(model, criterion, train_loader)
            test_err = util.evaluate(model, criterion, test_loader)

            loss_list.append(tloss)
            train_err_list.append(train_err)
            test_err_list.append(test_err)

            print('Epoch: {:03d}, Loss: {:.5f}, Train Err: {:.5f}, Test Err: {:.5f}'.format(epoch+1, tloss,
                                                                                            train_err, test_err))
        extra_epochs=0
        #Sometimes the optimizer tries to find other local minima, which means that at E500 the solution is not yet at local minima.
        while extra_epochs<200 and tloss>2.5:
            tloss = util.train(model, criterion, optimizer, train_loader)
            train_err = util.evaluate(model, criterion, train_loader)
            test_err = util.evaluate(model, criterion, test_loader)

            loss_list.append(tloss)
            train_err_list.append(train_err)
            test_err_list.append(test_err)
            extra_epochs+1

        print("\n")
        split_errors.append(test_err)

    print(f"Split errors: {split_errors} with average error {sum(split_errors) / VALIDATION_SPLITS}")

    #Evaluation on test set
    train_loader = DataLoader(train_graphs, batch_size = BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size = BATCH_SIZE)
    model, optimizer, criterion = util.init_model(6,0.001,0.1)
    loss_list = []
    train_err_list = []
    test_err_list = []

    model.train()
    for epoch in range(EPOCHS):
        tloss = util.train(model, criterion, optimizer, train_loader)
        train_err = util.evaluate(model, criterion, train_loader)
        test_err = util.evaluate(model, criterion, test_loader)

        loss_list.append(tloss)
        train_err_list.append(train_err)
        test_err_list.append(test_err)

        print('Epoch: {:03d}, Loss: {:.5f}, Train Err: {:.5f}, Test Err: {:.5f}'.format(epoch+1, tloss,
                                                                                        train_err, test_err))
    extra_epochs=0
    #Sometimes the optimizer tries to find other local minima, which means that at E500 the solution is not yet at local minima.
    while extra_epochs<200 and tloss>2.5:
        tloss = util.train(model, criterion, optimizer, train_loader)
        train_err = util.evaluate(model, criterion, train_loader)
        test_err = util.evaluate(model, criterion, test_loader)

        loss_list.append(tloss)
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        extra_epochs+1

    print("\n")
    util.evaluate(model, criterion, test_loader)

if __name__ == "__main__":
    main()