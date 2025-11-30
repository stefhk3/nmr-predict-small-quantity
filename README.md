This repository contains the code for the paper "NMR shift prediction from small data quantities" by Herman Rull, Markus Fischer, and Stefan Kuhn ([[https://doi.org/10.1186/s13321-023-00785-x]]).

# Prerequisites

Check out this repository using a git client, or download the repository using the green code button on the github page. The new directory nmr-predict-small-quantity is the directory from where all commands should be run.

We recommend to use anaconda. In anaconda, an environment with the required packages installed, can be created using this command:

    conda create --name <env> --file requirements.txt

After that, doing

    conda activate <env>

activates the environment.

If you want to create your own setup, the following packages are needed. They should include everything via dependencies, but this might depend on your setup: rdkit, mendeleev, pytorch, pytorch_geometric, pytorch_scatter, pytorch_sparse, scikit-learn

# Running

You can run the program with

    python3 main.py

On Linux systems, ./main.py should also work.

This will load a dataset, split it into training and test data, train a model using the training data, and test it using the test data. Training progress will be printed to console and the test results will be printed at the end.

By default, it will use the 19F data. If you want to do 13C, change the line

    supplier3d = Chem.rdmolfiles.SDMolSupplier("nmrshiftdb2withsignals_fl.sd",True, False, True

to

    supplier3d = Chem.rdmolfiles.SDMolSupplier("nmrshiftdb2withsignals_c.sd",True, False, True)

or change the comment symbol # between the two lines.

