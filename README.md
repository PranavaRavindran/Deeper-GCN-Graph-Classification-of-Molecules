# DeeperGCN Graph Classification

A PyTorch Geometric implementation of a graph classification model using Deeper GCN architectures, tested on the MUTAG molecular graph dataset.

## 🔍 Overview

This project applies a **Deeper Graph Convolutional Network (GCN)** to perform binary classification on graphs from the MUTAG dataset. It features manual data cleaning, normalization, edge dropout for regularization, and early stopping to prevent overfitting.

## 🧠 Model Architecture

- Initial `GCNConv` layer for feature transformation
- Multiple `DeepGCNLayer` blocks for deeper learning capacity
- Final `GCNConv` and global mean pooling
- Fully connected layer for graph-level classification

> Model depth, dropout, and hidden layer sizes are configurable

## 📦 Dependencies

Install required packages:

pip install torch
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install matplotlib networkx pandas

## Dataset: 

- MUTAG dataset from the [TUDataset collection](https://chrsmrrs.github.io/datasets/)

- Graphs represent molecules with labeled mutagenicity

## Preprocessing

  - Converts raw PyG dataset into a pandas.DataFrame for cleaning
  
  - Drops null rows (if any)
  
  - Transforms cleaned DataFrame back to PyG Data format
  
  - Applies NormalizeFeatures and manual edge dropout to training data

## How to Run

  Open the notebook and execute all cells:

   - jupyter notebook DeeperGCN_GraphClassification.ipynb

  Steps performed:

   - Dataset loading, visualization, and cleaning
    
   - Model instantiation with:
    
       - 6 total layers
    
       - 128 hidden channels
    
   - Custom data loader with edge dropout
    
   - Training with early stopping
    
   - Test accuracy evaluation
    
   - Model saved to deepergcn_mutag.pth

## Results

  - Best test accuracy is printed at the end of training (varies per run)

  - Default: 10 epochs, early stop if no improvement for 5 epochs

## Saving/Loading

  ### Save model:
  torch.save(model.state_dict(), "deepergcn_mutag.pth")

  ### Load model:
  model.load_state_dict(torch.load("deepergcn_mutag.pth"))

## License

  - MIT License. See LICENSE file for details.

## Acknowledgements

  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

  - [TUDataset](https://chrsmrrs.github.io/datasets/)

  - MUTAG dataset by Debnath et al.
