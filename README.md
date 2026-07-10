
<p align="center">
  <img src="https://raw.githubusercontent.com/zcaiwei/MultiChat/main/MultiChat_logo.png" alt="MultiChat Logo" width="160">
</p>

## Overview


MultiChat (Multi-omics Cell-cell communication inference via Heterogeneous graph ATtention network) is a computational framework designed to infer spatially resolved regulatory cell–cell communication at cellular resolution by integrating single-cell or spatial transcriptomics and chromatin accessibility data.

A detailed tutorial is available at [this website](https://multichat.readthedocs.io/en/latest/tutorials.html), and the example datasets are available at [Figshare](https://doi.org/10.6084/m9.figshare.30834524).


## Installation


The MultiChat package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework.

Follow the following steps in your terminal to install ```MultiChat``` in your local enviroment.


### 1. Set up Environment

Create a clean Conda environment to avoid conflicts:

```bash
# Create an environment called env_MultiChat with Python 3.12
conda create -n env_MultiChat python=3.12

# Activate the environment
conda activate env_MultiChat
```

### 2. Install Dependencies (PyTorch & PyG)
Since PyG relies on system-specific libraries, it is recommended to install it via Conda before installing MultiChat:

```
conda install pyg
conda install conda-forge::pytorch_scatter
conda install conda-forge::pytorch_cluster
conda install conda-forge::pytorch_sparse
```
**Note**: For specific CUDA versions or other installation methods for PyTorch/PyG, please refer to the [official PyG installation guide](https://github.com/pyg-team/pytorch_geometric#installation).


### 3. Install MultiChat


**Option A: Install from PyPI (Recommended)** 

The easiest way to install the stable version of MultiChat is via pip:

```
pip install scMultiChat
```


**Option B: Install from Source**

```
git clone https://github.com/zcaiwei/MultiChat.git
cd MultiChat-main

# Install requirements
pip install -r requirements.txt

# Build and install 
python setup.py build 
python setup.py install
```
**Note**

- The PyPI version has been tested on both **Linux and Windows environments**, and installation has been verified to work successfully. If you encounter any issues during installation or usage, please feel free to open an issue or leave a message.

## Input Format and Requirements

MultiChat takes either single-cell multi-omics data along with spatial transcriptomics data or spatial multi-omics data as inputs.


### 1. Gene Expression (RNA) File 

A Gene × Cell matrix (CSV or TXT). 

**Note**: Gene IDs must use standard gene symbols.


### 2. Chromatin Accessibility (ATAC) Data (**Optional**)

A Cell × Peak matrix (CSV or TXT). The data is **optional** and only required when performing **multi-layer signaling inference**.


### 3. Spatial Coordinates File (Optional)

A file (CSV/TXT) containing the spatial coordinates for each cell or spot.

- **Index**: `cell_id` (Must strictly match the cell names in the Gene Expression File)
- **Column 1**: `x` coordinate  
- **Column 2**: `y` coordinate  


### 4. Meta data File 

A file describing cell annotations:

- **Index**: `cell_id` (Must strictly match the cell names in the Gene Expression File)
- **Column 1**: `cell_type` (or cluster label)  

This file is used to define cell identities and enable cell-type-level analysis.





### 5. Database (**Optional**)

MultiChat provides a built-in curated database. Users may also supply a custom database by providing a CSV, TSV or TXT file containing ligand–receptor (L–R) or ligand–receptor–transcription factor–target gene (L–R–TF–TG) information. The file should contain the following columns:

#### 🔹 Ligand–Receptor Database 
Used when focusing on **significant ligand–receptor interactions** only.

- **Column 1**: `Ligand_Symbol`  
- **Column 2**: `Receptor_Symbol`  
- **Column 3**: `Pathway_Name`  (Optional, e.g., *EGF*) 

#### 🔹  Ligand–Receptor–TF–TG  Database 

Used when performing **multi-layer signaling inference**.

- **Column 1**: `Ligand_Symbol`  
- **Column 2**: `Receptor_Symbol`  
- **Column 3**: `TF_Symbol`  
- **Column 4**: `TG_Symbol`  
- **Column 5**: `Pathway_Name` (Optional)




## Usage


After the installation is complete, you can import and use `MultiChat` in your Python scripts or interactive sessions like this:

```
import MultiChat as MC
```

### 🔹 Mode 1: Multi-omics Multi-layer Signaling Inference with ATAC-seq data

Please refer to:  tutorials (e.g.,[ISSAAC-seq dataset example](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_ISSAAC.ipynb), [HumanHeart dataset example](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Heart.ipynb)  ) 


### 🔹 Mode 2: Multi-omics Multi-layer Signaling Inference without ATAC-seq data

If ATAC data is unavailable but you want to infer **multi-layer signaling** (L-R-TF-TG), MultiChat provides a simplified version.

Please refer to:  [`run_MultiChat-wto-chrom_acc.ipynb`](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat-wto-chrom_acc.ipynb)


### 🔹 Mode 3: Ligand-Receptor Interaction Inference without ATAC-seq data

If ATAC-seq data are unavailable and you only want to infer significant **intercellular ligand-receptor (L-R)** interactions, MultiChat provides a simplified workflow.

Please refer to:  [`run_MultiChat_for_ligand-receptor_identification.ipynb`](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_for_ligand-receptor_identification.ipynb)




## Tutorials


To get started with MultiChat, please refer to the `Tutorial` folder for step-by-step instructions.
- [Data Preparation Tutorial: Using single cell ISSAAC-seq multi-omics data of mouse cortex slices with MultiChat](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/data_preprocessing_on_ISSAAC.ipynb)
- [Tutorial 1: Quick Start Guide – Running MultiChat on Simulated Data](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Simulation.ipynb)
- [Tutorial 2: Running MultiChat on paired single-cell multi-omics data along with spatial transcriptomics data](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_ISSAAC.ipynb)
- [Tutorial 3:  Running MultiChat on unpaired single-cell multi-omics data along with spatial transcriptomics data](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Heart.ipynb)
- [Tutorial 4:  Running MultiChat on spatial multi-omics data](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_P22.ipynb)


## Support

If you have any questions, please feel free to contact us at: 📧[zhanglh@whu.edu.cn](mailto:zhanglh@whu.edu.cn). 
