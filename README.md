
![MultiChat_Logo](https://github.com/zcaiwei/MultiChat/blob/main/MultiChat_logo.png)

## Overview


MultiChat (Multi-omics Cell-cell communication inference via Heterogeneous graph ATtention network)  is a computational framework designed to infer CCC at single-cell resolution by integrating single-cell or spatial transcriptomics and epigenomics data.

A concise Read the Docs tutorial is available on [this website](https://multichat.readthedocs.io/en/latest/tutorials.html) and is currently under development. The example datasets are available through [Figshare](https://doi.org/10.6084/m9.figshare.30834524) provided in the tutorial.


## Installation


The MultiChat package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework.

Follow these steps in your terminal to get ```MultiChat``` set up in your local enviroment.


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

- The source code is **currently not publicly available** and will be released in a future update.

- Once released, **Option B will be fully functional** for users who prefer installation from source.

- For now, please use **Option A (PyPI installation)**. The PyPI version has been tested on both **Linux and Windows environments**, and installation has been verified to work successfully. If you encounter any issues during installation or usage, please feel free to open an issue or leave a message.

## Input Format and Requirements

MultiChat requires several input files to perform CCC inference. Depending on the analysis scope (e.g., ligand–receptor only vs. multi-layer signaling), some inputs are optional.

### 1. Gene Expression (RNA) File (Mandatory)

A Gene × Cell matrix (CSV or TXT). 

**Note**: Gene IDs must use standard gene symbols.


### 2. Database 

A file (CSV/TSV/TXT) containing known ligand–receptor (L–R) interactions or ligand-receptor-transcription factor-garget gene (L-R-TF-TG).

#### 🔹 Ligand–Receptor Database (Mandatory)
Used when focusing on **significant ligand–receptor interactions** only.

- **Column 1**: `Ligand_Symbol`  
- **Column 2**: `Receptor_Symbol`  
- **Column 3**: `Pathway_Name`  (Optional, e.g., *EGF*) 

#### 🔹  Ligand–Receptor–TF–TG  Database (Optional)

Used when performing **multi-layer signaling inference**.

- **Column 1**: `Ligand_Symbol`  
- **Column 2**: `Receptor_Symbol`  
- **Column 3**: `TF_Symbol`  
- **Column 4**: `TG_Symbol`  
- **Column 5**: `Pathway_Name` (Optional)

We provide a curated L–R database derived from CellChatDB an CellCall. Users may also supply a custom database.

**Note**: Gene symbols must strictly match those in your Gene Expression File. Any additional columns will be ignored.  


### 3. Meta File (**Mandatory**)

A file describing cell annotations:

- **Index**: `cell_id` (Must strictly match the cell names in the Gene Expression File)
- **Column 1**: `cell_type` (or cluster label)  

This file is used to define cell identities and enable cell-type-level analysis.




### 4. Spatial Coordinates File (Optional)

Required only for spatial CCC analysis. A file (CSV/TXT) containing the spatial coordinates for each cell or spot.

- **Index**: `cell_id` (Must strictly match the cell names in the Gene Expression File)
- **Column 1**: `x` coordinate  
- **Column 2**: `y` coordinate  

**Note**:If spatial information is not available, the strategy for selecting positive samples needs to be adjusted accordingly.


### 5. Chromatin Accessibility (ATAC) Data (**Optional**)
A Cell × Peak matrix (CSV or TXT). The data is **optional** and only required when performing **multi-layer signaling inference**.






## Usage


After the installation is complete, you can import and use `MultiChat` in your Python scripts or interactive sessions like this:

```
import MultiChat as MC
```

### 🔹 Mode 1: Ligand–Receptor Interaction Only (No scATAC-seq)

If ATAC data is unavailable, MultiChat can still identify **significant ligand–receptor interactions**.

Please refer to:  [`run_MultiChat-inter.ipynb`](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat-inter.ipynb)


### 🔹 Mode 2: Multi-layer Signaling Inference (No scATAC-seq)

If ATAC data is unavailable but you want to infer **extended signaling cascades** (L-R-TF-TG), MultiChat provides a simplified version.

Please refer to:  [`run_MultiChat-wto-chrom_acc.ipynb`](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat-wto-chrom_acc.ipynb)


### 🔹 Mode 3: Multi-omics Multi-layer Signaling Inference (Recommended)

When both gene expression and chromatin accessibility data are available, MultiChat performs multi-layer CCC inference.

Please refer to:  tutorials (e.g.,[ISSAAC-seq dataset example](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_ISSAAC.ipynb), [HumanHeart dataset example](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Heart.ipynb)  ) 



## Tutorials


To get started with MultiChat, please refer to the `Tutorial` folder for step-by-step instructions.
- [Data Preparation Tutorial: Using single cell ISSAAC-seq multi-omics data of mouse cortex slices with MultiChat](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/data_preprocessing_on_ISSAAC.ipynb)
- [Tutorial 1: Quick Start Guide, running MultiChat on simulation data](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Simusimp.ipynb)
- [Tutorial 2: Running MultiChat on single cell ISSAAC-seq multi-omics data of mouse cortex slices](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_ISSAAC.ipynb)
- [Tutorial 3: Running MultiChat on single cell multi-omics data of human myocardial infarction](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Heart.ipynb)


## Support

If you have any questions, please feel free to contact us at: 📧[zhanglh@whu.edu.cn](mailto:zhanglh@whu.edu.cn). 


