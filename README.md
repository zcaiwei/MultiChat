
![MultiChat_Logo](https://github.com/zcaiwei/MultiChat/blob/main/MultiChat_logo.png)

## Overview

MultiChat (Multi-omics Cell-cell communication inference via Heterogeneous graph ATtention network)  is a computational framework designed to infer CCC at single-cell resolution by integrating single-cell or spatial transcriptomics and epigenomics data.


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

## Usage
After the installation is complete, you can import and use `MultiChat` in your Python scripts or interactive sessions like this:

```
import MultiChat as MC
```



## Tutorials

To get started with MultiChat, please refer to the `Tutorial` folder for step-by-step instructions.
- [Data Preparation Tutorial: Using single cell ISSAAC-seq multi-omics data of mouse cortex slices with MultiChat](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/data_preprocessing_on_ISSAAC.ipynb)
- [Tutorial 1: Quick Start Guide, running MultiChat on simulation data](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Simusimp.ipynb)
- [Tutorial 2: Running MultiChat on single cell ISSAAC-seq multi-omics data of mouse cortex slices](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_ISSAAC.ipynb)
- [Tutorial 3: Running MultiChat on single cell multi-omics data of human myocardial infarction](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_Heart.ipynb)


## Support

If you have any questions, please feel free to contact us at: 📧[zhanglh@whu.edu.cn](mailto:zhanglh@whu.edu.cn). 


