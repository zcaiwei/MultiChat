
![RegChat_Overview](https://github.com/zcaiwei/MultiChat/blob/main/MultiChat_logo.png)

## Overview

MultiChat (Multi-omics Cell-cell communication inference via Heterogeneous graph ATtention network)  is a computational framework designed to infer CCC at single-cell resolution by integrating single-cell or spatial transcriptomics and epigenomics data.


## Installation
The MultiChat package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework.

First clone the repository. 

```
git clone https://github.com/zcaiwei/MultiChat.git
cd MultiChat-main
```

It's recommended to create a separate conda environment for running MultiChat:

```
#create an environment called env_MultiChat
conda create -n env_MultiChat python=3.12

#activate your environment
conda activate env_MultiChat
```

Install all the required packages. The torch-geometric library is required, please see the installation steps in https://github.com/pyg-team/pytorch_geometric#installation
```
conda install pyg
conda install conda-forge::pytorch_scatter
conda install conda-forge::pytorch_cluster
conda install conda-forge::pytorch_sparse
```

Install the requirements packages

```
pip install -r requirements.txt
```

To install MultiChat, use the following pip command:

```
python setup.py build
python setup.py install
```



## Tutorials

To get started with MultiChat, please refer to the `Tutorial` folder for step-by-step instructions.
- [Data Preparation Tutorial: Using single cell ISSAAC-seq multi-omics data of mouse cortex slices with MultiChat](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/data_preprocessing_on_ISSAAC.ipynb)
- [Tutorial 1: Quick Start Guide, running MultiChat on simulation data](https://github.com/lhzhanglabtools/RegChat/blob/main/Tutorial/run_RegChat_on_simulation_data.ipynb)
- [Tutorial 2: Running MultiChat on single cell ISSAAC-seq multi-omics data of mouse cortex slices](https://github.com/zcaiwei/MultiChat/blob/main/Tutorial/run_MultiChat_on_ISSAAC.ipynb)


## Support

If you have any questions, please feel free to contact us [zhanglh@whu.edu.cn](mailto:zhanglh@whu.edu.cn). 


