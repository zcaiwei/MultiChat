Welcome to MultiChat's documentation!
=====================================

MultiChat: Multi-omics Cell-cell communication inference via Heterogeneous graph ATtention network
==================================================================================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation
   Data preparation on ISSAAC <tutorials/data_preprocessing_on_ISSAAC>
   Tutorial 1: Quick start on simulation data <tutorials/run_MultiChat_on_Simusimp>
   Tutorial 2: MultiChat on ISSAAC-seq data <tutorials/run_MultiChat_on_ISSAAC>
   Tutorial 3: MultiChat on human heart data <tutorials/run_MultiChat_on_Heart>
   Optional mode: ligand-receptor interactions only <tutorials/run_MultiChat-inter>
   Optional mode: multi-layer signaling without chromatin accessibility <tutorials/run_MultiChat-wto-chrom_acc>

.. image:: _static/MultiChat_logo.png
   :width: 600
   :alt: MultiChat logo

Overview
========

MultiChat is a computational framework designed to infer cell-cell communication at single-cell resolution by integrating single-cell or spatial transcriptomics and epigenomics data.

The tutorials demonstrate how to prepare input data, run MultiChat on simulation and real multi-omics datasets, infer ligand-receptor communication, infer extended ligand-receptor-transcription factor-target gene signaling paths, and visualize communication strength and information flow.

Citation
========

If you use MultiChat in your work, please cite the associated publication when it becomes available.

Support
=======

For questions, please contact zhanglh@whu.edu.cn.
