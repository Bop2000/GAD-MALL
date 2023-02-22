# GAD-MALL
Generative Architected-Materials Design using Mutiobjective Active Learning Loop (GAD-MALL): A active learning pipeline for designing the architected materials

## Abstract

Architected materials that consist of multiple subelements arranged in particular orders can demonstrate a much broader range of properties than their constituent materials. However, the rational design of these materials generally relies on experts’ prior knowledge and requires painstaking effort. Here, we present a data-efficient method for the high-dimensional multi-property optimization of 3D-printed architected materials utilizing a machine learning (ML) cycle consisting of the finite element method (FEM) and 3D neural networks. Specifically, we applied our method to orthopedic implant design. Compared to expert designs, our experience-free method designed microscale heterogeneous architectures with a biocompatible elastic modulus and higher strength. Furthermore, inspired by the knowledge learned from the neural networks, we developed machine-human synergy, adapting the ML-designed architecture to fix a macroscale, irregularly shaped animal bone defect. Such adaptation exhibits 20% higher experimental load-bearing capacity than the expert design. Thus, our method opens a new paradigm for the fast and intelligent design of architected materials with tailored mechanical, physical, and chemical properties.

## Requirements

The codes are tested on Ubuntu 20.04, CUDA 11.1 with the following packages:

```shell
torch == 1.9.0
torch-geometric == 1.7.2
tensorboad == 2.5.0
scipy == 1.7.0
numpy == 1.20.2
```

**Note:** The graph classification tasks can be executed without torch-geometric (PyG). 

## Installation

You can install the required dependencies with the following code.

```shell
conda create -n ESGNN python=3.8
conda activate ESGNN
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge --yes
pip install tensorboard=2.5.0
CUDA=cu111
TORCH=1.9.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
pip install torch-geometric==1.7.2 
```

## Demo

To search for high-performance architected materials with specific elastic modulus and high yield strength using GAD-MALL, please run the following line in terminal:

```shell
bash run_GAD_MALL.sh
```

For more about `Finite Element Method`, see `README` in folder `Finite Element Method by ABAQUS`

For more about `Model Parameters Optimizing using Bayesian Optimization`, see `README` in folder `Model Parameter Optimize_Bayesian Optimization`

For more about `Other State-of-the-art Active Learning Methods` compared with our best method GAD-MALL, see `README` in folder `Other-State-of-the-art-Active-Learning`

For more about `TPMS-Gyroid Structure Generation`, see `README` in folder `TPMS-Gyroid-Generate-by-MATLAB`

## Dataset

The raw data `3D_CAE_train.npy` for training 3D-CAE can be downloaded [here](https://drive.google.com/file/d/1BfmD4bsPS2hG5zm7XGLHc8lpUN_WqhgV/view?usp=share_link).

The raw data for training 3D-CNN is in the folder `GAD-MALL/GAD-MALL/data/`. One big file named `Matrix60.npy` can be downloaded [here](https://drive.google.com/file/d/1VRH4X_mACxM82HoaplwV0ThaDiN3iPXm/view?usp=share_link).

## CItation

If you find this work interesting, welcome to cite our paper!

```

```
