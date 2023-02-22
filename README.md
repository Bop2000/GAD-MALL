# GAD-MALL
Generative Architected-Materials Design using Mutiobjective Active Learning Loop (GAD-MALL): A active learning pipeline for designing the architected materials

## Abstract

Architected materials that consist of multiple subelements arranged in particular orders can demonstrate a much broader range of properties than their constituent materials. However, the rational design of these materials generally relies on experts’ prior knowledge and requires painstaking effort. Here, we present a data-efficient method for the high-dimensional multi-property optimization of 3D-printed architected materials utilizing a machine learning (ML) cycle consisting of the finite element method (FEM) and 3D neural networks. Specifically, we applied our method to orthopedic implant design. Compared to expert designs, our experience-free method designed microscale heterogeneous architectures with a biocompatible elastic modulus and higher strength. Furthermore, inspired by the knowledge learned from the neural networks, we developed machine-human synergy, adapting the ML-designed architecture to fix a macroscale, irregularly shaped animal bone defect. Such adaptation exhibits 20% higher experimental load-bearing capacity than the expert design. Thus, our method opens a new paradigm for the fast and intelligent design of architected materials with tailored mechanical, physical, and chemical properties.

## Requirements

The codes are tested on Ubuntu 18.04, CUDA 11.4 with the following packages:

```shell
tensorflow-gpu == 2.5.0
keras == 2.3.1
tqdm == 4.59.0
scipy == 1.6.2
numpy == 1.19.2
pandas == 1.2.4
```

## Pipeline

To run the generative architecture design-multiobjective active learning loop (GAD-MALL) pipeline, please follow these steps:

1. Train 3D-CAE model as the generative model by running the `3D_CAE.py` Python code in the folder `GAD-MALL`, please run the following line in terminal:

```shell
python 3D_CAE.py
```

**Note:** The raw data `3D_CAE_train.npy` for training 3D-CAE can be downloaded [here](https://drive.google.com/file/d/1BfmD4bsPS2hG5zm7XGLHc8lpUN_WqhgV/view?usp=share_link).

2. Train 3D-CNN models as surrogate models of GAD-MALL by running the `3D_CNN.py` Python code in the folder `GAD-MALL`, please run the following line in terminal:

```shell
python 3D_CNN.py
```
**Note:** The raw data for training 3D-CNN is in the folder `GAD-MALL/GAD-MALL/data/`. One big file named `Matrix60.npy` can be downloaded [here](https://drive.google.com/file/d/1VRH4X_mACxM82HoaplwV0ThaDiN3iPXm/view?usp=share_link).

3. To search for high-performance architected materials with specific elastic modulus and high yield strength using GAD-MALL, please run the following line in terminal:

```shell
bash run_GAD_MALL.sh
```
4. After completing GAD-MALL process, you'll obtain porosity matrices with specific predicted elastic modulus (E=2500 MPa, E=5000 MPa) and the highest predicted yield strength. You can use these matrices to generate TPMS-Gyroid scaffolds. For more about `TPMS-Gyroid Structure Generation`, see code and `README` in folder `TPMS-Gyroid-Generate-by-MATLAB`
5. Conduct Finite Element Method (FEM) analysis of the TPMS-Gyroid scaffolds using ABAQUS (or other software capable of mechanical simulation). For more about `Finite Element Method`, see code and `README` in folder `Finite Element Method by ABAQUS`
6. Once you've combined the new mechanical property data from FEM results with your initial dataset in the `Datasets` folder, return to step 2 and run the `3D_CNN.py` Python code again to update your surrogate models. Continue this active learning loop until you find Gyroid porosity matrices with promising mechanical properties.

## More

For more about `Model Parameters Optimizing using Bayesian Optimization`, see code and `README` in folder `Model Parameter Optimize_Bayesian Optimization`

For more about `Other State-of-the-art Active Learning Methods` compared with our best method GAD-MALL, see cede and `README` in folder `Other-State-of-the-art-Active-Learning`

## CItation

If you find this work interesting, welcome to cite our paper!

```

```
