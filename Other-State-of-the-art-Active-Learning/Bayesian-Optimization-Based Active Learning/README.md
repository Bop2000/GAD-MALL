# Bayesian-optimization-based active learning pipeline

## Requirements

The codes are tested on Ubuntu 18.04, CUDA 11.4 with the following packages:

```shell
bayesian-optimization == 1.2.0
tensorflow-gpu == 2.5.0
keras == 2.3.1
tqdm == 4.59.0
scipy == 1.6.2
numpy == 1.19.2
pandas == 1.2.4
```

## Pipeline
1. Train 3D-CNN models as surrogate models of the Bayesian-optimization-based active learning, please run the following line in terminal:

```shell
python 3dCNN.py
```
2. Run the Bayesian-Optimization-based Active Learning code to conduct Bayesian optimization, please run the following line in terminal:

```shell
python Bayesian_Optimization_Based_Active_Learning.py
```

3. After completing the Bayesian optimization process, you'll obtain porosity matrices with specific predicted elastic modulus and the highest predicted yield strength. You can use these matrices to generate TPMS-Gyroid scaffolds. To generate Gyroid, refer to the code in the `TPMS-Gyroid-Generate-by-MATLAB` folder.
4. Conduct Finite Element Method (FEM) analysis of the TPMS-Gyroid scaffolds using ABAQUS (or other software capable of mechanical simulation). Refer to the code in the `Finite Element Method by ABAQUS` folder.
5. Once you've combined the new mechanical property data from FEM results with your initial dataset in the `Datasets` folder, return to step 1 and run the `3dCNN.py` Python code again. Continue this active learning loop until you find Gyroid porosity matrices with promising mechanical properties.
