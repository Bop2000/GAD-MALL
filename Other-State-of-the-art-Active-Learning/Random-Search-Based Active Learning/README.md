# Random-search-based active learning pipeline

## Pipeline
1. Train 3D-CNN models as surrogate models, please run the following line in terminal:

```shell
python 3dCNN_RS.py
```
2. Randomly sample 10 million points. Your surrogate models will then evaluate each of these points and rank them to find the ideal ones. Please run the following line in terminal:

```shell
python Random-Search-Based-Active-Learning.py
```

3. Use the porosity matrices with the specific predicted elastic modulus and the highest predicted yield strength obtained in the previous step to generate TPMS-Gyroid scaffolds. To generate Gyroid, refer to the code in the `TPMS-Gyroid-Generate-by-MATLAB` folder.
4. Conduct Finite Element Method (FEM) analysis of the TPMS-Gyroid scaffolds using ABAQUS (or other software capable of mechanical simulation). Refer to the code in the `Finite Element Method by ABAQUS` folder.
5. Combine the new mechanical property data obtained from FEM results with your initial dataset in the 'Datasets' folder. Then, return to step 1 and run the '3dCNN.py' Python code again. Continue this active learning loop until you find Gyroid porosity matrices with promising mechanical properties.
