To run the Bayesian-optimization-based active learning pipeline, you'll need to have TensorFlow installed in your Python environment. Follow these steps:

1. Train 3D-CNN models as surrogate models of the Bayesian-optimization-based active learning by running the '3dCNN.py' Python code in this folder.
2. Run the 'Bayesian-Optimization-Based-Active-Learning.py' Python code to conduct Bayesian optimization.
3. After completing the Bayesian optimization process, you'll obtain porosity matrices with specific predicted elastic modulus and the highest predicted yield strength. You can use these matrices to generate TPMS-Gyroid scaffolds. To generate Gyroid, refer to the code in the 'TPMS-Gyroid-Generate-by-MATLAB' folder.
4. Conduct Finite Element Method (FEM) analysis of the TPMS-Gyroid scaffolds using ABAQUS (or other software capable of mechanical simulation). Refer to the code in the 'Finite Element Method by ABAQUS' folder.
5. Once you've combined the new mechanical property data from FEM results with your initial dataset in the 'Datasets' folder, return to step 1 and run the '3dCNN.py' Python code again. Continue this active learning loop until you find Gyroid porosity matrices with promising mechanical properties.
