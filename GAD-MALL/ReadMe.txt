To run the generative architecture design-multiobjective active learning loop (GAD-MALL) pipeline, you'll need to have TensorFlow installed in your Python environment. Follow these steps:

1. Train 3D-CAE model as the generative model by running the '3D_CAE.py' Python code in this folder.
2. Train 3D-CNN models as surrogate models of GAD-MALL by running the '3D_CNN.py' Python code in this folder.
3. Run the 'GAD-MALL_Active_learning.py' Python code to conduct GAD-MALL.
4. After completing GAD-MALL process, you'll obtain porosity matrices with specific predicted elastic modulus (E=2500 MPa, E=5000 MPa) and the highest predicted yield strength. You can use these matrices to generate TPMS-Gyroid scaffolds. To generate Gyroid, refer to the code in the 'TPMS-Gyroid-Generate-by-MATLAB' folder.
5. Conduct Finite Element Method (FEM) analysis of the TPMS-Gyroid scaffolds using ABAQUS (or other software capable of mechanical simulation). Refer to the code in the 'Finite Element Method by ABAQUS' folder.
6. Once you've combined the new mechanical property data from FEM results with your initial dataset in the 'Datasets' folder, return to step 2 and run the '3D_CNN.py' Python code again. Continue this active learning loop until you find Gyroid porosity matrices with promising mechanical properties.

The file "3D_CAE_Train.npy" could be download through the link below, save the file in the folder 'data': 

https://drive.google.com/file/d/1BfmD4bsPS2hG5zm7XGLHc8lpUN_WqhgV/view?usp=share_link

The file "Matrix60.npy" could be download through the link below, save the file in the folder 'data': 

https://drive.google.com/file/d/1VRH4X_mACxM82HoaplwV0ThaDiN3iPXm/view?usp=share_link