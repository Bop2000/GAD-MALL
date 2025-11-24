# Finite Element Simulation Pipeline Automation
This is the guideline to automate the pipeline of FEM (Matlab for Gyroid structures generation in STL format → Hypermesh draw the mesh in .inp format → Abaqus conduct FEM analysis and output a .odb file for further calculation of mechanical properties).

## Environements
The developmental version of the package has been tested on the following systems and softwares.
- Windows 10
- Matlab R2020a
- Hypermesh 2019
- Abaqus 2018

## RUN
To run this automated pipeline, the only thing is to use PowerShell to run `BatchProcess.ps1`

**Note:** The initial porosity matrices are archived in the `finished.mat` file. Upon executing 'BatchProcess.ps1', the process unfolds in three stages. Firstly, Matlab is invoked to execute `test1.m`, which transforms the porosity matrices into corresponding Gyroid STL files. The second stage involves deploying Hypermesh to run `command.tcl`, which converts STL files into mesh files. In the final stage, Abaqus is employed to execute `BatchProcess_v0.4.py`, which conducts the Finite Element Method (FEM) analysis and archives the output as .odb files.

## Note
1. If the program cannot be run, please check if the paths in the code are correct.
2. If the FEM results are not rational, please kindly refer to the file `BatchProcess_1.py` in loacation `Finite Element Method by ABAQUS/abaqus_code`.
