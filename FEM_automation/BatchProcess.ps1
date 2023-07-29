#!powershell
# FileName:     BatchProcess.ps1

# User's scripts definition Beginning
$matlabScript='test1'
$hypermeshScript='E:\qinyu\daijiabao\mycommand.tcl'
$abaqusScript='E:\qinyu\daijiabao\BatchProcess_v0.4.py'

# User's scripts definition Ending

# Matlab Modeling Process
# echo 'Matlab Modeling...'
# matlab -nosplash -nodesktop -r $matlabScript | Out-Null

# HyperMesh Preprocessing
# echo 'HyperMesh Processing...'
# D:\HYPER\hm\bin\win64\hmopengl.exe -tcl $hypermeshScript | Out-Null

# Abaqus Modeling and processing
echo 'Abaqus Processing...'
abaqus cae -script $abaqusScript | Out-Null

# Exiting
echo 'Full Process Completed Successfully!'
echo 'Press any key for exitting...'
Pause
exit