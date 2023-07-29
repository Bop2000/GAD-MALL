#!powershell
# FileName:     5000BatchProcess.ps1
# Author:       Anonymous
# Date:         2021-08-02
# Version:      1.1.0
# Copyright:    Tsinghua University, Department of Mechanical Engineering

# User's scripts definition Beginning
$matlabScript='test1'
$hypermeshScript='E:\qinyu\daijiabao\mycommand.tcl'
$abaqusScript='E:\qinyu\daijiabao\BatchProcess_v0.5.py'

# User's scripts definition Ending

# Matlab Modeling Process
# echo 'Matlab Modeling...'
# matlab -nosplash -nodesktop -r $matlabScript | Out-Null

# HyperMesh Preprocessing
# echo 'HyperMesh Processing...'
# D:\HYPER\hm\bin\win64\hmopengl.exe -tcl $hypermeshScript | Out-Null

# Abaqus Modeling and processing
# Assign your required Model Number Here.
$ModelNum = 5006
# Assign your abaqus Parameter File Here.
$ParamFile = "./N.txt"
# Abaqus process
for ($ModelId = 4; $ModelId -le $ModelNum; $ModelId++) {
    echo "Model $ModelId loading into Abaqus..."
    echo $ModelId | Out-File -Encoding ascii -FilePath $ParamFile
    echo 'Abaqus Processing...'
    start E:\qinyu\daijiabao\suijishengcheng-matlab\killabaqus.bat
    while( -not ($(ls E:\qinyu\daijiabao\rpt\ | Select "Name" | findstr "abaqusModel$($ModelId).rpt"))){
    echo "waiting..."
    sleep 2
}
    sleep 1
    taskkill /IM "cmd.exe"
}
# Get-Process -Name "cmd" | Stop-Process
# Exiting
echo 'Full Process Completed Successfully!'
# The round-off work.
del $ParamFile
echo 'Press any key for exitting...'
Pause
exit