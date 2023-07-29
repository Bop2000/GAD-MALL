#!powershell
# FileName:     5000BatchProcess.ps1
# Author:       Anonymous
# Date:         2021-08-02
# Version:      1.1.0
# Copyright:    Tsinghua University, Department of Mechanical Engineering

# User's scripts definition Beginning
$matlabScript='test1'
$hypermeshScript='E:\qinyu\daijiabao\mycommand.tcl'

# User's scripts definition Ending

# Matlab Modeling Process
echo 'Matlab Modeling...'
matlab -nosplash -nodesktop -r $matlabScript | Out-Null

# HyperMesh Preprocessing
echo 'HyperMesh Processing...'
D:\HYPER\hm\bin\win64\hmopengl.exe -tcl $hypermeshScript | Out-Null

echo 'Press any key for exitting...'
Pause
exit