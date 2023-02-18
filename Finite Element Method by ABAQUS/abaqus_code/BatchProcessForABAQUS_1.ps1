#!powershell
# FileName:     5000BatchProcess.ps1
# Author:       Anonymous
# Date:         2021-08-02
# Version:      1.1.0
# Copyright:    Tsinghua University, Department of Mechanical Engineering

# Abaqus Modeling and processing
# Assign your required Model Number Here.
$ModelNum = 6
# Assign your abaqus Parameter File Here.
$ParamFile = "./N_1.txt"
# Abaqus process
for ($ModelId = 1; $ModelId -le $ModelNum; $ModelId++) {
    echo "Model $ModelId loading into Abaqus..."
    echo $ModelId | Out-File -Encoding ascii -FilePath $ParamFile
    echo 'Abaqus Processing...'
    start cmd_1.bat
    while( -not ($(ls E:\daijiabao\abaqus1000randomTi\BO_R10\rpt\ | Select "Name" | findstr "Ti_BO_R10_$($ModelId).rpt"))){
    sleep 2
}
    sleep 10
    $a_1 = Get-Content cmd_1.txt
    $a_1.split(",")>a_1.txt
    $b_1 = (Get-Content a_1.txt)[1..1]
    taskkill /pid $b_1
    del cmd_1.txt
    del a_1.txt
    sleep 10
}
# Exiting
echo 'Full Process Completed Successfully!'
# The round-off work.
del $ParamFile
echo 'Press any key for exitting...'
Pause
exit