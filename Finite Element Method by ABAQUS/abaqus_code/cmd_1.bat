    title=cmd_1
    tasklist /v /fo csv | findstr /i cmd_1 >cmd_1.txt
    abaqus cae -script BatchProcess_1.py