@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
nvcc -c -Xptxas -v src/cuda/kernels.cu -I src/common -O3 -DGAFIME_BUILDING_DLL > regs.txt 2>&1
type regs.txt
