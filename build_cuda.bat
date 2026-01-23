@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
cd /d C:\Users\Hamza\Desktop\GAFIME
nvcc -O3 -shared -Xcompiler /MD -DGAFIME_BUILDING_DLL -I src/common -o gafime_cuda.dll src/cuda/kernels.cu
echo Build complete!
pause
