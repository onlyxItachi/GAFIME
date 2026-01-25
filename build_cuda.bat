@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -shared -o gafime_cuda.dll src/cuda/kernels.cu -I src/common -DGAFIME_BUILDING_DLL -O3
if %errorlevel% neq 0 exit /b %errorlevel%
echo Build Successful
