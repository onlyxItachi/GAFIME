@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
cd /d C:\Users\Hamza\Desktop\GAFIME
nvcc -O3 -shared -Xcompiler /MD -DGAFIME_BUILDING_DLL -I src/common -o gafime_cuda.dll src/cuda/kernels.cu > build.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Build failed! See build.log
    type build.log
    exit /b %ERRORLEVEL%
)
echo Build complete!
exit /b 0
