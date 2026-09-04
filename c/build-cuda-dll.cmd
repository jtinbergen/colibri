@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
cd /d "%~dp0"
echo === Rebuilding coli_cuda.dll with COLI_CUDA_BUILDING_DLL ===
"%CUDA_HOME%\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_61 -Xcompiler=/W3 -shared -Wno-deprecated-gpu-targets ^
  -D COLI_CUDA_BUILDING_DLL ^
  -o coli_cuda.dll backend_cuda.cu backend_cuda_dp4a.cu ^
  -L"%CUDA_HOME%\lib\x64" -lcudart
if errorlevel 1 (echo NVCC FAILED & exit /b 2)
echo === Done ===
dir coli_cuda.dll | findstr coli_cuda
