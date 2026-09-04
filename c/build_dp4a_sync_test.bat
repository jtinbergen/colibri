@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\src\colibri\c
set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin
set CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
"%CUDA_BIN%\nvcc.exe" -O2 -std=c++17 -arch=sm_61 -DCOLI_CUDA_BUILDING_DLL=0 ^
  -ccbin "%CCBIN%" ^
  tests\test_dp4a_sync.cu backend_cuda.cu backend_cuda_dp4a.cu ^
  -o tests\test_dp4a_sync.exe -lcudart
if errorlevel 1 exit /b 1
endlocal
