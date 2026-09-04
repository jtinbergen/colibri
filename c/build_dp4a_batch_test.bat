@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
set CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
pushd "%~dp0"
nvcc -O3 -std=c++17 -arch=sm_61 -DCOLI_CUDA_BUILDING_DLL=0 ^
    -ccbin "%CCBIN%" backend_cuda_dp4a.cu tests\test_dp4a_batch.cu ^
    -o tests\test_dp4a_batch.exe -lcudart
set RC=%ERRORLEVEL%
popd
exit /b %RC%
