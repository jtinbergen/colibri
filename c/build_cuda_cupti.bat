@echo off
rem Build a diagnostic CUDA backend beside the normal backend.
rem The resulting DLL is loaded only when COLI_CUDA_DLL_PATH points at it.
setlocal
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set CUPTI_HOME=%CUDA_HOME%\extras\CUPTI
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=%CUDA_HOME%\bin;%CUPTI_HOME%\lib64;%PATH%
set CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
pushd "%~dp0"
nvcc -O3 -std=c++17 -arch=sm_61 -Xcompiler=-W3 -shared ^
    -ccbin "%CCBIN%" -DCOLI_CUDA_BUILDING_DLL -DCOLI_CUPTI_TRACE ^
    -I"%CUPTI_HOME%\include" ^
    backend_cuda.cu backend_cuda_dp4a.cu ^
    -L"%CUDA_HOME%\lib\x64" -lcudart ^
    -L"%CUPTI_HOME%\lib64" -lcupti ^
    -o coli_cuda_cupti.dll
set RC=%ERRORLEVEL%
popd
echo exit=%RC%
exit /b %RC%
