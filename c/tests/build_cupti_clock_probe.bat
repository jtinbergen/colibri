@echo off
setlocal
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set CUPTI_HOME=%CUDA_HOME%\extras\CUPTI
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=%CUDA_HOME%\bin;%CUPTI_HOME%\lib64;%PATH%
set CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
pushd "D:\src\colibri\c"
nvcc -O2 -std=c++17 -arch=sm_61 -Xcompiler=-W3 -ccbin "%CCBIN%" ^
  -I"%CUPTI_HOME%\include" tests\cupti_clock_probe.cu ^
  -L"%CUDA_HOME%\lib\x64" -lcudart ^
  -L"%CUPTI_HOME%\lib64" -lcupti -o tests\cupti_clock_probe.exe
set RC=%ERRORLEVEL%
popd
echo exit=%RC%
exit /b %RC%
