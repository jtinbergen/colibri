@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "%VSWHERE%" for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" exit /b 1
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
pushd "%~dp0"
"%CUDA_BIN%\nvcc.exe" -O3 -std=c++17 -arch=sm_61 -Xcompiler=/W3 tests/test_dense_exact.cu backend_cuda.cu backend_cuda_dp4a.cu -o tests/test_dense_exact.exe
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
