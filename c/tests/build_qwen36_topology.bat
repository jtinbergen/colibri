@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0.."
cl /nologo /W4 /EHsc /std:c11 tests\test_qwen36_topology.c qwen36_topology.c /Fe:tests\test_qwen36_topology.exe
if errorlevel 1 exit /b %errorlevel%
tests\test_qwen36_topology.exe
