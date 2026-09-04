@echo off
setlocal
cd /d "%~dp0.."
C:\msys64\mingw64\bin\gcc.exe -D_FILE_OFFSET_BITS=64 -O2 -Wall -Wextra tests/test_st_mirror.c -o tests/test_st_mirror.exe
if errorlevel 1 exit /b %errorlevel%
tests\test_st_mirror.exe
