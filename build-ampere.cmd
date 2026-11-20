@echo off
REM Configure and build the Ampere (sm_86) fork on HomePC.  %1 = ninja -j
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo [buildenv] vcvars64.bat failed & exit /b 1 )

set "CUDA_PATH=C:\Neurolib\Tools\cuda\13.4.3"
set "CUDAToolkit_ROOT=%CUDA_PATH%"
set "CUDA_HOME=%CUDA_PATH%"
set "TMP=C:\Neurolib\Tools\basalt-ampere-3060\build-tmp"
set "TEMP=%TMP%"
if not exist "%TMP%" mkdir "%TMP%"
set "VS_CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

cd /d "C:\Neurolib\Tools\basalt-ampere-3060"
"%VS_CMAKE%" -S . -B build-3060 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=86 -DBASALT_MODELS=qwen4exp
"%NINJA%" -C build-3060 -k 25 -j %1
