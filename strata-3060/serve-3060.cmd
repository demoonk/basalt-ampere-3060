@echo off
rem Strata 3060 - the same launcher idea as serve-3060.cmd, for the Strata engine.
rem   no argument / 16k   ctx 16384, spec 4, kv int8        -> http://127.0.0.1:8103/v1
rem   32k                 ctx 32768, drafting off
rem   64k                 ctx 65536, kv-resident 32768, the MTP draft layer
rem   stop                kill the engine and the serve python
rem Guards: only one engine at a time (Basalt and Strata each page-lock 31.6 GiB),
rem and the run refuses to start below 40 GiB of free RAM.
setlocal
set ROOT=C:\Neurolib\Tools\strata-3060
set PY=C:\Neurolib\Tools\Strata\terminal\current\python\python.exe
set SERVER=C:\Neurolib\Tools\Strata\terminal\0.1.59\serve\server.py
set PORT=8103

if /i "%1"=="stop" goto stop

tasklist /FI "IMAGENAME eq basalt.exe" | find /i "basalt.exe" >nul
if not errorlevel 1 (
  echo Basalt is running. Strata and Basalt each pin 31.6 GiB of RAM and take the whole card:
  echo stop Basalt first with the "Basalt 3060 stop" shortcut.
  exit /b 1
)
tasklist /FI "IMAGENAME eq strata.exe" | find /i "strata.exe" >nul
if not errorlevel 1 (
  echo Strata is already running. Stop it with the "Strata 3060 stop" shortcut first.
  exit /b 1
)

for /f "delims=" %%f in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1048576)"') do set FREE=%%f
if %FREE% LSS 40 (
  echo Only %FREE% GiB of RAM free. The expert arena page-locks 31.64 GiB, and the engine's own
  echo guard refuses below 2 GiB. Close things and try again.
  exit /b 3
)
echo Free RAM %FREE% GiB. The arena pins 31.6 GiB, the expert cache takes what the 12 GB card leaves
echo after the weights (about 4.8 GiB, auto). Engine ready in 8-15 minutes on this CPU.

if /i "%1"=="32k" (set CFG=%ROOT%\strata-32k.json) else if /i "%1"=="64k" (set CFG=%ROOT%\strata-64k.json) else (set CFG=%ROOT%\strata-16k.json)

title Strata 3060 server %1
cd /d "%ROOT%"
"%PY%" "%SERVER%" --engine strata --config "%CFG%" --port %PORT%
exit /b %errorlevel%

:stop
taskkill /F /IM strata.exe >nul 2>nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'Strata.*server\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo Strata stopped.
exit /b 0
