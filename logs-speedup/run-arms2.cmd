@echo off
set EXE=C:\Neurolib\Tools\basalt-ampere-3060\bin\basalt.exe
set MODEL=C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt
set TOK=C:\Neurolib\Tools\basalt-ampere-3060\data\bench\conv.ids
set DIR=C:\Neurolib\Tools\basalt-ampere-3060\logs-speedup

call :run tail5 --route-tail-skip 5
call :run prior2 --route-prior 2
call :run kvq40 --kv q4_0
call :run adapt0 --adapt-swaps 0
call :run spec4 --spec 4
call :run gpuonly2 --gpu-only-full --no-pool
call :run base2
exit /b 0

:run
set NAME=%1
shift
set FLAGS=%1
:moreargs
shift
if "%1"=="" goto doneargs
set FLAGS=%FLAGS% %1
goto moreargs
:doneargs
taskkill /f /im basalt.exe >nul 2>&1
ping -n 5 127.0.0.1 >nul
echo [=%NAME%] %FLAGS%
%EXE% --model %MODEL% --tokens-file %TOK% --max-new 400 --greedy --seed 42 --kv int8 --max-context 8192 --spec 8 %FLAGS% > %DIR%\%NAME%.log 2>&1
findstr /r "decode  window sizes verify window pool multi pcie experts expert cache [0-9] error warning" %DIR%\%NAME%.log
exit /b 0
