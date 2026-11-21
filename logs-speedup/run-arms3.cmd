@echo off
set EXE=C:\Neurolib\Tools\basalt-ampere-3060\bin\basalt.exe
set MODEL=C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt
set TOK=C:\Neurolib\Tools\basalt-ampere-3060\data\bench\conv.ids
set DIR=C:\Neurolib\Tools\basalt-ampere-3060\logs-speedup

call :run spec12 1 --spec 12
call :run spec12 2 --spec 12
call :run spec16 1 --spec 16
call :run tail3 1 --route-tail-skip 3
call :run adapt0 2 --adapt-swaps 0
call :run base3 1
call :run kvq40 2 --kv q4_0
exit /b 0

:run
set NAME=%1
set REP=%2
shift
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
echo [=%NAME% rep%REP%] %FLAGS%
%EXE% --model %MODEL% --tokens-file %TOK% --max-new 800 --greedy --seed 42 --kv int8 --max-context 8192 --spec 8 %FLAGS% > %DIR%\%NAME%-r%REP%.log 2>&1
findstr /r "decode  speculation verify window pool multi" %DIR%\%NAME%-r%REP%.log
exit /b 0
