@echo off
set EXE=C:\Neurolib\Tools\basalt-ampere-3060\bin\basalt.exe
set MODEL=C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt
set TOK=C:\Neurolib\Tools\basalt-ampere-3060\data\bench\conv.ids
set DIR=C:\Neurolib\Tools\basalt-ampere-3060\logs-speedup

call :run tail1 r1 --route-tail-skip 1
call :run tail3 r1 --route-tail-skip 3
call :run prior1 r1 --route-prior 1
call :run adapt1 r1 --adapt-every 1
call :run swaps256 r1 --adapt-swaps 256
call :run tail2p r1 --route-tail-skip 2 --adapt-every 1
call :run tail2kv r1 --route-tail-skip 2 --kv q4_0
call :run base r3
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
echo [=%NAME%-%REP%] %FLAGS%
%EXE% --model %MODEL% --tokens-file %TOK% --max-new 1200 --greedy --seed 42 --kv int8 --max-context 8192 --spec 8 %FLAGS% > %DIR%\%NAME%-%REP%.log 2>&1
findstr /r "decode  speculation verify window pool multi expert cache [0-9]" %DIR%\%NAME%-%REP%.log
exit /b 0
