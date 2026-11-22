@echo off
rem ============================================================
rem  Basalt serve - RTX 3060 12 GB (sm_86), Qwen3.8-Flash-Next GSQ-RCO Q2_0
rem  Main config serve-3060.json: kv int8, ctx 131072 (elastic KV), spec 4,
rem  expert-cache auto, --mmap-experts, --adapt-swaps 0, --route-prior 1
rem  -> 36.4 tok/s decode on a 13k-token prompt, hit 72.7% (SPEEDUP2-3060.md, par. 13.1)
rem     the same prompt without --route-prior gives 22.8-26.1 tok/s but fewer crashes
rem
rem  serve-3060.cmd            start the server on http://127.0.0.1:8105/v1
rem  serve-3060.cmd spec8      same, an 8-token verify window
rem  serve-3060.cmd safe       ctx 131072 without --route-prior (the routing stays exact)
rem  serve-3060.cmd 16k        ctx 16384 instead of 131072 (a bit faster, less RAM)
rem  serve-3060.cmd 64k        ctx 65536 (the ceiling before long-prompt prefill breaks)
rem  serve-3060.cmd stable     no --mmap-experts: fewer crashes, ~12 tok/s
rem  serve-3060.cmd stop       kill basalt.exe and the server's python
rem  Ctrl+C stops the server. Orphans of an earlier run are killed before start.
rem  Start is refused when free RAM < 40 GiB (mmap pins ~31.6 GiB page-locked).
rem ============================================================
setlocal
cd /D C:\Neurolib\Tools\basalt-ampere-3060

call :cleanup
if "%1"=="stop" (
    echo Basalt server stopped.
    goto end
)

rem ---- memory budget: mmap-experts pins ~31.6 GiB of page-locked RAM ----
for /f %%a in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1048576)"') do set FREEGB=%%a
echo Free RAM before start: %FREEGB% GiB
if "%1"=="stable" goto runstable
if %FREEGB% LSS 40 (
    echo Not enough free RAM to start with --mmap-experts: %FREEGB% GiB free, 40 GiB needed.
    echo Close other programs, or use the stable variant: serve-3060.cmd stable
    goto end
)

if "%1"=="spec8" goto run8
if "%1"=="safe" goto runsafe
if "%1"=="16k" goto run16k
if "%1"=="64k" goto run64k
echo Starting Basalt on http://127.0.0.1:8105/v1  (ctx 131072, spec 4, route-prior 1, mmap experts)
bin\basalt.exe serve --config serve-3060.json
goto end

:run8
echo Starting Basalt on http://127.0.0.1:8105/v1  (ctx 131072, spec 8, route-prior 1)
bin\basalt.exe serve --config serve-3060-spec8.json
goto end

:runsafe
echo Starting Basalt on http://127.0.0.1:8105/v1  (ctx 131072, spec 4, exact routing: no route-prior)
bin\basalt.exe serve --config serve-3060-safe.json
goto end

:run16k
echo Starting Basalt on http://127.0.0.1:8105/v1  (ctx 16384, spec 4, route-prior 1)
bin\basalt.exe serve --config serve-3060-16k.json
goto end

:run64k
echo Starting Basalt on http://127.0.0.1:8105/v1  (ctx 65536, spec 4, route-prior 1)
bin\basalt.exe serve --config serve-3060-65k.json
goto end

:runstable
echo Starting Basalt on http://127.0.0.1:8105/v1  (no mmap: stable, slow)
bin\basalt.exe serve --config serve-3060-stable.json
goto end

:end
endlocal
exit /b 0

:cleanup
taskkill /f /im basalt.exe >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'serve.server' } | ForEach-Object { taskkill /f /pid $_.ProcessId }"
goto :eof
