$model = "C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt"
$tok = "C:\Neurolib\Tools\basalt-ampere-3060\data\bench\conv.ids"
foreach ($fb in 28, 16, 8, 4) {
    Get-Process basalt -ErrorAction SilentlyContinue | ForEach-Object { taskkill /f /pid $_.Id }
    $log = "C:\Neurolib\Tools\basalt-ampere-3060\sweep-fb$fb.log"
    cmd /C "C:\Neurolib\Tools\basalt-ampere-3060\bin\basalt.exe --model $model --tokens-file $tok --max-new 60 --greedy --seed 42 --kv int8 --max-context 8192 --spec 8 --exp fin_blocks=$fb > $log 2>&1"
    $d = Select-String -Path $log -Pattern "decode " | Select-Object -Last 1
    $o = Select-String -Path $log -Pattern "^output" | Select-Object -Last 1
    Write-Output ("fin_blocks={0} :: {1}" -f $fb, $d.Line)
    Write-Output ("  out: " + $o.Line.Substring(0, [Math]::Min(80, $o.Line.Length)))
}
