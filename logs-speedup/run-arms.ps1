$exe = "C:\Neurolib\Tools\basalt-ampere-3060\bin\basalt.exe"
$model = "C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt"
$tok = "C:\Neurolib\Tools\basalt-ampere-3060\data\bench\conv.ids"
$dir = "C:\Neurolib\Tools\basalt-ampere-3060\logs-speedup"

$arms = @(
  @{n='base';     f=''},
  @{n='tail2';   f='--route-tail-skip 2'},
  @{n='prior1';  f='--route-prior 1'},
  @{n='cache6000'; f='--expert-cache 6000'},
  @{n='smt15';   f='--pool-workers 15 --pool-affinity all'},
  @{n='workers8'; f='--pool-workers 8'},
  @{n='gpuonly'; f='--gpu-only-full'}
)

foreach ($a in $arms) {
    Get-Process basalt -ErrorAction SilentlyContinue | ForEach-Object { taskkill /f /pid $_.Id }
    Start-Sleep -Seconds 4
    $os = Get-CimInstance Win32_OperatingSystem
    $log = "$dir\$($a.n).log"
    Write-Output ("[{0}] RAM free {1} GiB :: {2}" -f $a.n, ($os.FreePhysicalMemory/1MB), $a.f)
    cmd /C "$exe --model $model --tokens-file $tok --max-new 400 --greedy --seed 42 --kv int8 --max-context 8192 --spec 8 $($a.f) > $log 2>&1"
    foreach ($s in (Select-String -Path $log -Pattern 'decode |prefill [0-9]|pool multi|verify window|pcie experts|expert cache [0-9]|gpu-only|error')) {
        Write-Output ("  " + $s.Line.Trim())
    }
}
