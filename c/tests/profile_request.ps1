param(
    [string]$RequestFile = 'D:\src\colibri\c\tests\chat_req.json',
    [string]$Label = 'run'
)

$sampleJob = Start-Job -ArgumentList $Label -ScriptBlock {
    param($sampleLabel)
    $lastCpu = $null
    $lastTime = [DateTime]::UtcNow
    while ($true) {
        $now = [DateTime]::UtcNow
        $gpu = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>$null
        $p = Get-Process qwen36 -ErrorAction SilentlyContinue
        $cpuPct = 0.0
        if ($p -and $null -ne $lastCpu) {
            $dt = ($now - $lastTime).TotalSeconds
            if ($dt -gt 0) { $cpuPct = (($p.CPU - $lastCpu) / $dt) / [Environment]::ProcessorCount * 100.0 }
        }
        if ($p) { $lastCpu = $p.CPU; $lastTime = $now }
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
        if ($gpu) {
            $g = ($gpu -split ',') | ForEach-Object { $_.Trim() }
            $ramUsed = if ($os) { ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1048576.0 } else { 0 }
            $ramTotal = if ($os) { $os.TotalVisibleMemorySize / 1048576.0 } else { 0 }
            $ws = if ($p) { $p.WorkingSet64 / 1GB } else { 0 }
            $priv = if ($p) { $p.PrivateMemorySize64 / 1GB } else { 0 }
            '{0} {1} GPU={2}% MEMUTIL={3}% VRAM={4}/{5}MiB POWER={6}W TEMP={7}C PCIe={8}x{9} CPU={10:N1}% RAM={11:N1}/{12:N1}GB WS={13:N2}GB PRIV={14:N2}GB' -f (Get-Date -Format HH:mm:ss),$sampleLabel,$g[0],$g[1],$g[2],$g[3],$g[4],$g[5],$g[6],$g[7],$cpuPct,$ramUsed,$ramTotal,$ws,$priv
        }
        Start-Sleep -Milliseconds 1000
    }
}

$sw = [Diagnostics.Stopwatch]::StartNew()
$curl = curl.exe -sS -o NUL -w 'HTTP=%{http_code} curl_s=%{time_total}' -H 'Content-Type: application/json' --data-binary "@$RequestFile" http://127.0.0.1:8000/v1/chat/completions
$sw.Stop()
Stop-Job $sampleJob | Out-Null
$samples = Receive-Job $sampleJob
Remove-Job $sampleJob
Write-Output ('RESULT ' + $curl + (' elapsed_s={0:N3}' -f $sw.Elapsed.TotalSeconds))
$samples
