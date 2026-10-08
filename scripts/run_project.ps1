param(
    [ValidateSet('memory','sft','training','serving','final')]
    [string]$Through = 'final',
    [switch]$WaitForSmoke
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runLogs = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $runLogs | Out-Null
$hostState = Join-Path $runLogs 'windows_runner.json'
$runState = [ordered]@{
    status = 'running'
    pid = $PID
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    through = $Through
    root = $projectRoot
}
$runState | ConvertTo-Json | Set-Content -LiteralPath $hostState -Encoding utf8
Add-Type -Namespace LLMProject -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
[LLMProject.Awake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
    Set-Location -LiteralPath $projectRoot
    if ($WaitForSmoke) {
        $runState.status = 'waiting_for_memory_test'
        $runState | ConvertTo-Json | Set-Content -LiteralPath $hostState -Encoding utf8
        do {
            & wsl.exe -d Ubuntu-24.04 -- bash -lc "pgrep -f '[s]rc.smoke' >/dev/null"
            $smokeIsRunning = $LASTEXITCODE -eq 0
            if ($smokeIsRunning) { Start-Sleep -Seconds 10 }
        } while ($smokeIsRunning)
        $runState.status = 'running'
        $runState | ConvertTo-Json | Set-Content -LiteralPath $hostState -Encoding utf8
    }
    $linuxCommand = 'source "$HOME/.venvs/llm-reasoning-grpo/bin/activate" && exec python -u -m src.pipeline --through ' + $Through
    & wsl.exe -d Ubuntu-24.04 -- bash -lc $linuxCommand
    $runExitCode = $LASTEXITCODE
    $runState.status = if ($runExitCode -eq 0) { 'complete' } else { 'failed' }
    $runState.exit_code = $runExitCode
} catch {
    $runState.status = 'failed'
    $runState.error = $_.Exception.Message
    $runExitCode = 1
    throw
} finally {
    [LLMProject.Awake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    $runState.ended_at_utc = [DateTime]::UtcNow.ToString('o')
    $runState | ConvertTo-Json | Set-Content -LiteralPath $hostState -Encoding utf8
}
exit $runExitCode
