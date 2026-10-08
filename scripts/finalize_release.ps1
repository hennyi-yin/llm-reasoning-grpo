param([Parameter(Mandatory=$true)][int]$RunnerPid)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot 'logs/release_runner.json'
$state = [ordered]@{
    status = 'waiting_for_measured_pipeline'
    pid = $PID
    runner_pid = $RunnerPid
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    uploaded = $false
}
$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
try {
    # Follow only the already running launcher, without starting a second GPU run.
    if (Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $RunnerPid
    }
    $runner = Get-Content -LiteralPath (Join-Path $projectRoot 'logs/windows_runner.json') -Raw | ConvertFrom-Json
    if ($runner.pid -ne $RunnerPid -or $runner.status -ne 'complete' -or $runner.exit_code -ne 0) {
        throw 'The expected measured pipeline did not finish successfully; Release packaging was not started.'
    }
    $state.status = 'packaging'
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    Add-Type -Namespace LLMProject -Name ReleaseAwake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
    [LLMProject.ReleaseAwake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    Set-Location -LiteralPath $projectRoot
    & wsl.exe -d Ubuntu-24.04 -- bash -lc 'source "$HOME/.venvs/llm-reasoning-grpo/bin/activate" && exec python -u -m scripts.package_release'
    $packageExitCode = $LASTEXITCODE
    if ($packageExitCode -ne 0) { throw "Release packaging failed with exit $packageExitCode." }
    $state.status = 'complete'
} catch {
    $state.status = 'failed'
    $state.error = $_.Exception.Message
    throw
} finally {
    if ($state.status -ne 'waiting_for_measured_pipeline' -and ('LLMProject.ReleaseAwake' -as [type])) {
        [LLMProject.ReleaseAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    }
    $state.ended_at_utc = [DateTime]::UtcNow.ToString('o')
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
}
