param([switch]$ProbeScale)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Add-Type -Namespace MathSft -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
[MathSft.Awake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
    $linux = 'source ~/.venvs/llm-reasoning-grpo/bin/activate && source scripts/runtime_env.sh && exec python -u -m scripts.run_math_sft'
    if ($ProbeScale) { $linux += ' --probe-scale' }
    & wsl.exe -d Ubuntu-24.04 -- bash -lc $linux
    $runCode = $LASTEXITCODE
} finally {
    [MathSft.Awake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
exit $runCode
