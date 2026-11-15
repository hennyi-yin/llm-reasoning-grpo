param([ValidateSet('independent','self-generate','self-compare','serving','multisample','selector')][string]$Mode = 'independent')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Add-Type -Namespace MathExtension -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
[MathExtension.Awake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
    $command = 'source ~/.venvs/llm-reasoning-grpo/bin/activate && source scripts/runtime_env.sh && exec python -u -m scripts.run_math_extension ' + $Mode
    & wsl.exe -d Ubuntu-24.04 -- bash -lc $command
    $runCode = $LASTEXITCODE
} finally {
    [MathExtension.Awake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
exit $runCode
