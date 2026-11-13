param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')]
    [string]$RunName,
    [string]$Resume
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
Add-Type -Namespace MathPilot -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
[MathPilot.Awake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
    $linuxCommand = 'source ~/.venvs/llm-reasoning-grpo/bin/activate && source scripts/runtime_env.sh && exec python -u -m scripts.run_math_pilot'
    $pilotOutput = 'results/followup'
    if ($RunName) {
        $pilotOutput = "results/followup/work/reproductions/$RunName"
        $linuxCommand = "export LLM_MATH_PILOT_OUT=$pilotOutput; " + $linuxCommand
    }
    if ($Resume) {
        if ($Resume -notmatch ('^' + [regex]::Escape($pilotOutput) + '/work/math_adapter/checkpoint-[0-9]+$')) {
            throw 'Resume must be a project-relative checkpoint path for the selected Math pilot run.'
        }
        $linuxCommand += ' --resume ' + $Resume
    }
    & wsl.exe -d Ubuntu-24.04 -- bash -lc $linuxCommand
    $code = $LASTEXITCODE
} finally {
    [MathPilot.Awake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
exit $code
