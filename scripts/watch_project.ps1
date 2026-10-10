param(
    [ValidateRange(2, 3600)]
    [int]$IntervalSeconds = 10,
    [switch]$Once,
    [switch]$ShowGpu
)

# Read-only: this script never starts, stops, or resumes the project runner.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $projectRoot 'logs/windows_runner.json'
$pipelinePath = Join-Path $projectRoot 'results/pipeline_state.json'
$releaseRunnerPath = Join-Path $projectRoot 'logs/release_runner.json'

do {
    if (-not $Once -and -not [Console]::IsOutputRedirected) { Clear-Host }
    Write-Output ('项目监测  ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
    try {
        if (Test-Path -LiteralPath $runnerPath) {
            $runner = Get-Content -LiteralPath $runnerPath -Raw -Encoding utf8 | ConvertFrom-Json
            $runnerProcess = Get-Process -Id $runner.pid -ErrorAction SilentlyContinue
            $processInfo = if ($runnerProcess) { $runnerProcess.ProcessName } else { '已退出' }
            Write-Output ('后台状态: {0} | Windows PID: {1} | 进程: {2}' -f $runner.status, $runner.pid, $processInfo)
            if ($runner.error) { Write-Output ('后台错误: ' + $runner.error) }
        } else {
            Write-Output '尚无后台运行状态文件。'
        }

        if (Test-Path -LiteralPath $releaseRunnerPath) {
            $release = Get-Content -LiteralPath $releaseRunnerPath -Raw -Encoding utf8 | ConvertFrom-Json
            $releaseAlive = [bool](Get-Process -Id $release.pid -ErrorAction SilentlyContinue)
            if ($release.repository) {
                Write-Output ('GitHub 发布: {0} | PID: {1} | 存活: {2} | 已核验文件: {3}/{4}' -f $release.status, $release.pid, $releaseAlive, $release.completed_assets, $release.total_assets)
            } else {
                Write-Output ('旧本地打包记录: {0} | PID: {1} | 存活: {2}' -f $release.status, $release.pid, $releaseAlive)
            }
            if ($release.asset) { Write-Output ('当前上传文件: ' + $release.asset) }
            if ($release.error) { Write-Output ('发布错误: ' + $release.error) }
        }

        if (Test-Path -LiteralPath $pipelinePath) {
            $pipeline = Get-Content -LiteralPath $pipelinePath -Raw -Encoding utf8 | ConvertFrom-Json
            $stages = @($pipeline.stages.PSObject.Properties)
            $complete = @($stages | Where-Object { $_.Value.status -eq 'complete' })
            $active = @($stages | Where-Object { $_.Value.status -eq 'running' })
            $failed = @($stages | Where-Object { $_.Value.status -eq 'failed' })
            Write-Output ('已完成阶段数: ' + $complete.Count + '（不是总任务百分比）')
            foreach ($stage in $active) {
                $started = [DateTimeOffset]::Parse($stage.Value.started_at)
                $elapsed = [DateTimeOffset]::UtcNow - $started
                Write-Output ('当前阶段: {0} | 已运行: {1:0.0} 分钟' -f $stage.Name, $elapsed.TotalMinutes)
                $stageLog = Join-Path $projectRoot ('logs/pipeline/' + $stage.Name + '.log')
                if (Test-Path -LiteralPath $stageLog) {
                    Write-Output ('最近日志: ' + $stageLog)
                    Get-Content -LiteralPath $stageLog -Tail 4 -Encoding utf8
                }
            }
            foreach ($stage in $failed) {
                $failureLabel = if ($runnerProcess -and $runner.status -eq 'running') { '保留的先前失败记录' } else { '失败阶段' }
                Write-Output ('{0}: {1} | {2}' -f $failureLabel, $stage.Name, $stage.Value.error)
            }
            if (-not $active.Count) {
                Write-Output '没有标记为运行中的实验阶段；后台可能在核验旧记录或本地打包。'
                $latestConsole = Get-ChildItem -LiteralPath (Join-Path $projectRoot 'logs') -Filter 'project-runner*.console.log' |
                    Where-Object { $_.Name -notlike '*.error.console.log' } |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($latestConsole) { Get-Content -LiteralPath $latestConsole.FullName -Tail 3 -Encoding utf8 }
            }
        }
    } catch {
        Write-Output ('状态文件读取未完成，下次刷新重试: ' + $_.Exception.Message)
    }

    if ($ShowGpu) {
        & wsl.exe -d Ubuntu-24.04 -- nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
    }
    Write-Output 'Ctrl+C 只停止监测。HF 基准在完整预热/测量结束时更新日志，长时间无新行并不代表停止。'
    if (-not $Once) { Start-Sleep -Seconds $IntervalSeconds }
} while (-not $Once)
