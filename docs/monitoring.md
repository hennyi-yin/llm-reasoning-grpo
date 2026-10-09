# 查看后台进度

在 PowerShell 中切换到项目目录，再运行只读监测脚本。它每 10 秒刷新后台进程、当前阶段和最近日志；不会启动、停止或恢复实验。

```powershell
Set-Location -LiteralPath 'C:\Users\25696\Downloads\LLM Post Training&Inference\llm-reasoning-grpo'
pwsh -NoProfile -File .\scripts\watch_project.ps1
```

只打印一次快照：

```powershell
pwsh -NoProfile -File .\scripts\watch_project.ps1 -Once
```

同时显示显卡利用率、显存和温度：

```powershell
pwsh -NoProfile -File .\scripts\watch_project.ps1 -ShowGpu
```

按 `Ctrl+C` 只会退出监测，隐藏的实验进程仍会继续。HF 基准日志只在完整预热和每次正式测量开始/结束时记录进度；batch=1 每次可能超过一小时。不要把日志没有新行当成进程已停止。

查看完整阶段及训练记录（不加载模型或数据集）：

```powershell
wsl -d Ubuntu-24.04 -- /home/hengyi/.venvs/llm-reasoning-grpo/bin/python -m src.progress
```

跟随某个阶段的原始日志，例如当前的 HF 基准：

```powershell
Get-Content -LiteralPath .\logs\pipeline\offline_hf.log -Tail 10 -Wait
```

后台状态文件是 `logs/windows_runner.json`，流水线阶段文件是 `results/pipeline_state.json`。后台状态变为 `complete` 表示流水线和本地发布打包均已成功；`failed` 时保留错误记录和原始日志以供定位。`running` 应同时有存活的 Windows 进程，状态文件本身不能证明进程仍存活。
