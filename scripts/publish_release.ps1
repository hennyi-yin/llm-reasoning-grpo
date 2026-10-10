param(
    [Parameter(Mandatory)][string]$GitHubCli,
    [Parameter(Mandatory)][string]$Repository,
    [string]$Tag = 'v1.0.0'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$runnerPath = Join-Path $root 'logs\release_runner.json'
$runState = [ordered]@{status='running'; pid=$PID; repository=$Repository; tag=$Tag; started_at_utc=[DateTime]::UtcNow.ToString('o')}
Add-Type -Namespace LLMRelease -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
function Save-State { $runState | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $runnerPath -Encoding utf8 }
function Gh-Json([string]$Endpoint) {
    $response = & $GitHubCli api $Endpoint
    if ($LASTEXITCODE -ne 0) { throw "GitHub API failed: $Endpoint" }
    return ($response | ConvertFrom-Json)
}
function Verify-Asset($Asset, $Expected) {
    if ($Asset.state -ne 'uploaded' -or $Asset.size -ne $Expected.bytes -or $Asset.digest -ne ('sha256:'+$Expected.sha256)) {
        throw "Remote asset size/state/SHA256 mismatch: $($Expected.name); preserve it for inspection"
    }
}
try {
    [LLMRelease.Awake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    Save-State
    $summary = Get-Content results\summary.json -Raw | ConvertFrom-Json
    $proof = Get-Content results\compatibility\final_cpu_table_reproduction.json -Raw | ConvertFrom-Json
    $package = Get-Content release_assets\manifest.json -Raw | ConvertFrom-Json
    $summaryHash = (Get-FileHash results\summary.json -Algorithm SHA256).Hash.ToLower()
    if ($summary.status -ne 'measured' -or $proof.status -ne 'verified' -or $package.summary_sha256 -ne $summaryHash) {
        throw 'Finish and verify the complete experiment before publishing'
    }
    foreach ($property in $proof.source_hashes.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLower() -ne $property.Value) {
            throw "CPU proof source changed: $($property.Name)"
        }
    }
    $localCommit = (& git rev-parse $Tag).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Create and push the verified release tag first' }
    $remoteRef = Gh-Json "repos/$Repository/git/ref/tags/$Tag"
    if ($remoteRef.object.type -ne 'commit' -or $remoteRef.object.sha -ne $localCommit) {
        throw 'Remote release tag differs from the local commit'
    }
    $runs = Gh-Json "repos/$Repository/actions/runs?head_sha=$localCommit"
    if (-not @($runs.workflow_runs | Where-Object { $_.name -eq 'CPU protocol tests' -and $_.conclusion -eq 'success' }).Count) {
        throw 'Wait for the release commit CPU workflow to pass'
    }
    $files = @($package.artifacts | ForEach-Object { $_.upload_files } | ForEach-Object { $_ })
    $manifestPath = Join-Path $root 'release_assets\manifest.json'
    $files += [pscustomobject]@{name='manifest.json'; bytes=(Get-Item $manifestPath).Length; sha256=(Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLower()}
    $runState.total_assets = $files.Count
    $runState.completed_assets = 0
    Save-State
    foreach ($file in $files) {
        $path = Join-Path (Join-Path $root 'release_assets') $file.name
        if ([IO.Path]::GetFileName($file.name) -ne $file.name -or (Get-Item -LiteralPath $path).Length -ne $file.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $file.sha256) {
            throw "Local release asset differs from manifest: $($file.name)"
        }
    }
    $releases = @(Gh-Json "repos/$Repository/releases?per_page=100")
    $release = @($releases | Where-Object tag_name -EQ $Tag)
    if ($release.Count -gt 1) { throw 'Ambiguous existing release' }
    if (-not $release.Count) {
        & $GitHubCli release create $Tag --repo $Repository --verify-tag --draft --title 'Frozen post-training and serving experiments on RTX 5070' --notes-file docs\release_notes.md
        if ($LASTEXITCODE -ne 0) { throw 'Could not create draft release' }
        $release = @(Gh-Json "repos/$Repository/releases/tags/$Tag")
    }
    $release = $release[0]
    $receipts = @()
    foreach ($file in $files) {
        $runState.asset = $file.name
        Save-State
        $assets = @(Gh-Json "repos/$Repository/releases/$($release.id)/assets?per_page=100")
        $existing = @($assets | Where-Object name -EQ $file.name)
        if ($existing.Count -gt 1) { throw "Duplicate remote asset: $($file.name)" }
        if (-not $existing.Count) {
            if (-not $release.draft) { throw 'Do not add missing files to an already published release' }
            Write-Output "Uploading $($file.name) ($($file.bytes) bytes)"
            & $GitHubCli release upload $Tag (Join-Path (Join-Path $root 'release_assets') $file.name) --repo $Repository
            if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($file.name); rerun to verify and continue completed files" }
            $assets = @(Gh-Json "repos/$Repository/releases/$($release.id)/assets?per_page=100")
            $existing = @($assets | Where-Object name -EQ $file.name)
        }
        if ($existing.Count -ne 1) { throw "Missing uploaded asset: $($file.name)" }
        Verify-Asset $existing[0] $file
        $receipts += [ordered]@{name=$file.name; id=$existing[0].id; bytes=$file.bytes; sha256=$file.sha256; remote_digest=$existing[0].digest; url=$existing[0].browser_download_url}
        $runState.completed_assets = $receipts.Count
        Save-State
    }
    if ($release.draft) {
        & $GitHubCli release edit $Tag --repo $Repository --draft=false --latest
        if ($LASTEXITCODE -ne 0) { throw 'Publishing the verified draft release failed' }
    }
    $published = Gh-Json "repos/$Repository/releases/tags/$Tag"
    if ($published.draft -or @($published.assets).Count -ne $files.Count) { throw 'Release publication verification failed' }
    $evidence = [ordered]@{status='published'; uploaded=$true; repository=$Repository; url=$published.html_url; tag=$Tag; commit=$localCommit; checked_at_utc=[DateTime]::UtcNow.ToString('o'); summary_sha256=$summaryHash; assets=$receipts}
    $evidence | ConvertTo-Json -Depth 10 | Set-Content results\publication.json -Encoding utf8
    $runState.status = 'complete'
    Write-Output $published.html_url
} catch {
    $runState.status = 'failed'
    $runState.error = $_.Exception.Message
    throw
} finally {
    [LLMRelease.Awake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    $runState.ended_at_utc = [DateTime]::UtcNow.ToString('o')
    Save-State
}
