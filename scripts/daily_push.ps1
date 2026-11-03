# Commit and push any changes in this project to GitHub (run daily by Windows Task Scheduler).
# Does nothing when there are no changes. Log: %LOCALAPPDATA%\convlstm_milano_daily_push.log
$ErrorActionPreference = "Stop"
$log = Join-Path $env:LOCALAPPDATA "convlstm_milano_daily_push.log"
$project = Split-Path -Parent $PSScriptRoot

function Write-Log([string]$msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -FilePath $log -Append -Encoding utf8
}

try {
    Set-Location $project
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Log "No changes to push."
        exit 0
    }
    $files = (git diff --cached --name-only | Measure-Object -Line).Lines
    git commit -q -m "Daily update $(Get-Date -Format 'yyyy-MM-dd') ($files files changed)"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    git push -q origin HEAD 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "git push failed (commit is saved locally; it will go out with the next push)" }
    Write-Log "Pushed $files changed files."
}
catch {
    Write-Log "ERROR: $_"
    exit 1
}
