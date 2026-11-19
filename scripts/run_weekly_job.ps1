param(
    [switch]$WithSweeps,
    [switch]$NoSync,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot

try {
    $LogDir = Join-Path $RepoRoot "output\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $LogFile = Join-Path $LogDir "weekly_refresh_$Timestamp.log"

    $RefreshArgs = @("scripts\weekly_refresh.py")
    if ($WithSweeps) { $RefreshArgs += "--with-sweeps" }
    if ($NoSync) { $RefreshArgs += "--no-sync" }

    "[$(Get-Date -Format o)] Starting weekly_refresh.py $($RefreshArgs -join ' ')" | Tee-Object -FilePath $LogFile -Append
    & $PythonExe @RefreshArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "weekly_refresh.py exited with code $LASTEXITCODE"
    }

    "[$(Get-Date -Format o)] Running pytest verification..." | Tee-Object -FilePath $LogFile -Append
    & $PythonExe -m pytest -v 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "pytest exited with code $LASTEXITCODE"
    }

    "[$(Get-Date -Format o)] Weekly refresh and verification completed cleanly." | Tee-Object -FilePath $LogFile -Append
}
finally {
    Pop-Location
}

