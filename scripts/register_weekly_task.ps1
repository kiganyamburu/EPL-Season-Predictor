param(
    [string]$TaskName = "EPL-Season-Predictor-WeeklyRefresh",
    [ValidateSet("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")]
    [string]$DayOfWeek = "Tuesday",
    [string]$Time = "09:00",
    [switch]$WithSweeps,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Unregistered scheduled task '$TaskName'."
    exit 0
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$JobScript = Join-Path $RepoRoot "scripts\run_weekly_job.ps1"
if (-not (Test-Path $JobScript)) {
    throw "Missing runner script: $JobScript"
}

$ArgList = "-NoProfile -ExecutionPolicy Bypass -File `"$JobScript`""
if ($WithSweeps) {
    $ArgList += " -WithSweeps"
}

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ArgList -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DayOfWeek -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
Write-Host "Registered Windows Scheduled Task '$TaskName' ($DayOfWeek at $Time) -> $JobScript"

