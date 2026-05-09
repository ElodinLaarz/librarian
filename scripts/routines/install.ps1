# Install routined as a recurring task on Windows via Task Scheduler.
# - Creates a logon trigger that starts the daemon when you sign in.
# - Adds a 5-minute watchdog trigger so it self-heals after a crash.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall

param(
    [switch]$Uninstall,
    [string]$Python = "python",
    [string]$Config = "$PSScriptRoot\config.yaml"
)

$ErrorActionPreference = "Stop"
$TaskName = "routined"
$LogDir = Join-Path $env:USERPROFILE ".routined"
$Log = Join-Path $LogDir "daemon.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if ($Uninstall) {
    Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false
    Write-Host "uninstalled $TaskName"
    return
}

if (-not (Test-Path $Config)) {
    throw "config not found at $Config — copy config.example.yaml to config.yaml first."
}

$script = Join-Path $PSScriptRoot "routined.py"
$cmd = "`"$Python`" `"$script`" --config `"$Config`""

# Watchdog: start only if not already running, redirect output to log
$watchdog = @"
`$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { `$_.CommandLine -like '*routined.py*' };
if (-not `$running) {
    Start-Process -WindowStyle Hidden -FilePath '$Python' `
        -ArgumentList @('"$script"','--config','"$Config"') `
        -RedirectStandardOutput '$Log' -RedirectStandardError '$Log';
}
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watchdog))
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -EncodedCommand $encoded"

$logon = New-ScheduledTaskTrigger -AtLogOn
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration ([TimeSpan]::FromDays(3650))

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action `
    -Trigger @($logon, $repeat) `
    -Settings $settings `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "installed scheduled task '$TaskName'. logs: $Log"
