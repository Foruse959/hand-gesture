param(
  [string]$Config = "$PSScriptRoot\config.json",
  [string]$ProfileId = "",
  [ValidateSet("global", "browser", "presentation")]
  [string]$Context = "global",
  [switch]$Headless,
  [switch]$Background
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

$pythonCandidates = @(
  "$PSScriptRoot\.venv\Scripts\python.exe",
  "$repoRoot\.venv\Scripts\python.exe"
)

$pythonExe = "python"
foreach ($candidate in $pythonCandidates) {
  if (Test-Path $candidate) {
    $pythonExe = $candidate
    break
  }
}

$args = @(
  "$PSScriptRoot\agent.py",
  "--config", $Config,
  "--context", $Context
)

if ($ProfileId) {
  $args += @("--profile-id", $ProfileId)
}
if ($Headless) {
  $args += "--headless"
}

if ($Background) {
  Start-Process -FilePath $pythonExe -ArgumentList $args -WindowStyle Minimized | Out-Null
  Write-Host "Desktop agent launched in background."
  exit 0
}

& $pythonExe @args
