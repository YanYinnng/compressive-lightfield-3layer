param([Parameter(ValueFromRemainingArguments=$true)][string[]]$LabArguments)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Path $PSScriptRoot -Parent)
$taskPython = 'D:\anaconda\envs\test\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = (Get-Command python).Source
}
& $taskPython -m lightfield_lab.benchmark @LabArguments
exit $LASTEXITCODE
