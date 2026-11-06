# 本机一键检查。uv 的全局缓存在本机 sandbox 下不可写,故重定向到仓库内。
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    $env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"

    Write-Host "== ruff ==" -ForegroundColor Cyan
    & .\.venv\Scripts\ruff.exe check . --cache-dir .ruff-cache
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "== mypy ==" -ForegroundColor Cyan
    & .\.venv\Scripts\mypy.exe src/phaseweaver
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "== pytest ==" -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pytest -q
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
