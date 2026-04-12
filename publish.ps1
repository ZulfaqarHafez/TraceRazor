# TraceRazor — publish all Python packages to PyPI
# Run from the repo root: .\publish.ps1
# Requires: pip install build twine

$ErrorActionPreference = "Stop"

$packages = @("tracerazor", "crewai", "openai-agents", "langgraph")

foreach ($pkg in $packages) {
    Write-Host "`n==> Building $pkg" -ForegroundColor Cyan
    Push-Location "integrations\$pkg"

    # Clean previous build artifacts
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

    python -m build
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed for $pkg"; Pop-Location; exit 1 }

    Write-Host "==> Uploading $pkg" -ForegroundColor Green
    twine upload dist/*
    if ($LASTEXITCODE -ne 0) { Write-Error "Upload failed for $pkg"; Pop-Location; exit 1 }

    Pop-Location
}

Write-Host "`nAll packages published successfully." -ForegroundColor Green