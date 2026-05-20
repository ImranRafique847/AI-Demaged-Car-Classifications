# Load .env file and set environment variables
$envFile = Join-Path (Split-Path $MyInvocation.MyCommand.Path) ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*[^#=]+=' -and -not $_.StartsWith('#')) {
            $name, $value = $_.Split('=')
            $name = $name.Trim()
            $value = $value.Trim()
            Set-Item -Path "Env:$name" -Value $value -Force
        }
    }
    Write-Host ".env loaded successfully" -ForegroundColor Green
} else {
    Write-Host ".env file not found at $envFile" -ForegroundColor Red
}
