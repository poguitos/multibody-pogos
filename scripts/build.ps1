<#
.SYNOPSIS
    Configure, build and test the project with the MSVC environment loaded.

.DESCRIPTION
    A plain PowerShell window does not know where the compiler is. This script
    loads the Visual Studio build environment and then drives CMake through
    the presets in CMakePresets.json.

    The number of compilers that run at once is limited by the build system
    itself (MBD_COMPILE_JOBS, default 1), not by this script. See "Build
    limits" in README.md before raising it.

.EXAMPLE
    scripts\build.ps1
    Build everything.

.EXAMPLE
    scripts\build.ps1 -Target test_model -Test
    Build one test executable, then run its tests.

.EXAMPLE
    scripts\build.ps1 -Test -Filter "Drivetrain"
    Build everything, then run the tests whose name contains "Drivetrain".
#>
param(
    [string[]]$Target = @(),
    [switch]$Test,
    [string]$Filter = "",
    [switch]$Reconfigure,
    [ValidateSet("dev", "ci")][string]$Preset = "dev"
)

$ErrorActionPreference = "Stop"
# Accept both "-Target a,b" typed in PowerShell and the single string "a,b"
# that arrives when the script is started with powershell -File.
$Target = @($Target | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
$buildDir = if ($Preset -eq "ci") { Join-Path $root "build-ci" } else { Join-Path $root "build" }

function Import-MsvcEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio with the C++ workload."
    }
    $vsPath = & $vswhere -latest -prerelease -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsPath) {
        throw "No Visual Studio installation with the C++ tools was found."
    }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

    # Run vcvars64 in cmd, then copy the resulting environment into this session.
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Loading the MSVC environment from '$vcvars' failed."
    }
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE."
    }
}

Import-MsvcEnvironment

# Ninja learns which headers a file includes from notes the compiler prints.
# On a non-English Visual Studio those notes are translated, and Ninja
# recognises them by a prefix that CMake records, byte for byte, when it
# configures. If the build later runs under a different console code page the
# bytes no longer match, the notes are not recognised, and header changes stop
# triggering rebuilds without any error. So the code page is fixed here, and a
# build directory configured under another one is configured again.
$codePage = 65001
& chcp.com $codePage | Out-Null
$stamp = Join-Path $buildDir ".mbd_codepage"
$configuredCodePage = if (Test-Path $stamp) { (Get-Content $stamp -TotalCount 1).Trim() } else { "" }

Push-Location $root
try {
    $needsConfigure = $Reconfigure `
        -or -not (Test-Path (Join-Path $buildDir "build.ninja")) `
        -or ($configuredCodePage -ne "$codePage")
    if ($needsConfigure) {
        Invoke-Checked cmake @("--preset", $Preset)
        Set-Content -Path $stamp -Value $codePage -Encoding ascii
    }

    $buildArgs = @("--build", "--preset", $Preset)
    if ($Target.Count -gt 0) { $buildArgs += @("--target") + $Target }
    Invoke-Checked cmake $buildArgs

    if ($Test) {
        $testArgs = @("--preset", $Preset)
        if ($Filter) { $testArgs += @("-R", $Filter) }
        Invoke-Checked ctest $testArgs
    }
}
finally {
    Pop-Location
}
