param(
    [ValidateSet('cpu','no_contact','gpu','gpu_refined')][string]$Mode='cpu'
)
$ErrorActionPreference='Stop'
# Match the launch environment of the existing E:\IsaacLab installation.
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:OMNI_USER_HOME='E:\IsaacLab\userdata'
$env:OV_USER_HOME='E:\IsaacLab\userdata'
$env:TEMP='E:\IsaacLab\tmp'
$env:TMP='E:\IsaacLab\tmp'
$env:PYTHONIOENCODING='utf-8'
$cubeArgs=@('--tag','validated_cpu')
if ($Mode -eq 'no_contact') { $cubeArgs=@('--no-contact','--tag','no_contact') }
if ($Mode -eq 'gpu') { $cubeArgs=@('--gpu','--tag','validated_gpu') }
if ($Mode -eq 'gpu_refined') { $cubeArgs=@('--gpu','--hz','960','--iterations','128','--steps','5760','--tag','gpu_matched_timing') }
& 'E:\IsaacLab\env\python.exe' -u (Join-Path $PSScriptRoot 'contact_test.py') @cubeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$cubeTag=$cubeArgs[$cubeArgs.IndexOf('--tag')+1]
$cubeResult=Get-Content -Raw (Join-Path $PSScriptRoot ($cubeTag+'_result.json')) | ConvertFrom-Json
if ($Mode -eq 'no_contact') {
    foreach ($phase in $cubeResult.results) {
        if ([Math]::Abs($phase.mean_angle_deg) -gt 1 -or $phase.fixture_contact_headers -ne 0) { exit 1 }
    }
} elseif (-not $cubeResult.passed) { exit 1 }
