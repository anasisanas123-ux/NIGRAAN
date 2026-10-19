$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$rootPrefix = $projectRoot.TrimEnd('\') + '\'

function Move-ProjectItem([string]$RelativeSource, [string]$RelativeDestination) {
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $RelativeSource))
    $destinationPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $RelativeDestination))
    foreach ($targetPath in @($sourcePath, $destinationPath)) {
        if (-not $targetPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing to move an item outside the project.'
        }
    }
    if (-not (Test-Path -LiteralPath $sourcePath)) { return }
    if (Test-Path -LiteralPath $destinationPath) { throw "Archive destination already exists: $RelativeDestination" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null
    Move-Item -LiteralPath $sourcePath -Destination $destinationPath
}

$legacyPatterns = @('multi_camera_before_*.py', 'install_*.py', 'fix_*.py', 'switch_to_botsort.py')
$legacyFiles = Get-ChildItem -LiteralPath $projectRoot -File | Where-Object {
    $fileName = $_.Name
    @($legacyPatterns | Where-Object { $fileName -like $_ }).Count -gt 0
}
foreach ($file in $legacyFiles) { Move-ProjectItem $file.Name (Join-Path 'archive\legacy_patches' $file.Name) }
foreach ($file in (Get-ChildItem -LiteralPath $projectRoot -File -Filter 'test_*.py')) {
    Move-ProjectItem $file.Name (Join-Path 'archive\manual_checks' $file.Name)
}
Move-ProjectItem 'debug_ghost.py' 'archive\manual_checks\debug_ghost.py'
Move-ProjectItem 'backup_before_roboflow_20260828_155506' 'archive\backup_before_roboflow_20260828_155506'
Move-ProjectItem 'repo_tree.txt' 'archive\reference\repo_tree.txt'
Move-ProjectItem 'snippet.txt' 'archive\reference\snippet.txt'
foreach ($file in (Get-ChildItem -LiteralPath $projectRoot -File -Filter '*.md' | Where-Object Name -ne 'README.md')) {
    Move-ProjectItem $file.Name (Join-Path 'docs' $file.Name)
}
foreach ($name in @('_refactor_runtime_once.py', '_refactor_feed_once.py', '_compact_sources_once.py')) {
    Move-ProjectItem (Join-Path 'scripts' $name) (Join-Path 'archive\refactor_tools' $name)
}
Write-Output 'Archived historical patches and manual checks; guides are now in docs/.'
