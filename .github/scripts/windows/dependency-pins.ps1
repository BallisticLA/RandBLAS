# Reads installers/dependency-pins.env, the file naming the BLAS++ and LAPACK++
# commits every build lane uses. Dot-sourced by the Windows provisioner
# (setup.ps1) and by the CI steps that turn the pins into cache keys. The rules
# match .github/scripts/dependency-pins.sh: KEY=value lines, '#' comment lines,
# exactly the six expected keys, no spaces or quotes in a value, and each *_REF
# a full 40-character commit hash. LF and CRLF line endings both parse, so a
# checkout that ignored .gitattributes still works.

function Read-DependencyPins {
    param([Parameter(Mandatory = $true)][string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Path. This clone looks incomplete."
    }
    $expected = @("BLASPP_URL", "BLASPP_REF", "BLASPP_TAG",
        "LAPACKPP_URL", "LAPACKPP_REF", "LAPACKPP_TAG")
    $pins = [ordered]@{}
    foreach ($raw in @(Get-Content -LiteralPath $Path)) {
        $line = $raw.TrimEnd("`r")
        if ($line -eq "" -or $line.StartsWith("#")) { continue }
        if ($line -cnotmatch '^([A-Z][A-Z0-9_]*)=(.*)$') {
            throw "${Path}: not a KEY=value line: $line"
        }
        $key = $Matches[1]
        $value = $Matches[2]
        if ($expected -cnotcontains $key) { throw "${Path}: unknown key $key." }
        if ($pins.Contains($key)) { throw "${Path}: $key is set twice." }
        if ($value -match '[\s"'']') { throw "${Path}: $key has a space or a quote in its value." }
        if ($key.EndsWith("_REF") -and $value -cnotmatch '^[0-9a-f]{40}$') {
            throw "${Path}: $key must be a full 40-character commit hash, not '$value'."
        }
        if ($key.EndsWith("_URL") -and $value -eq "") { throw "${Path}: $key is empty." }
        $pins[$key] = $value
    }
    foreach ($key in $expected) {
        if (-not $pins.Contains($key)) { throw "${Path}: $key is missing." }
    }
    return $pins
}

function Export-DependencyPins {
    # For CI: logs each pin and appends it to $env:GITHUB_OUTPUT as KEY=value,
    # where cache keys pick the hashes up as step outputs.
    param([Parameter(Mandatory = $true)][string] $Path)
    $pins = Read-DependencyPins $Path
    foreach ($entry in $pins.GetEnumerator()) {
        Write-Host "$($entry.Key)=$($entry.Value)"
        if ($env:GITHUB_OUTPUT) {
            Add-Content -Path $env:GITHUB_OUTPUT -Value "$($entry.Key)=$($entry.Value)"
        }
    }
}
