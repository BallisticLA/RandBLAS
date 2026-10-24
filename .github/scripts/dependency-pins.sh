#!/usr/bin/env bash
# Reads, checks and applies installers/dependency-pins.env, the file naming
# the BLAS++ and LAPACK++ commits every build lane uses. The installers parse
# it themselves; this script serves the CI workflows and composite actions.
#
#   dependency-pins.sh export <pins-file>
#       Check the file, then append its KEY=value lines to $GITHUB_OUTPUT
#       (stdout outside GitHub Actions), for fetch steps and cache keys.
#   dependency-pins.sh fetch <url> <sha> <dest>
#       Fetch exactly that commit into a fresh <dest>; fail unless HEAD is it.
#   dependency-pins.sh lint <pins-file>
#       Each *_REF must be fetchable from its *_URL, and a non-empty *_TAG
#       must point at its *_REF there.
#   dependency-pins.sh match <pins-file> <other-pins-file>
#       Both files must pin the same values (their comments may differ).
set -euo pipefail

die() { printf '::error::%s\n' "$*" >&2; exit 1; }

KEYS=(BLASPP_URL BLASPP_REF BLASPP_TAG LAPACKPP_URL LAPACKPP_REF LAPACKPP_TAG)

# Prints the file's KEY=value lines with carriage returns, comments and blank
# lines removed, after checking that they are exactly the expected keys with
# well-formed values. The same rules are in windows/dependency-pins.ps1.
read_pins() {  # <pins-file>
    local file="$1" line key value seen=" "
    [[ -f "$file" ]] || die "$file not found."
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" == \#* ]] && continue
        [[ "$line" =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]] || die "$file: not a KEY=value line: $line"
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        [[ " ${KEYS[*]} " == *" $key "* ]] || die "$file: unknown key $key."
        [[ "$seen" != *" $key "* ]] || die "$file: $key is set twice."
        [[ "$value" != *[[:space:]\"\']* ]] || die "$file: $key has a space or a quote in its value."
        case "$key" in
            *_REF) [[ "$value" =~ ^[0-9a-f]{40}$ ]] \
                       || die "$file: $key must be a full 40-character commit hash, not '$value'." ;;
            *_URL) [[ -n "$value" ]] || die "$file: $key is empty." ;;
        esac
        seen+="$key "
        printf '%s=%s\n' "$key" "$value"
    done < "$file"
    for key in "${KEYS[@]}"; do
        [[ "$seen" == *" $key "* ]] || die "$file: $key is missing."
    done
}

# Sets the six pin variables in the calling shell.
load_pins() {  # <pins-file>
    local pins key value
    pins="$(read_pins "$1")"
    while IFS='=' read -r key value; do
        printf -v "$key" '%s' "$value"
    done <<< "$pins"
}

fetch_commit() {  # <url> <sha> <dest>
    local url="$1" sha="$2" dest="$3" head
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || die "fetch wants a full 40-character commit hash, not '$sha'."
    rm -rf "$dest"
    git init --quiet "$dest"
    git -C "$dest" fetch --quiet --depth 1 "$url" "$sha" \
        || die "$sha could not be fetched from $url."
    git -C "$dest" checkout --quiet --detach FETCH_HEAD
    head="$(git -C "$dest" rev-parse HEAD)"
    [[ "$head" == "$sha" ]] || die "$dest is at $head after fetching $sha from $url."
}

cmd_export() {  # <pins-file>
    local pins
    pins="$(read_pins "$1")"
    printf 'Dependency pins from %s:\n%s\n' "$1" "$pins"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        printf '%s\n' "$pins" >> "$GITHUB_OUTPUT"
    fi
}

cmd_lint() {  # <pins-file>
    local BLASPP_URL BLASPP_REF BLASPP_TAG LAPACKPP_URL LAPACKPP_REF LAPACKPP_TAG
    local name var url ref tag listing tagged
    load_pins "$1"
    LINT_SCRATCH="$(mktemp -d)"
    trap 'rm -rf "$LINT_SCRATCH"' EXIT
    for name in BLASPP LAPACKPP; do
        var="${name}_URL"; url="${!var}"
        var="${name}_REF"; ref="${!var}"
        var="${name}_TAG"; tag="${!var}"
        # `git ls-remote <url> <sha>` matches ref NAMES, so it prints nothing
        # for a commit hash; fetching the commit is what proves it exists.
        fetch_commit "$url" "$ref" "$LINT_SCRATCH/$name"
        echo "OK: ${name}_REF $ref exists at $url"
        if [[ -n "$tag" ]]; then
            listing="$(git ls-remote "$url" "refs/tags/$tag" "refs/tags/$tag^{}")" \
                || die "git ls-remote $url failed."
            # An annotated tag lists its peeled commit as <tag>^{}; a
            # lightweight one lists the commit directly.
            tagged="$(awk -v t="refs/tags/$tag^{}" '$2 == t { print $1 }' <<< "$listing")"
            [[ -n "$tagged" ]] || tagged="$(awk -v t="refs/tags/$tag" '$2 == t { print $1 }' <<< "$listing")"
            [[ -n "$tagged" ]] || die "${name}_TAG $tag does not exist at $url."
            [[ "$tagged" == "$ref" ]] \
                || die "${name}_TAG $tag points at $tagged at $url, but ${name}_REF is $ref."
            echo "OK: ${name}_TAG $tag points at ${name}_REF"
        else
            echo "note: ${name}_TAG is empty, so only the commit was checked."
        fi
    done
}

cmd_match() {  # <pins-file> <other-pins-file>
    local ours theirs
    [[ -f "$2" ]] || die "$2 not found: a RandBLAS submodule that predates the pins file cannot be checked against."
    ours="$(read_pins "$1" | LC_ALL=C sort)"
    theirs="$(read_pins "$2" | LC_ALL=C sort)"
    if [[ "$ours" != "$theirs" ]]; then
        printf '::error::%s and %s pin different BLAS++/LAPACK++ (< %s, > %s):\n' "$1" "$2" "$1" "$2" >&2
        diff <(printf '%s\n' "$ours") <(printf '%s\n' "$theirs") >&2 || true
        exit 1
    fi
    echo "OK: $1 and $2 pin the same BLAS++/LAPACK++."
}

[[ $# -ge 1 ]] || die "usage: $0 export|fetch|lint|match ..."
command="$1"; shift
case "$command" in
    export) [[ $# -eq 1 ]] || die "usage: $0 export <pins-file>"; cmd_export "$@" ;;
    fetch)  [[ $# -eq 3 ]] || die "usage: $0 fetch <url> <sha> <dest>"; fetch_commit "$@" ;;
    lint)   [[ $# -eq 1 ]] || die "usage: $0 lint <pins-file>"; cmd_lint "$@" ;;
    match)  [[ $# -eq 2 ]] || die "usage: $0 match <pins-file> <other-pins-file>"; cmd_match "$@" ;;
    *)      die "unknown command '$command'; expected export, fetch, lint or match." ;;
esac
