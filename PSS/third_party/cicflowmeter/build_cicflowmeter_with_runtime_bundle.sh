#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/temp_cicflowmeter_src"
RUNTIME_DIR="${SCRIPT_DIR}/runtime_bundle"
UPSTREAM_URL="https://github.com/ahlashkari/CICFlowMeter.git"
FORCE_DOWNLOAD=0

usage() {
    cat <<'EOF'
Build the official CICFlowMeter distribution used by PSS.

Usage:
  bash build_cicflowmeter_with_runtime_bundle.sh [--download-cfm]

Options:
  --download-cfm, -d  discard the cached upstream checkout and clone it again
  --help, -h          show this help message

Prerequisites: Git, Maven, a Java 8 JDK, and either unzip or the JDK jar tool.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --download-cfm|-d)
            FORCE_DOWNLOAD=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for command_name in git mvn java; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command not found: $command_name" >&2
        exit 1
    fi
done

safe_remove_tree() {
    local target="$1"
    case "$target" in
        "${SCRIPT_DIR}/temp_cicflowmeter_src"|"${SCRIPT_DIR}/runtime_bundle")
            rm -rf -- "$target"
            ;;
        *)
            echo "Refusing to remove unexpected path: $target" >&2
            exit 1
            ;;
    esac
}

if [[ "$FORCE_DOWNLOAD" -eq 1 ]]; then
    safe_remove_tree "$SOURCE_DIR"
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    echo "Cloning CICFlowMeter from ${UPSTREAM_URL}"
    git clone --depth 1 "$UPSTREAM_URL" "$SOURCE_DIR"
else
    echo "Using cached CICFlowMeter checkout at ${SOURCE_DIR}"
fi

JNETPCAP_JAR="${SOURCE_DIR}/jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar"
if [[ ! -f "$JNETPCAP_JAR" ]]; then
    echo "Bundled jnetpcap.jar not found at ${JNETPCAP_JAR}" >&2
    exit 1
fi

echo "Installing the bundled jnetpcap JAR into the local Maven repository"
mvn -q install:install-file \
    -Dfile="$JNETPCAP_JAR" \
    -DgroupId=org.jnetpcap \
    -DartifactId=jnetpcap \
    -Dversion=1.4.1 \
    -Dpackaging=jar

echo "Building the CICFlowMeter application distribution"
(
    cd "$SOURCE_DIR"
    chmod +x gradlew
    ./gradlew clean distZip
)

DIST_ZIP="$(find "${SOURCE_DIR}/build/distributions" -maxdepth 1 -type f -name 'CICFlowMeter-*.zip' -print -quit)"
if [[ -z "$DIST_ZIP" || ! -f "$DIST_ZIP" ]]; then
    echo "CICFlowMeter distribution ZIP was not produced" >&2
    exit 1
fi

safe_remove_tree "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"

if command -v unzip >/dev/null 2>&1; then
    unzip -q "$DIST_ZIP" -d "$RUNTIME_DIR"
elif command -v jar >/dev/null 2>&1; then
    (
        cd "$RUNTIME_DIR"
        jar -xf "$DIST_ZIP"
    )
else
    echo "Neither unzip nor the JDK jar tool is available" >&2
    exit 1
fi

DIST_DIR="$(find "$RUNTIME_DIR" -mindepth 1 -maxdepth 1 -type d -name 'CICFlowMeter*' -print -quit)"
if [[ -z "$DIST_DIR" || ! -d "${DIST_DIR}/lib" || ! -d "${DIST_DIR}/lib/native" ]]; then
    echo "The extracted distribution does not contain the expected lib/native layout" >&2
    exit 1
fi

git -C "$SOURCE_DIR" rev-parse HEAD > "${RUNTIME_DIR}/.source_commit"

echo "CICFlowMeter runtime bundle created at: ${DIST_DIR}"
echo "Upstream commit: $(cat "${RUNTIME_DIR}/.source_commit")"
