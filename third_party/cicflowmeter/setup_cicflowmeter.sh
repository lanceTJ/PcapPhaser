#!/bin/bash
# PcapPhaser CICFlowMeter One-Click Local Build Script (Final Stable Version)
# Uses pure-Java jnetpcap bundled in official repo, no external downloads required
# Supports --download_cfm 1 to force re-download source code (default: cache for fast debugging)
# Tested on: Ubuntu 24.04 / macOS / WSL2 + JDK17 + Maven 3.9
# Date: 2025-11-19

set -e

# Default: do not force re-download
FORCE_DOWNLOAD=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        --download_cfm=1|-d=1|--download_cfm)
            FORCE_DOWNLOAD=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--download_cfm 1 | -d 1]   # Only add this flag to force re-download source"
            ;;
    esac
done

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$TARGET_DIR/temp_cicflowmeter_build"
JAR_FINAL="$TARGET_DIR/cicflowmeter.jar"

echo "=== PcapPhaser CICFlowMeter Smart Build Script  ==="

# Reuse cached source when not forcing download
if [ "$FORCE_DOWNLOAD" -eq 0 ] && [ -d "$TEMP_DIR" ]; then
    echo "Local source code detected, reusing cache"
    echo "To force fresh clone, run: bash $0 --download_cfm 1"
else
    echo "Cleaning and re-cloning official source code..."
    rm -rf "$TEMP_DIR"
    git clone https://github.com/ahlashkari/CICFlowMeter.git "$TEMP_DIR"
fi

cd "$TEMP_DIR"

# Auto-select platform-specific bundled pure-Java jnetpcap.jar
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    JNETPCAP_JAR="jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar"
else
    JNETPCAP_JAR="jnetpcap/win/jnetpcap-1.4.r1425/jnetpcap.jar"
fi

if [ ! -f "$JNETPCAP_JAR" ]; then
    echo "ERROR: Bundled jnetpcap.jar not found, repo may be incomplete"
    exit 1
fi

echo "Installing bundled pure-Java jnetpcap to local Maven repository "
mvn install:install-file \
    -Dfile="$JNETPCAP_JAR" \
    -DgroupId=org.jnetpcap \
    -DartifactId=jnetpcap \
    -Dversion=1.4.r1425 \
    -Dpackaging=jar -q

echo "Forcing dependency update and building project (this may take several minutes)..."
mvn -U -q package -DskipTests

echo "Copying final JAR to project directory"
# Compatible with official actual output filename pattern
if ls target/CICFlowMeter*.jar 1>/dev/null 2>&1; then
    cp target/CICFlowMeter*.jar "$JAR_FINAL"
elif ls target/cicflowmeter*.jar 1>/dev/null 2>&1; then
    cp target/cicflowmeters*.jar "$JAR_FINAL"
else
    echo "ERROR: No JAR generated in target/. Maven build may have failed"
    ls -la target/
    exit 1
fi

echo "=== Build completed successfully! ==="
echo "JAR location: $JAR_FINAL (~15 MB)"
ls -lh "$JAR_FINAL"
echo "You can now run CFMRunner.py directly"
echo "Tip: When network is unstable, just run this script without arguments → completes in seconds"