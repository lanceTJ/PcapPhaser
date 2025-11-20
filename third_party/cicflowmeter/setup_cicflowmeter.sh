#!/bin/bash
# PcapPhaser CICFlowMeter Official Distribution Build Script (Fully Offline-Capable Version)
# Automatically downloads gradle-4.10.2-all.zip if network available
# If download fails, prompts user to manually place it (exactly what you did)
# Then modifies gradle-wrapper.properties to local file path -> completely offline build
# Generates pure CLI bin/cfm distribution - NO GUI ever
# Supports --download_cfm 1 to force re-download source
# Date: 2025-11-20

set -e

FORCE_DOWNLOAD=0
for arg in "$@"; do
    case $arg in
        --download_cfm=1|-d=1|--download_cfm)
            FORCE_DOWNLOAD=1
            ;;
    esac
done

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$TARGET_DIR/temp_cicflowmeter_src"
DIST_DIR="$TARGET_DIR/cfm_dist"
GRADLE_ZIP="$TEMP_DIR/gradle-4.10.2-all.zip"
PROPERTIES_FILE="$TEMP_DIR/gradle/wrapper/gradle-wrapper.properties"

echo "=== PcapPhaser CICFlowMeter Build ==="

# Reuse cache or fresh clone
if [ "$FORCE_DOWNLOAD" -eq 0 ] && [ -d "$TEMP_DIR" ]; then
    echo "Using cached source code"
else
    echo "Cloning official source code..."
    rm -rf "$TEMP_DIR"
    git clone https://github.com/ahlashkari/CICFlowMeter.git "$TEMP_DIR"
fi

cd "$TEMP_DIR"

# Step: Ensure gradle distribution zip exists locally (offline-first)
if [ ! -f "$GRADLE_ZIP" ]; then
    echo "Attempting to download gradle-4.10.2-all.zip (required by old wrapper)..."
    if curl -L -o "$GRADLE_ZIP" https://services.gradle.org/distributions/gradle-4.10.2-all.zip; then
        echo "Auto download succeeded"
    else
        echo "Auto download failed (common in restricted networks)"
        echo "Please manually download gradle-4.10.2-all.zip from:"
        echo "    https://services.gradle.org/distributions/gradle-4.10.2-all.zip"
        echo "and place it at: $GRADLE_ZIP"
        echo "Press Enter after placing the file..."
        read -r
        if [ ! -f "$GRADLE_ZIP" ]; then
            echo "File still not found, exiting"
            exit 1
        fi
        echo "Manual file detected, continuing"
    fi
else
    echo "Local gradle-4.10.2-all.zip found, using offline mode"
fi

# Step: Force wrapper to use local zip (completely offline build)
echo "Patching gradle-wrapper.properties to use local distribution..."
sed -i "s|distributionUrl=.*|distributionUrl=file:\\\\/$GRADLE_ZIP|g" "$PROPERTIES_FILE"

# Force Java 8 for build (your environment has it)
JAVA8_HOME=$(ls -d /usr/lib/jvm/java-8-openjdk* | head -1)  # auto find
if [ -z "$JAVA8_HOME" ]; then
    echo "Java 8 not found, please check installation"
    exit 1
fi
export JAVA_HOME="$JAVA8_HOME"
export PATH="$JAVA_HOME/bin:$PATH"
echo "Using Java 8: $(java -version 2>&1 | head -1)"

echo "Building distribution using official gradlew"
chmod +x gradlew
./gradlew distZip --quiet

echo "Extracting distribution package using jar command"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"
cd "$DIST_DIR"
"$JAVA_HOME/bin/jar" -xf $TEMP_DIR/build/distributions/CICFlowMeter-*.zip

chmod +x CICFlowMeter-*/bin/cfm* 2>/dev/null || true

echo "=== Build completed successfully (fully offline capable)! ==="
echo "Distribution: $DIST_DIR/CICFlowMeter-4.0"
echo "CLI tool: $DIST_DIR/CICFlowMeter-4.0/bin/cfm"
echo "All future runs will be completely offline and complete in seconds"