#!/bin/bash
# Automatic download script for CICFlowMeter (MIT License)
# Author: PcapPhaser Team
# Date: 2025

set -e

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_URL="https://github.com/ahlashkari/CICFlowMeter/releases/latest/download/CICFlowMeter.jar"
LICENSE_URL="https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/LICENSE"

echo "Downloading CICFlowMeter.jar (latest release)..."
curl -L -o "$TARGET_DIR/cicflowmeter.jar" "$JAR_URL"

echo "Downloading official LICENSE..."
curl -L -o "$TARGET_DIR/LICENSE_CICFlowMeter.txt" "$LICENSE_URL"

echo "CICFlowMeter setup completed!"
echo "Location: $TARGET_DIR/cicflowmeter.jar"
echo "License : $TARGET_DIR/LICENSE_CICFlowMeter.txt"
echo "You can now run CFMRunner directly."