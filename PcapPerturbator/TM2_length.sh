#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate
export MPLBACKEND="${MPLBACKEND:-Agg}"

if grep -q "__REPO_ROOT__" plans/length.json; then
  sed -i "s|__REPO_ROOT__|${REPO_ROOT}|g" plans/length.json
fi

mkdir -p data/outroot vis

pcapperturbator \
  --in-root demo_inputs \
  --out-root data/outroot/tm2_length \
  --backend threads \
  --workers 1 \
  --seed 42 \
  --plan plans/length.json \
  --verbose

python scripts/length_cmp.py \
  --benign demo_inputs/demo/benign_source.pcap \
  --before demo_inputs/demo/cap_attack.pcap \
  --after data/outroot/tm2_length/demo/cap_attack.pcap \
  --out-png vis/TM2_length_hist.png \
  --out-json vis/TM2_length_stats.json
