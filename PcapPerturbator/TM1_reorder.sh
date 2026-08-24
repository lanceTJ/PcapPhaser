#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

mkdir -p data/outroot vis

pcapperturbator \
  --in-root data/inroot \
  --out-root data/outroot/tm1_reorder \
  --backend threads \
  --workers 1 \
  --seed 42 \
  --plan plans/reorder.json \
  --verbose

python scripts/reorder_cmp.py \
  --before data/inroot/demo/cap_attack.pcap \
  --after data/outroot/tm1_reorder/demo/cap_attack.pcap.pcap \
  --out-json vis/TM1_reorder_stats.json
