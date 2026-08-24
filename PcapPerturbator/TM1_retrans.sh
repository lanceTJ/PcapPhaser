#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

mkdir -p data/outroot vis

pcapperturbator \
  --in-root demo_inputs \
  --out-root data/outroot/tm1_retrans \
  --backend threads \
  --workers 1 \
  --seed 42 \
  --plan plans/retrans.json \
  --verbose

python scripts/tm1_counts.py \
  --before demo_inputs/demo/cap_attack.pcap \
  --after data/outroot/tm1_retrans/demo/cap_attack.pcap \
  --mode retrans \
  --out-json vis/TM1_retrans_stats.json
