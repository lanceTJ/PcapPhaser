#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

mkdir -p data/outroot vis

pcapperturbator \
  --in-root data/inroot \
  --out-root data/outroot/tm1_loss \
  --backend threads \
  --workers 1 \
  --seed 42 \
  --plan plans/loss.json \
  --verbose

python scripts/tm1_counts.py \
  --before data/inroot/demo/cap_attack.pcap \
  --after data/outroot/tm1_loss/demo/cap_attack.pcap.pcap \
  --mode loss \
  --out-json vis/TM1_loss_stats.json
