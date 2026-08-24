#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

if grep -q "__REPO_ROOT__" plans/rate.json; then
  sed -i "s|__REPO_ROOT__|${REPO_ROOT}|g" plans/rate.json
fi

mkdir -p data/outroot vis

pcapperturbator \
  --in-root data/inroot \
  --out-root data/outroot/tm2_rate \
  --backend threads \
  --workers 1 \
  --seed 42 \
  --plan plans/rate.json \
  --verbose

AFTER_PCAP="data/outroot/tm2_rate/demo/cap_attack.pcap.pcap"
if [[ ! -f "${AFTER_PCAP}" ]]; then
  echo "TM2 rate perturbation failed: output pcap not generated" >&2
  exit 1
fi

python scripts/rate_cmp.py \
  --benign data/inroot/demo/benign_source.pcap \
  --before data/inroot/demo/cap_attack.pcap \
  --after data/outroot/tm2_rate/demo/cap_attack.pcap.pcap \
  --out-png vis/TM2_rate_iat.png \
  --out-json vis/TM2_rate_stats.json
