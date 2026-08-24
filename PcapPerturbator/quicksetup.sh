#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REBUILD_ASSETS="${REBUILD_ASSETS:-0}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install -e .
sudo apt-get install python3-tk -y

pip install cython matplotlib numpy scipy scapy

mkdir -p external demo_assets data/outroot vis plans scripts

if [[ ! -d external/TrafficManipulator ]]; then
  git clone https://github.com/dongtsi/TrafficManipulator.git external/TrafficManipulator
fi

python - <<'PY'
import re
import sys
from pathlib import Path

try:
    import numpy as np
except Exception as e:
    print(f"WARNING: failed to import numpy in current environment: {e}", file=sys.stderr)
    raise SystemExit(0)

def parse_version(v: str):
    nums = []
    for part in re.split(r'[.+-]', v):
        m = re.match(r'(\d+)', part)
        if not m:
            break
        nums.append(int(m.group(1)))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])

ver = np.__version__
repo = Path("external/TrafficManipulator")

if parse_version(ver) > (1, 8, 0):
    replacements = [
        (re.compile(r'\bnp\.Int\b'),       'np.int'),
        (re.compile(r'\bnumpy\.Int\b'),    'numpy.int'),
        (re.compile(r'\bnp\.Inf\b'),       'np.inf'),
        (re.compile(r'\bnumpy\.Inf\b'),    'numpy.inf'),
    ]
    exts = {".py", ".pyx"}
    total_fixed = 0

    if repo.exists():
        for path in repo.rglob("*"):
            if not path.is_file() or path.suffix not in exts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            new_text = text
            for pattern, repl in replacements:
                new_text, count = pattern.subn(repl, new_text)
                total_fixed += count

            if new_text != text:
                path.write_text(new_text, encoding="utf-8")

    if total_fixed > 0:
        print(
            f"INFO: numpy {ver} (> 1.8) detected, auto-replaced "
            f"{total_fixed} deprecated np.Int / numpy.Int occurrence(s) "
            f"under {repo}. Continuing.",
            file=sys.stderr,
        )
    else:
        print(
            f"INFO: numpy {ver} (> 1.8) detected, no deprecated np.Int / "
            f"numpy.Int found under {repo}. Continuing.",
            file=sys.stderr,
        )
else:
    print(
        f"INFO: numpy {ver} (<= 1.8) detected, no replacement needed. Continuing.",
        file=sys.stderr,
    )
PY


pushd external/TrafficManipulator/AfterImageExtractor >/dev/null
python setup.py build_ext --inplace
popd >/dev/null

test -f data/inroot/demo/cap_attack.pcap
test -f data/inroot/demo/benign_source.pcap

if [[ "${REBUILD_ASSETS}" == "1" || ! -f demo_assets/mimic_set.npy ]]; then
  python external/TrafficManipulator/extractor.py \
    -i data/inroot/demo/benign_source.pcap \
    -o demo_assets/mimic_set.npy
fi

if [[ "${REBUILD_ASSETS}" == "1" || ! -f demo_assets/init_benign.pcap ]]; then
  python - <<'PY'
from scapy.all import rdpcap, wrpcap
pkts = rdpcap("data/inroot/demo/benign_source.pcap", count=200)
if len(pkts) == 0:
    raise SystemExit("benign_source.pcap contains no packets")
wrpcap("demo_assets/init_benign.pcap", pkts[:min(len(pkts), 200)])
print("Wrote demo_assets/init_benign.pcap")
PY
fi

if [[ "${REBUILD_ASSETS}" == "1" || ! -f demo_assets/model.pkl || ! -f demo_assets/normalizer.pkl ]]; then
  read FM AD < <(python - <<'PY'
import numpy as np
n = int(np.load("demo_assets/mimic_set.npy").shape[0])
if n < 50:
    raise SystemExit(f"Too few feature rows for a stable setup: {n}")
fm = max(10, min(500, n // 10))
ad = max(fm + 10, min(n, n // 2))
if ad <= fm:
    ad = min(n, fm + 10)
print(fm, ad)
PY
  )
  echo "Using FMgrace=${FM}, ADgrace=${AD}"

  python external/TrafficManipulator/KitNET/model.py \
    -M train \
    -tf demo_assets/mimic_set.npy \
    -mf demo_assets/model.pkl \
    -fm "${FM}" \
    -ad "${AD}"

  python external/TrafficManipulator/tools.py \
    -tf demo_assets/mimic_set.npy \
    -mf demo_assets/model.pkl \
    -nf demo_assets/normalizer.pkl \
    -fm "${FM}" \
    -ad "${AD}"
fi

python - <<'PY'
from pathlib import Path
root = Path(".").resolve()
for path in [Path("plans/length.json"), Path("plans/rate.json")]:
    text = path.read_text(encoding="utf-8")
    text = text.replace("__REPO_ROOT__", str(root))
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")
PY

echo
echo "Quick setup finished."
echo "Assets ready under demo_assets/."
echo "Next steps:"
echo "  bash quicksetup.sh"
echo "  bash TM1_loss.sh"
echo "  bash TM1_retrans.sh"
echo "  bash TM1_reorder.sh"
echo "  bash TM2_length.sh"
echo "  bash TM2_rate.sh"
