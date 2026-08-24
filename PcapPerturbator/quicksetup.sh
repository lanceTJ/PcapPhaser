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
export MPLBACKEND="${MPLBACKEND:-Agg}"

if ! python - <<'PY'
import os
import sysconfig
from pathlib import Path

include_dirs = [Path(sysconfig.get_paths()["include"])]
if os.environ.get("PYTHON_INCLUDE_DIR"):
    include_dirs.append(Path(os.environ["PYTHON_INCLUDE_DIR"]))
raise SystemExit(0 if any((path / "Python.h").is_file() for path in include_dirs) else 1)
PY
then
  echo "Python development headers are required to build TrafficManipulator." >&2
  echo "On Ubuntu/Debian, install: sudo apt-get install build-essential python3-dev" >&2
  exit 1
fi

if [[ -n "${PYTHON_INCLUDE_DIR:-}" ]]; then
  export CFLAGS="${CFLAGS:-} -I${PYTHON_INCLUDE_DIR}"
fi

python -m pip install --upgrade pip setuptools wheel
pip install -e .

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

# DoHBrw captures use Linux cooked capture (SLL) link-layer headers. Upstream
# TrafficManipulator only handles Ethernet when it creates a link-layer crafted
# packet, so add the equivalent Scapy CookedLinux branch deterministically.
rebuilder = repo / "rebuilder.py"
legacy_branch = """                if groupList[i].haslayer(Ether):
                    pkt[Ether].remove_payload()
                else:
                    raise RuntimeError(\"Error in rebuilder!\")
"""
sll_branch = """                if groupList[i].haslayer(Ether):
                    pkt[Ether].remove_payload()
                elif groupList[i].haslayer(CookedLinux):
                    pkt[CookedLinux].remove_payload()
                else:
                    raise RuntimeError(\"Error in rebuilder!\")
"""
source = rebuilder.read_text(encoding="utf-8")
if legacy_branch in source:
    rebuilder.write_text(source.replace(legacy_branch, sll_branch, 1), encoding="utf-8")
    print("INFO: enabled Linux cooked capture support in TrafficManipulator.", file=sys.stderr)
elif sll_branch not in source:
    raise RuntimeError("Unsupported TrafficManipulator rebuilder.py layout")

# Scapy exposes Linux cooked link-layer addresses as bytes. Normalize them to
# text before AfterImage concatenates its flow keys.
feature_extractor = repo / "AfterImageExtractor" / "FeatureExtractor.py"
source = feature_extractor.read_text(encoding="utf-8")
legacy_addresses = """            srcMAC = packet.src
            dstMAC = packet.dst
"""
direct_text_addresses = """            srcMAC = str(packet.src)
            dstMAC = str(packet.dst)
"""
text_addresses = """            srcMAC = str(getattr(packet, "src", ""))
            dstMAC = str(getattr(packet, "dst", ""))
"""
if legacy_addresses in source or direct_text_addresses in source:
    old_addresses = legacy_addresses if legacy_addresses in source else direct_text_addresses
    feature_extractor.write_text(
        source.replace(old_addresses, text_addresses, 1),
        encoding="utf-8",
    )
    print("INFO: normalized Scapy link-layer addresses for AfterImage.", file=sys.stderr)
elif text_addresses not in source:
    raise RuntimeError("Unsupported TrafficManipulator FeatureExtractor.py layout")

source = feature_extractor.read_text(encoding="utf-8")
legacy_l2_fallback = """                    srcIP = packet.src  # src MAC
                    dstIP = packet.dst  # dst MAC
"""
safe_l2_fallback = """                    srcIP = srcMAC
                    dstIP = dstMAC
"""
if legacy_l2_fallback in source:
    feature_extractor.write_text(
        source.replace(legacy_l2_fallback, safe_l2_fallback, 1),
        encoding="utf-8",
    )
elif safe_l2_fallback not in source:
    raise RuntimeError("Unsupported TrafficManipulator L2 fallback layout")
PY


pushd external/TrafficManipulator/AfterImageExtractor >/dev/null
python setup.py build_ext --inplace
popd >/dev/null

test -f demo_inputs/demo/cap_attack.pcap
test -f demo_inputs/demo/benign_source.pcap

if [[ "${REBUILD_ASSETS}" == "1" || ! -f demo_assets/mimic_set.npy ]]; then
  python external/TrafficManipulator/extractor.py \
    -i demo_inputs/demo/benign_source.pcap \
    -o demo_assets/mimic_set.npy
fi

if [[ "${REBUILD_ASSETS}" == "1" || ! -f demo_assets/init_benign.pcap ]]; then
  python - <<'PY'
from scapy.all import rdpcap, wrpcap
pkts = rdpcap("demo_inputs/demo/benign_source.pcap", count=200)
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
