# PcapPerturbator

PcapPerturbator is a PCAP perturbation framework for reproducible traffic-robustness experiments.

It is designed for two complementary use cases:

1. **Streaming packet-level perturbation** for fast, scalable batch processing on large PCAP corpora.
2. **TrafficManipulator-assisted one-dimensional perturbation** for controlled robustness studies where packet **length** and packet **rate / timing** must be isolated as separate experimental axes.

The framework keeps the high-throughput chunked pipeline for simple packet-level operations and adds file-level offline stages for TrafficManipulator-backed transformations. The result is a single toolkit that can be used both for broad perturbation sweeps and for carefully controlled staircase-strength experiments.

## Core capabilities

### Streaming stages

These stages run inside the chunked streaming pipeline and are suitable for large datasets.

- `loss`: randomly drop packets
- `retransmit`: duplicate packets
- `reorder`: reorder packets inside local segments while preserving valid output ordering
- `seq_offset`: modify TCP sequence numbers and recompute checksums

### Offline TrafficManipulator-assisted stages

These stages run at file level and are intended for controlled perturbation generation.

- `length_manip`: use TrafficManipulator as an intermediate generator, then project the result back to a final PCAP where **packet count is unchanged**, **timestamps are unchanged**, and **only packet lengths are modified**
- `rate_manip`: select a subset of flows, run TrafficManipulator in time-only mode, then merge the changed timestamps back so the final PCAP keeps **packet count unchanged**, **packet lengths unchanged**, and modifies **timing only** on the selected subset

## Design goals

PcapPerturbator is built around the following principles:

- **Reproducibility**: deterministic seed mixing and per-file metadata output
- **Scalability**: chunked streaming path for low-overhead perturbations
- **Dimensional control**: isolate length and rate as separate experimental axes
- **Artifact readiness**: output mirrored directory layout, classic PCAP output, and sidecar JSON metadata for auditing
- **Research compatibility**: integrate TrafficManipulator without forcing the whole system into its research-script execution model

## Repository layout

```text
pcapperturbator/
├─ batch.py           # directory-level orchestration
├─ cli.py             # CLI entrypoint
├─ io.py              # buffered classic-PCAP writer
├─ manip_stages.py    # length_manip / rate_manip implementations
├─ perturbations.py   # streaming-stage registry
├─ pipeline.py        # stage dispatcher and execution pipeline
├─ stream.py          # PCAP / PCAPNG readers
├─ tm_bridge.py       # TrafficManipulator compatibility bridge
├─ utils.py           # logging / filesystem helpers
└─ verify.py          # invariant checks and strength metrics
```

## Installation

### 1. Create a Python environment

The local package requires Python 3.9+, while upstream TrafficManipulator was originally tested on Python 3.6 and depends on `cython`, `scapy`, `numpy`, `scipy`, and `matplotlib`. The safest practical compromise for this integrated project is to use **Python 3.9** in one shared environment: it satisfies this package, and the bridge in `pcapperturbator/tm_bridge.py` patches the legacy `time.clock()` call used by upstream TrafficManipulator. Upstream TrafficManipulator explicitly documents Python 3.6 testing, lists `cython` and `scapy` as special dependencies, and pins `Cython==0.29.11`, `matplotlib==3.0.3`, `numpy==1.18.5`, `scapy==2.4.2`, and `scipy==1.4.1`.

Example setup:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PcapPerturbator

From the project root:

```bash
pip install -e .
```

### 3. Clone TrafficManipulator into a fixed local path

```bash
git clone https://github.com/dongtsi/TrafficManipulator.git external/TrafficManipulator
```

The bridge imports TrafficManipulator modules directly from the repository path you provide in the plan file, so the repository must exist locally and the dependencies must be installed in the **same Python environment** used to run `pcapperturbator`.

### 4. Install TrafficManipulator dependencies in the same environment

Upstream TrafficManipulator marks `cython` and `scapy` as special dependencies and publishes a requirements set containing `Cython==0.29.11`, `matplotlib==3.0.3`, `numpy==1.18.5`, `scapy==2.4.2`, and `scipy==1.4.1`.

A practical installation sequence is:

```bash
sudo apt-get install python3-tk
pip install cython scapy scipy matplotlib
```

If you need stricter compatibility with the original repository, install the pinned versions documented upstream instead. That pin set comes from the upstream README and reflects the environment they reported, not a constraint enforced by this package.

### 5. Build the AfterImageExtractor extension

TrafficManipulator's README instructs you to compile the `AfterImageExtractor` Cython module before running its Kitsune-based pipeline.

```bash
cd external/TrafficManipulator/AfterImageExtractor
python setup.py build_ext --inplace
cd ..
```

If this step is skipped, TrafficManipulator typically fails later during import or feature extraction.

## Preparing TrafficManipulator inputs

`length_manip` and `rate_manip` both depend on TrafficManipulator inputs. The bridge uses the same core arguments that upstream `main.py` expects:

- `mal_pcap`: malicious traffic PCAP
- `mimic_set`: benign feature set to mimic (`.npy`)
- `normalizer`: compiled feature normalizer (`.pkl`)
- `init_pcap`: preparatory / warm-up traffic PCAP

These required arguments and defaults are defined in upstream `main.py`, and TrafficManipulator constructs its `Manipulator` object from `mal_pcap`, `mimic_set`, `normalizer`, and `init_pcap`.

### 1. Prepare a benign feature set (`mimic_set.npy`)

TrafficManipulator uses a benign feature array as the target feature set. Upstream also provides `extractor.py` to turn a raw PCAP into features with:

```bash
python extractor.py -i ./example/test.pcap -o ./example/test.npy
```

That command is documented in the upstream README and `extractor.py`.

In practice, you usually generate features from benign traffic and save them as a NumPy array such as:

```text
/data/features/train_ben.npy
```

### 2. Generate a normalizer (`normalizer.pkl`)

TrafficManipulator's `tools.py` builds the normalizer from a feature matrix. Upstream exposes the relevant arguments as:

- `-tf / --feat_file_path`
- `-mf / --model_file_path`
- `-nf / --normalizer_file_path`
- `-fm / --FMgrace`
- `-ad / --ADgrace`

and serializes the resulting normalizer to the target pickle file.

A typical command is:

```bash
cd external/TrafficManipulator
python KitNET/model.py -M train -tf example/train_ben.npy

python tools.py \
  -tf ./example/train_ben.npy \
  -nf ./example/normalizer.pkl
```

### 3. Prepare an initialization PCAP (`init_pcap`)

TrafficManipulator loads an initialization PCAP when constructing the Kitsune feature extractor. In upstream code, the constructor defaults this to `./data/empty.pcap`, reads it with Scapy, and uses it to initialize the global feature extractor.

For this integrated framework, the most reliable choice is to create or keep a tiny valid PCAP and reference it explicitly in the plan file, for example:

```text
external/TrafficManipulator/data/empty.pcap
```

Do not point `init_pcap` at a missing path. If it is absent, TrafficManipulator fails before the manipulation stage begins.

## Minimal environment checklist

Before you run `length_manip` or `rate_manip`, verify all of the following:

- `external/TrafficManipulator` exists locally
- TrafficManipulator dependencies are installed in the same environment as `pcapperturbator`
- `external/TrafficManipulator/AfterImageExtractor` has been compiled with `build_ext --inplace`
- `mimic_set.npy` exists and is readable
- `normalizer.pkl` exists and is readable
- `init_pcap` exists and is a valid PCAP file
- the plan file points to the correct local paths

If any of these are missing, most failures happen at import time or immediately when TrafficManipulator starts building the feature extractor.

## Running the toolkit

### Quick CLI for streaming stages

```bash
pcapperturbator \
  --in-root /data/in \
  --out-root /data/out_stream \
  --backend threads \
  --workers 4 \
  --seed 42 \
  --seq-offset 0.02:500
```

Supported quick flags:

- `--loss`
- `--retransmit`
- `--seq-offset`

### Plan-based execution for mixed pipelines

Use a JSON plan whenever you want to combine stages or invoke TrafficManipulator-assisted stages.

```bash
pcapperturbator \
  --in-root /data/in \
  --out-root /data/out \
  --backend processes \
  --workers 2 \
  --seed 42 \
  --plan plan.json \
  --verbose
```

## Plan schema

A plan is a list of stage dictionaries. Stages run in order.

Each stage has the form:

```json
{
  "type": "stage_name",
  "pct": 0.1,
  "params": {}
}
```

Notes:

- `pct` is required for stochastic packet-level stages and for `rate_manip` flow selection.
- `pct` is not used by `length_manip`; its effective strength is controlled by TrafficManipulator parameters and the projected byte budget.
- `params` is stage-specific.

## Supported stages

### 1. `loss`

Randomly drops packets.

```json
[{"type": "loss", "pct": 0.05, "params": {}}]
```

### 2. `retransmit`

Duplicates packets.

```json
[{"type": "retransmit", "pct": 0.03, "params": {}}]
```

### 3. `reorder`

Locally reorders packets inside chunk-level segments.

```json
[{"type": "reorder", "pct": 1.0, "params": {"m": 10}}]
```

### 4. `seq_offset`

Changes TCP sequence numbers on a subset of packets.

```json
[{"type": "seq_offset", "pct": 0.02, "params": {"offset": 500}}]
```

### 5. `length_manip`

Runs TrafficManipulator on the full input PCAP, reconstructs its intermediate mutation result, extracts crafted-packet byte budgets, and transfers those budgets to original packets in the same flow direction.

```json
[
  {
    "type": "length_manip",
    "params": {
      "tm": {
        "repo": "external/TrafficManipulator",
        "mimic_set": "/data/features/train_ben.npy",
        "normalizer": "/data/features/normalizer.pkl",
        "init_pcap": "external/TrafficManipulator/data/empty.pcap"
      },
      "particle": {"w": 0.7298, "c1": 1.49618, "c2": 1.49618},
      "pso": {"max_iter": 3, "particle_num": 6, "grp_size": 3},
      "manipulator": {
        "grp_size": 100,
        "min_time_extend": 3.0,
        "max_time_extend": 6.0,
        "max_cft_pkt": 1,
        "max_crafted_pkt_prob": 0.05
      },
      "budget_basis": "packet_len",
      "cap_bytes": 1460,
      "spill_mode": "forward",
      "pad_byte": "00"
    }
  }
]
```

Final output invariants:

- packet count unchanged
- timestamps unchanged
- only packet lengths modified

### 6. `rate_manip`

Selects a subset of flows, runs TrafficManipulator in time-only mode with crafted-packet generation disabled, maps the changed timestamps back to original packet ordinals, and merges the selected subset back with the untouched traffic.

```json
[
  {
    "type": "rate_manip",
    "pct": 0.2,
    "params": {
      "tm": {
        "repo": "external/TrafficManipulator",
        "mimic_set": "/data/features/train_ben.npy",
        "normalizer": "/data/features/normalizer.pkl",
        "init_pcap": "external/TrafficManipulator/data/empty.pcap"
      },
      "particle": {"w": 0.7298, "c1": 1.49618, "c2": 1.49618},
      "pso": {"max_iter": 3, "particle_num": 6, "grp_size": 3},
      "manipulator": {
        "grp_size": 100,
        "min_time_extend": 3.0,
        "max_time_extend": 6.0
      },
      "select": {"mode": "flow_uniform"},
      "merge": {"sort_by_ts": true},
      "verify": {"tau_usec": 1}
    }
  }
]
```

Final output invariants:

- packet count unchanged
- packet lengths unchanged
- timestamps modified only for the selected subset

## Verification and metadata

Every processed PCAP emits a sidecar metadata JSON file. The metadata captures:

- input and output path
- executed plan
- seed and chunk size
- runtime statistics
- stage-level verification results

### Length-stage metrics

- `S_len = transferred_bytes / total_bytes_original`
- `P_len = changed_packets / total_packets`
- packet-count invariant
- timestamp invariant

### Rate-stage metrics

- `S_rate = mean(|Δt' - Δt| / max(Δt, 1))`
- `P_rate = changed_packets / total_packets`
- packet-count invariant
- length invariant

These values are intended to be used as the actual perturbation-strength axis when running robustness curves.

## Output behavior

- The input directory structure is mirrored under the output root.
- Output packet captures are written as classic PCAP files.
- Metadata is written next to each output PCAP.
- `--resume` skips files whose target output already exists.

## Performance notes

- Streaming stages are the fast path.
- `seq_offset` is slower than pure index-based operations because packets must be parsed and rewritten.
- `length_manip` and `rate_manip` are file-level offline stages and are substantially slower than streaming stages because they invoke TrafficManipulator and perform whole-file reconstruction.
- For TrafficManipulator-assisted stages, `processes` is generally a safer backend than `threads`.

## Troubleshooting

### TrafficManipulator import fails

Check that:

- `params.tm.repo` points to the local TrafficManipulator repository root
- the repository exists on disk
- dependencies were installed into the same Python environment as `pcapperturbator`
- `AfterImageExtractor` was compiled successfully

### `normalizer.pkl` or `mimic_set.npy` is missing

Generate them first. `tools.py` creates the normalizer from a feature matrix, and `extractor.py` turns a PCAP into a feature matrix.

### `init_pcap` path is wrong

Provide an existing valid PCAP. Upstream TrafficManipulator reads this file during `Manipulator` initialization.

### Python 3.11+ compatibility concerns

Upstream TrafficManipulator was published against an older Python stack and uses `time.clock()` in its processing path. This project patches that specific legacy call in the bridge layer, but the most conservative setup remains a Python 3.9 environment. The upstream repository itself reports Python 3.6 testing.
