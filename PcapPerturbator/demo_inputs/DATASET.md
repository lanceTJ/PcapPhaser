# CIRA-CIC-DoHBrw-2020 demo captures

The two compact captures in `demo/` are derived from the public
CIRA-CIC-DoHBrw-2020 dataset released by the Canadian Institute for
Cybersecurity at the University of New Brunswick.

The official dataset page permits redistribution, republication, and mirroring
provided that DoHMeter and the dataset paper are cited:

> Mohammadreza MontazeriShatoori, Logan Davidson, Gurdip Kaur, and Arash
> Habibi Lashkari. "Detection of DoH Tunnels using Time-series Classification
> of Encrypted Traffic." 5th IEEE Cyber Science and Technology Congress, 2020.

Official source: <https://www.unb.ca/cic/datasets/dohbrw-2020.html>

## Files

| File | Label | Derivation | Packets | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| `demo/benign_source.pcap` | Benign | Packets 15,001-20,000 from `benign/pcap/Google/small/dump.pcap` | 5,000 | 2,720,200 | `ad9f0faa3f29e63f9612f7180a806ec72e86cb9bacf2a0a9471623ba798fa4d5` |
| `demo/cap_attack.pcap` | Malicious DoH tunnel | Unmodified `dns2tcp_tunnel_99911_doh2_2020-03-31T02:41:41.265959.pcap` | 9,060 | 2,096,248 | `98db430aa80da6eca9473be7bbe9caa2f2a3dbbcc340c5af1eaa2b057279c01c` |

The benign subset was created at complete-packet boundaries with Wireshark
`editcap -r`; it is not a byte-level truncation. These captures are intentionally
small enough for installation checks, perturbation demonstrations, and PSS
artifact evaluation.
