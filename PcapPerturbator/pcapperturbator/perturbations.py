from __future__ import annotations

from scapy.layers.inet import IP, TCP



def perturb_packet_loss(pkt):
    """Drop the packet."""
    return None



def perturb_retransmit(pkt):
    """Duplicate the packet once."""
    return [pkt, pkt.copy()]



def perturb_seq_offset(pkt, offset: int = 1000):
    """Apply a TCP sequence-number offset and recalculate checksums."""
    if TCP not in pkt:
        return pkt

    mutated = pkt.copy()
    mutated[TCP].seq = int(mutated[TCP].seq) + int(offset)
    if IP in mutated:
        del mutated[IP].chksum
    del mutated[TCP].chksum
    return mutated


PERTURBATIONS = {
    "loss": perturb_packet_loss,
    "retransmit": perturb_retransmit,
    "retrans": perturb_retransmit,
    "seq_offset": perturb_seq_offset,
}
