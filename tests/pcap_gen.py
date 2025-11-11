from scapy.all import Ether, IP, TCP, wrpcap
import time

# Generate 10 packets in a single flow: src=192.168.1.1, dst=192.168.1.2, sport=12345, dport=80, proto=6
packets = []
base_time = time.time()

for i in range(10):
    pkt = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80, seq=i, ack=0)
    pkt.time = base_time + i * 1.0  # Increment timestamp by 1 second each
    if i == 7:  # Set FIN flag on 8th packet
        pkt[TCP].flags = 'F'
    if i == 5:  # Simulate timeout: jump timestamp by 70 seconds
        pkt.time += 70
    packets.append(pkt)

# Write to test.pcap
wrpcap("test.pcap", packets)
print("Generated test.pcap with 10 packets.")