import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import os
import glob  # Import glob for pattern matching in directories
from scapy.all import sniff, IP, TCP, UDP  # Use scapy for PCAP parsing to handle large packets

# Function to get normalized flow key (5-tuple) for bidirectional merging
def get_flow_key(pkt):
    if IP not in pkt:
        return None
    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
    proto = pkt[IP].proto
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        return None
    # Normalize: smaller IP first
    if src_ip > dst_ip:
        return (dst_ip, src_ip, dport, sport, proto)
    return (src_ip, dst_ip, sport, dport, proto)

# Main function to process multiple PCAPs and collect packet counts per flow
def analyze_pcap_flows(pcap_files, idle_timeout=120.0):
    packet_counts = []  # List to hold packet count per flow
    for pcap_path in pcap_files:
        flows = defaultdict(lambda: {
            'packet_count': 0,
            'start_time': None,
            'last_time': None
        })
        
        # Use scapy's sniff for streaming offline PCAP reading, process each packet
        def process_packet(pkt):
            key = get_flow_key(pkt)
            if key is None:
                return
            
            current_time = float(pkt.time)
            flow = flows[key]
            if flow['start_time'] is None:
                flow['start_time'] = current_time
            
            flow['last_time'] = current_time
            flow['packet_count'] += 1
            
            # Check for TCP termination (FIN or RST flag)
            if TCP in pkt and (pkt[TCP].flags.F or pkt[TCP].flags.R):
                if flow['packet_count'] > 0:
                    packet_counts.append(flow['packet_count'])
                del flows[key]  # Release memory
        
        sniff(offline=pcap_path, prn=process_packet, store=False)  # Stream mode, no storage
        
        # Post-process remaining flows, apply timeout filter
        for key, flow in list(flows.items()):
            if flow['last_time'] - flow['start_time'] <= idle_timeout and flow['packet_count'] > 0:
                packet_counts.append(flow['packet_count'])
    
    return packet_counts

# Function to visualize the distribution as histogram
def visualize_distribution(packet_counts, output_file='output_dist.png'):
    if not packet_counts:
        print("No flows found.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(packet_counts, bins=50, color='blue', edgecolor='black')
    plt.title('Distribution of Packet Counts per Flow')
    plt.xlabel('Number of Packets per Flow')
    plt.ylabel('Number of Flows')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Distribution saved to {output_file}")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pcap_flow_packet_dist.py <folder_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.isdir(input_path):
        print(f"Error: {input_path} is not a directory.")
        sys.exit(1)
    
    # Match all pcap files starting with 'cap' in the folder
    pcap_files = glob.glob(os.path.join(input_path, 'cap*'))
    if not pcap_files:
        print("No pcap files starting with 'cap' found in the directory.")
        sys.exit(1)
    
    counts = analyze_pcap_flows(pcap_files)
    visualize_distribution(counts)