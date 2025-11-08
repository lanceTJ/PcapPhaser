import sys
import pandas as pd
import matplotlib.pyplot as plt

# Main function to analyze CSV, compute stats, plot distribution, and generate report
def analyze_csv(csv_path):
    # Load CSV into DataFrame
    df = pd.read_csv(csv_path)
    
    # Calculate total packets per flow: Tot Fwd Pkts + Tot Bwd Pkts
    df['Total Packets'] = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
    
    # Filter out flows with zero packets if any
    valid_df = df[df['Total Packets'] > 0]
    packet_counts = valid_df['Total Packets'].tolist()
    
    if not packet_counts:
        print("No valid flows with packets > 0 found.")
        return
    
    # Compute key statistics
    total_flows = len(valid_df)
    max_packets = valid_df['Total Packets'].max()
    min_packets = valid_df['Total Packets'].min()
    mean_packets = valid_df['Total Packets'].mean()
    median_packets = valid_df['Total Packets'].median()
    std_packets = valid_df['Total Packets'].std()
    p99_packets = valid_df['Total Packets'].quantile(0.99)
    p98_packets = valid_df['Total Packets'].quantile(0.98)
    p97_packets = valid_df['Total Packets'].quantile(0.97)
    p96_packets = valid_df['Total Packets'].quantile(0.96)
    p95_packets = valid_df['Total Packets'].quantile(0.95)
    p94_packets = valid_df['Total Packets'].quantile(0.94)
    p93_packets = valid_df['Total Packets'].quantile(0.93)
    p92_packets = valid_df['Total Packets'].quantile(0.92)
    p91_packets = valid_df['Total Packets'].quantile(0.91)
    p90_packets = valid_df['Total Packets'].quantile(0.9)
    p80_packets = valid_df['Total Packets'].quantile(0.8)
    p70_packets = valid_df['Total Packets'].quantile(0.7)
    p60_packets = valid_df['Total Packets'].quantile(0.6)
    p50_packets = valid_df['Total Packets'].quantile(0.5)
    p40_packets = valid_df['Total Packets'].quantile(0.4)
    p30_packets = valid_df['Total Packets'].quantile(0.3)
    p20_packets = valid_df['Total Packets'].quantile(0.2)
    p10_packets = valid_df['Total Packets'].quantile(0.1)
    p5_packets = valid_df['Total Packets'].quantile(0.05)
    proportion_over_100 = (valid_df['Total Packets'] > 100).sum() / total_flows * 100
    
    # Generate text report
    report = f"""
Flow Packet Count Statistics Report:
- Total number of valid flows: {total_flows}
- Maximum packets per flow: {max_packets}
- Minimum packets per flow (>0): {min_packets}
- Average packets per flow: {mean_packets:.2f}
- Median packets per flow: {median_packets}
- Standard deviation of packets: {std_packets:.2f}
- 99th percentile packets: {p99_packets}
- 98th percentile packets: {p98_packets}
- 97th percentile packets: {p97_packets}
- 96th percentile packets: {p96_packets}
- 95th percentile packets: {p95_packets}
- 94th percentile packets: {p94_packets}
- 93th percentile packets: {p93_packets}
- 92th percentile packets: {p92_packets}
- 91th percentile packets: {p91_packets}
- 90th percentile packets: {p90_packets}
- 80th percentile packets: {p80_packets}
- 70th percentile packets: {p70_packets}
- 60th percentile packets: {p60_packets}
- 50th percentile packets: {p50_packets}
- 40th percentile packets: {p40_packets}
- 30th percentile packets: {p30_packets}
- 20th percentile packets: {p20_packets}
- 10th percentile packets: {p10_packets}
- 5th percentile packets: {p5_packets}

- Proportion of flows with >100 packets: {proportion_over_100:.2f}%
"""
    print(report)
    with open('output_report.txt', 'w') as f:
        f.write(report)
    print("Report saved to output_report.txt")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(packet_counts, bins=50, color='blue', edgecolor='black')
    plt.title('Distribution of Total Packets per Flow')
    plt.xlabel('Number of Packets per Flow')
    plt.ylabel('Number of Flows')
    plt.grid(True)
    plt.savefig('output_dist.png')
    plt.close()
    print("Distribution saved to output_dist.png")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    analyze_csv(csv_path)