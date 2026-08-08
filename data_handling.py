from scapy.all import PcapReader, IP
import pandas as pd
import numpy as np
import os


def group_by_bursts(df, iat_threshold=1.0):
    """
    dividing into bursts
    """
    # Creating an id for each burst
    df['burst_id'] = (df['iat'] > iat_threshold).cumsum()

    bursts = df.groupby(['source_file', 'burst_id']).agg(
        packet_count=('packet_size', 'count'),
        total_bytes=('packet_size', 'sum'),
        mean_packet_size=('packet_size', 'mean'),
        std_packet_size=('packet_size', 'std'),
        mean_iat=('iat', 'mean'),
        uplink_ratio=('direction', lambda x: x.mean())
    ).reset_index()

    bursts = bursts.fillna(0)
    return bursts


def group_by_time_window(df, window_size='1S'):
    """
    dividing into time windows
    """
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('datetime')

    windows = df.groupby('source_file').resample(window_size).agg(
        packet_count=('packet_size', 'count'),
        total_bytes=('packet_size', 'sum'),
        mean_packet_size=('packet_size', 'mean'),
        std_packet_size=('packet_size', 'std')
    ).reset_index()

    windows = windows[windows['packet_count'] > 0].fillna(0)
    return windows


def process_wireshark_captures(pcap_files_list, capture_device_ip):
    all_data = []

    for file_path in pcap_files_list:
        print(f"Processing {file_path}...")

        prev_time = None

        with PcapReader(file_path) as pcap_reader:
            for pkt in pcap_reader:
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst

                    if src_ip != capture_device_ip and dst_ip != capture_device_ip:
                        continue

                    pkt_size = len(pkt)
                    timestamp = float(pkt.time)

                    # 1- from user, 0- to user
                    direction = 1 if src_ip == capture_device_ip else 0

                    iat = 0.0 if prev_time is None else timestamp - prev_time
                    prev_time = timestamp

                    all_data.append({
                        'source_file': os.basename(file_path),
                        'timestamp': timestamp,
                        'packet_size': pkt_size,
                        'direction': direction,
                        'iat': iat
                    })
    print("Finished extracting features from all files.")
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    files_to_process = []

    # The IP of the device we use
    experiment_ip = ""

    traffic_df = process_wireshark_captures(files_to_process, experiment_ip)
    print(traffic_df.head())
