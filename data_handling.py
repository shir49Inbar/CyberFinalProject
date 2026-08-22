import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader


EVENT_COLUMNS = {
    "session_id",
    "trial_id",
    "pre_send_time",
    "post_send_time",
    "action",
    "variant",
    "result",
}

FEATURE_COLUMNS = [
    "packet_count",
    "total_bytes",
    "uplink_packets",
    "downlink_packets",
    "uplink_bytes",
    "downlink_bytes",
    "uplink_ratio",
    "mean_packet_size",
    "std_packet_size",
    "min_packet_size",
    "max_packet_size",
    "mean_iat",
    "std_iat",
    "max_iat",
    "traffic_duration",
    "first_packet_delay",
    "tcp_packet_count",
    "udp_packet_count",
    "burst_count",
    "max_burst_packets",
    "direction_changes",
]


def load_events(path):
    events = pd.read_csv(path)
    missing = EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    events = events[events["result"].str.lower() == "success"].copy()
    events["pre_send_time"] = pd.to_datetime(
        events["pre_send_time"], utc=True, errors="raise"
    )
    events["post_send_time"] = pd.to_datetime(
        events["post_send_time"], utc=True, errors="raise"
    )
    return events.sort_values("pre_send_time")


def read_packets(pcap_paths, device_ip):
    rows = []
    for path in sorted(pcap_paths):
        print(f"Reading {path}...")
        with PcapReader(str(path)) as reader:
            for packet in reader:
                if IP not in packet:
                    continue

                source = packet[IP].src
                destination = packet[IP].dst
                if device_ip not in (source, destination):
                    continue

                wire_length = getattr(packet, "wirelen", None) or len(packet)
                rows.append(
                    {
                        "timestamp": float(packet.time),
                        "packet_size": int(wire_length),
                        "direction": 1 if source == device_ip else 0,
                        "protocol": (
                            "tcp" if TCP in packet else "udp" if UDP in packet else "other"
                        ),
                    }
                )

    if not rows:
        raise ValueError(f"No IPv4 packets for {device_ip} were found")
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _aggregate_window(packets, start, end):
    selected = packets[
        (packets["timestamp"] >= start) & (packets["timestamp"] < end)
    ].copy()
    if selected.empty:
        return {column: 0 for column in FEATURE_COLUMNS}

    selected["iat"] = selected["timestamp"].diff().fillna(0)
    selected["burst_id"] = (selected["iat"] > 1.0).cumsum()
    burst_sizes = selected.groupby("burst_id").size()
    uplink = selected[selected["direction"] == 1]
    downlink = selected[selected["direction"] == 0]

    return {
        "packet_count": len(selected),
        "total_bytes": int(selected["packet_size"].sum()),
        "uplink_packets": len(uplink),
        "downlink_packets": len(downlink),
        "uplink_bytes": int(uplink["packet_size"].sum()),
        "downlink_bytes": int(downlink["packet_size"].sum()),
        "uplink_ratio": float(selected["direction"].mean()),
        "mean_packet_size": float(selected["packet_size"].mean()),
        "std_packet_size": float(selected["packet_size"].std(ddof=0)),
        "min_packet_size": int(selected["packet_size"].min()),
        "max_packet_size": int(selected["packet_size"].max()),
        "mean_iat": float(selected["iat"].mean()),
        "std_iat": float(selected["iat"].std(ddof=0)),
        "max_iat": float(selected["iat"].max()),
        "traffic_duration": float(
            selected["timestamp"].iloc[-1] - selected["timestamp"].iloc[0]
        ),
        "first_packet_delay": float(selected["timestamp"].iloc[0] - start),
        "tcp_packet_count": int((selected["protocol"] == "tcp").sum()),
        "udp_packet_count": int((selected["protocol"] == "udp").sum()),
        "burst_count": int(len(burst_sizes)),
        "max_burst_packets": int(burst_sizes.max()),
        "direction_changes": int(selected["direction"].diff().abs().fillna(0).sum()),
    }


def build_session_features(
    packets,
    events,
    event_window_seconds=15,
    idle_window_seconds=15,
    idle_guard_seconds=5,
    max_idle_windows=20,
):
    rows = []
    session_id = str(events["session_id"].iloc[0])

    for event in events.itertuples(index=False):
        start = event.pre_send_time.timestamp()
        features = _aggregate_window(
            packets, start, start + event_window_seconds
        )
        rows.append(
            {
                "session_id": session_id,
                "trial_id": event.trial_id,
                "event_time": event.pre_send_time.isoformat(),
                "label": event.action.lower(),
                "variant": event.variant,
                **features,
            }
        )

    exclusions = [
        (
            event.pre_send_time.timestamp() - idle_guard_seconds,
            event.pre_send_time.timestamp()
            + event_window_seconds
            + idle_guard_seconds,
        )
        for event in events.itertuples(index=False)
    ]

    idle_number = 0
    cursor = float(packets["timestamp"].min())
    capture_end = float(packets["timestamp"].max())
    while (
        cursor + idle_window_seconds <= capture_end
        and idle_number < max_idle_windows
    ):
        window_end = cursor + idle_window_seconds
        overlaps_event = any(
            cursor < excluded_end and window_end > excluded_start
            for excluded_start, excluded_end in exclusions
        )
        if not overlaps_event:
            idle_number += 1
            rows.append(
                {
                    "session_id": session_id,
                    "trial_id": f"{session_id}_IDLE_{idle_number:03d}",
                    "event_time": pd.to_datetime(cursor, unit="s", utc=True).isoformat(),
                    "label": "idle",
                    "variant": f"{idle_window_seconds}s",
                    **_aggregate_window(packets, cursor, window_end),
                }
            )
        cursor = window_end

    return pd.DataFrame(rows)


def build_dataset(dataset_root, device_ip, output_path=None):
    root = Path(dataset_root)
    log_paths = sorted((root / "logs").glob("events_*.csv"))
    if not log_paths:
        raise FileNotFoundError(
            f"No events_*.csv files found in {root / 'logs'}")

    session_frames = []
    for log_path in log_paths:
        events = load_events(log_path)
        if events.empty:
            continue

        session_ids = events["session_id"].astype(str).unique()
        if len(session_ids) != 1:
            raise ValueError(f"{log_path} must contain exactly one session_id")
        session_id = session_ids[0]

        pcap_paths = sorted(
            (root / "captures").glob(f"{session_id}_full*.pcapng")
        )
        if not pcap_paths:
            raise FileNotFoundError(
                f"No {session_id}_full*.pcapng files in {root / 'captures'}"
            )

        packets = read_packets(pcap_paths, device_ip)
        session_frames.append(build_session_features(packets, events))

    if not session_frames:
        raise ValueError("No successful event rows were found")

    dataset = pd.concat(session_frames, ignore_index=True)
    destination = Path(output_path or root / "derived" / "features.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False)
    print(f"Wrote {len(dataset)} samples to {destination}")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build event-aligned WhatsApp traffic features."
    )
    parser.add_argument(
        "dataset_root", help="Folder containing captures/ and logs/")
    parser.add_argument("--device-ip", required=True,
                        help="WhatsApp VM IPv4 address")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.dataset_root, args.device_ip, args.output)


if __name__ == "__main__":
    main()
