# WhatsApp traffic classification

This project builds event-aligned metadata features from truncated Wireshark
captures and trains a classifier for `idle`, `text`, and `photo` activity.
It does not decrypt traffic.

## Dataset layout

```text
WhatsAppStudy/
├── captures/
│   ├── S001_full_00001_....pcapng
│   └── S002_full_00001_....pcapng
├── logs/
│   ├── events_S001.csv
│   └── events_S002.csv
└── derived/
    └── features.csv
```

Each event log must contain:

```csv
session_id,trial_id,pre_send_time,post_send_time,action,variant,result
S001,S001_T001,2026-08-15T18:21:06.058+03:00,2026-08-15T18:21:06.377+03:00,text,1_char,success
```

Capture filenames must use the matching `<session_id>_full` prefix. This
prevents pilot captures from being mixed into a full session. Packet size
uses the original on-wire length, not the 128-byte captured snapshot length.

## Build features

```powershell
python data_handling.py C:\WhatsAppStudy --device-ip 192.168.250.10
```

Each active sample covers the 15 seconds beginning at `pre_send_time`.
Non-overlapping 15-second idle samples are extracted outside guarded event
windows. The resulting labels come from `events_*.csv`, never filenames.

## Train and evaluate

Collect at least two sessions before training. Prefer four or more:

```powershell
python models.py C:\WhatsAppStudy\derived\features.csv `
    --test-session S004 `
    --model artifacts\whatsapp_random_forest.joblib
```

The complete test session is held out. Randomly splitting windows from one
capture would leak connection- and session-specific patterns into evaluation.

The repository also includes a model-ready, privacy-reduced example at
`datasets/features.csv`. It excludes raw captures and exact timestamps:

```powershell
python models.py datasets/features.csv --test-session S002
```
