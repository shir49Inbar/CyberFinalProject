# Dataset

`features.csv` contains 160 event-aligned aggregate traffic samples:

| Session | Idle | Photo | Text |
|---|---:|---:|---:|
| S001 | 20 | 20 | 40 |
| S002 | 20 | 20 | 40 |

Each active sample covers 15 seconds beginning immediately before the
automated send action. Idle samples use non-overlapping 15-second windows
outside guarded event intervals.

The public dataset excludes raw PCAPs, IP addresses, payload bytes, exact
timestamps, and Power Automate event logs. It retains only labels and
aggregate packet size, direction, timing, burst, and protocol features used
by `models.py`.

The two sessions were collected on one device, account, network, and day.
They are suitable for pipeline development and preliminary evaluation, not
for claims of generalization.
