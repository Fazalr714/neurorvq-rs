# neurorvq-rs Benchmark Results

**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)  
**Backend:** NdArray + Rayon  
**Iterations:** 10 (after 2 warmup)

| Configuration | Modality | Channels | Patches | Construct (ms) | Encode (ms) | Tokenize (ms) |
|---|---|---:|---:|---:|---:|---:|
| EEG 4ch x64t | EEG | 4 | 64 | 53 | 841.4 ± 4.4 | 847.7 ± 6.7 |
| EEG 8ch x32t | EEG | 8 | 32 | 13 | 844.7 ± 3.9 | 847.0 ± 5.7 |
| EEG 16ch x16t | EEG | 16 | 16 | 13 | 846.3 ± 5.6 | 848.9 ± 4.4 |
| EEG 32ch x8t | EEG | 32 | 8 | 13 | 850.5 ± 4.3 | 850.5 ± 4.8 |
| EEG 64ch x4t | EEG | 64 | 4 | 13 | 854.2 ± 5.1 | 854.1 ± 4.2 |
| ECG 4ch x150t | ECG | 4 | 150 | 13 | 2211.1 ± 3.5 | 2212.2 ± 4.7 |
| ECG 8ch x75t | ECG | 8 | 75 | 13 | 2215.1 ± 4.0 | 2218.4 ± 5.9 |
| ECG 12ch x50t | ECG | 12 | 50 | 12 | 2219.1 ± 5.5 | 2219.7 ± 4.9 |
| ECG 15ch x40t | ECG | 15 | 40 | 12 | 2219.6 ± 4.6 | 2223.5 ± 6.4 |
| EMG 4ch x64t | EMG | 4 | 64 | 48 | 1340.1 ± 2.7 | 1342.9 ± 5.0 |
| EMG 8ch x32t | EMG | 8 | 32 | 26 | 1345.0 ± 3.9 | 1348.4 ± 4.2 |
| EMG 16ch x16t | EMG | 16 | 16 | 26 | 1346.9 ± 4.8 | 1348.6 ± 4.1 |

### Charts

![Tokenize Latency](tokenize_latency.svg)

![Encode Latency](encode_latency.svg)

![Construction Time](construction_time.svg)

![EEG Scaling](eeg_scaling.svg)

