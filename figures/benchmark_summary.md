# neurorvq-rs Benchmark Results

**Platform:** Apple M4 Pro, 64 GB RAM, macOS (arm64)  
**Backend:** NdArray + Rayon  
**Iterations:** 10 (after 2 warmup)

| Configuration | Modality | Channels | Patches | Construct (ms) | Encode (ms) | Tokenize (ms) |
|---|---|---:|---:|---:|---:|---:|
| EEG 4ch x64t | EEG | 4 | 64 | 60 | 831.2 ± 4.3 | 832.0 ± 4.4 |
| EEG 8ch x32t | EEG | 8 | 32 | 12 | 831.8 ± 4.6 | 832.2 ± 3.7 |
| EEG 16ch x16t | EEG | 16 | 16 | 12 | 831.1 ± 4.6 | 833.5 ± 4.2 |
| EEG 32ch x8t | EEG | 32 | 8 | 12 | 834.0 ± 4.2 | 832.7 ± 5.0 |
| EEG 64ch x4t | EEG | 64 | 4 | 12 | 834.8 ± 4.0 | 835.7 ± 4.4 |
| ECG 4ch x150t | ECG | 4 | 150 | 11 | 2169.4 ± 2.4 | 2170.6 ± 4.4 |
| ECG 8ch x75t | ECG | 8 | 75 | 11 | 2176.7 ± 13.6 | 2172.7 ± 3.5 |
| ECG 12ch x50t | ECG | 12 | 50 | 12 | 2174.8 ± 4.0 | 2175.7 ± 3.2 |
| ECG 15ch x40t | ECG | 15 | 40 | 13 | 2177.9 ± 4.1 | 2179.7 ± 2.8 |
| EMG 4ch x64t | EMG | 4 | 64 | 64 | 1311.4 ± 3.4 | 1311.5 ± 5.2 |
| EMG 8ch x32t | EMG | 8 | 32 | 25 | 1315.8 ± 4.7 | 1314.4 ± 4.9 |
| EMG 16ch x16t | EMG | 16 | 16 | 25 | 1315.5 ± 3.7 | 1317.8 ± 3.5 |

### Charts

![Tokenize Latency](tokenize_latency.svg)

![Encode Latency](encode_latency.svg)

![Construction Time](construction_time.svg)

![EEG Scaling](eeg_scaling.svg)

