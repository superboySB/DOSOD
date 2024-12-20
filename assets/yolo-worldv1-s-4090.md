```
[09/07/2024-21:04:10] [I] === Performance summary ===
[09/07/2024-21:04:10] [I] Throughput: 802.713 qps
[09/07/2024-21:04:10] [I] Latency: min = 1.26953 ms, max = 6.30933 ms, mean = 1.40011 ms, median = 1.29671 ms, percentile(90%) = 1.5542 ms, percentile(95%) = 1.68701 ms, percentile(99%) = 2.52319 ms
[09/07/2024-21:04:10] [I] Enqueue Time: min = 0.995972 ms, max = 6.14551 ms, mean = 1.20034 ms, median = 1.1221 ms, percentile(90%) = 1.39355 ms, percentile(95%) = 1.51538 ms, percentile(99%) = 2.34814 ms
[09/07/2024-21:04:10] [I] H2D Latency: min = 0.203857 ms, max = 0.341919 ms, mean = 0.250004 ms, median = 0.242584 ms, percentile(90%) = 0.280029 ms, percentile(95%) = 0.292236 ms, percentile(99%) = 0.311523 ms
[09/07/2024-21:04:10] [I] GPU Compute Time: min = 0.881592 ms, max = 5.87183 ms, mean = 0.99288 ms, median = 0.887817 ms, percentile(90%) = 1.151 ms, percentile(95%) = 1.27808 ms, percentile(99%) = 2.11157 ms
[09/07/2024-21:04:10] [I] D2H Latency: min = 0.115723 ms, max = 3.60303 ms, mean = 0.157228 ms, median = 0.155029 ms, percentile(90%) = 0.166626 ms, percentile(95%) = 0.167114 ms, percentile(99%) = 0.168457 ms
[09/07/2024-21:04:10] [I] Total Host Walltime: 3.00232 s
[09/07/2024-21:04:10] [I] Total GPU Compute Time: 2.39284 s
[09/07/2024-21:04:10] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[09/07/2024-21:04:10] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[09/07/2024-21:04:10] [W] * GPU compute time is unstable, with coefficient of variance = 31.6956%.
[09/07/2024-21:04:10] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/07/2024-21:04:10] [I] Explanations of the performance metrics are printed in the verbose logs.
```