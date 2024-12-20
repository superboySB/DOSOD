```
[09/07/2024-21:35:18] [I] === Performance summary ===
[09/07/2024-21:35:18] [I] Throughput: 481.898 qps
[09/07/2024-21:35:18] [I] Latency: min = 2.21338 ms, max = 21.5757 ms, mean = 2.36123 ms, median = 2.22137 ms, percentile(90%) = 2.2915 ms, percentile(95%) = 2.64233 ms, percentile(99%) = 5.23608 ms
[09/07/2024-21:35:18] [I] Enqueue Time: min = 1.29114 ms, max = 21.397 ms, mean = 1.80525 ms, median = 1.59399 ms, percentile(90%) = 2.1189 ms, percentile(95%) = 2.59033 ms, percentile(99%) = 5.50952 ms
[09/07/2024-21:35:18] [I] H2D Latency: min = 0.200684 ms, max = 0.259521 ms, mean = 0.206849 ms, median = 0.202393 ms, percentile(90%) = 0.231079 ms, percentile(95%) = 0.235962 ms, percentile(99%) = 0.244141 ms
[09/07/2024-21:35:18] [I] GPU Compute Time: min = 1.89233 ms, max = 21.2131 ms, mean = 2.02429 ms, median = 1.89844 ms, percentile(90%) = 1.90262 ms, percentile(95%) = 2.25281 ms, percentile(99%) = 4.84247 ms
[09/07/2024-21:35:18] [I] D2H Latency: min = 0.116577 ms, max = 10.974 ms, mean = 0.130076 ms, median = 0.120117 ms, percentile(90%) = 0.152832 ms, percentile(95%) = 0.157898 ms, percentile(99%) = 0.166748 ms
[09/07/2024-21:35:18] [I] Total Host Walltime: 4.15026 s
[09/07/2024-21:35:18] [I] Total GPU Compute Time: 4.04858 s
[09/07/2024-21:35:18] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[09/07/2024-21:35:18] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[09/07/2024-21:35:18] [W] * GPU compute time is unstable, with coefficient of variance = 44.7285%.
[09/07/2024-21:35:18] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/07/2024-21:35:18] [I] Explanations of the performance metrics are printed in the verbose logs.
```