```
[09/02/2024-17:14:34] [I] === Performance summary ===
[09/02/2024-17:14:34] [I] Throughput: 1099.65 qps
[09/02/2024-17:14:34] [I] Latency: min = 1.12085 ms, max = 5.30969 ms, mean = 1.17602 ms, median = 1.17322 ms, percentile(90%) = 1.18164 ms, percentile(95%) = 1.18457 ms, percentile(99%) = 1.20386 ms
[09/02/2024-17:14:34] [I] Enqueue Time: min = 0.697998 ms, max = 5.14844 ms, mean = 0.865324 ms, median = 0.88208 ms, percentile(90%) = 0.89624 ms, percentile(95%) = 0.901215 ms, percentile(99%) = 0.947876 ms
[09/02/2024-17:14:34] [I] H2D Latency: min = 0.202148 ms, max = 0.24292 ms, mean = 0.219057 ms, median = 0.221542 ms, percentile(90%) = 0.225708 ms, percentile(95%) = 0.227783 ms, percentile(99%) = 0.236572 ms
[09/02/2024-17:14:34] [I] GPU Compute Time: min = 0.803711 ms, max = 4.93372 ms, mean = 0.818735 ms, median = 0.809937 ms, percentile(90%) = 0.811035 ms, percentile(95%) = 0.811035 ms, percentile(99%) = 0.812012 ms
[09/02/2024-17:14:34] [I] D2H Latency: min = 0.112061 ms, max = 0.168457 ms, mean = 0.138235 ms, median = 0.142334 ms, percentile(90%) = 0.147095 ms, percentile(95%) = 0.149506 ms, percentile(99%) = 0.15715 ms
[09/02/2024-17:14:34] [I] Total Host Walltime: 3.00186 s
[09/02/2024-17:14:34] [I] Total GPU Compute Time: 2.70264 s
[09/02/2024-17:14:34] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[09/02/2024-17:14:34] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[09/02/2024-17:14:34] [W] * GPU compute time is unstable, with coefficient of variance = 20.4616%.
[09/02/2024-17:14:34] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/02/2024-17:14:34] [I] Explanations of the performance metrics are printed in the verbose logs.
```