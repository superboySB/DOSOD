```
[09/02/2024-17:47:04] [I] === Performance summary ===
[09/02/2024-17:47:04] [I] Throughput: 1574.34 qps
[09/02/2024-17:47:04] [I] Latency: min = 0.94249 ms, max = 1.51343 ms, mean = 0.949318 ms, median = 0.948975 ms, percentile(90%) = 0.951172 ms, percentile(95%) = 0.951843 ms, percentile(99%) = 0.953735 ms
[09/02/2024-17:47:04] [I] Enqueue Time: min = 0.539093 ms, max = 1.61792 ms, mean = 0.582978 ms, median = 0.587158 ms, percentile(90%) = 0.598022 ms, percentile(95%) = 0.601318 ms, percentile(99%) = 0.614258 ms
[09/02/2024-17:47:04] [I] H2D Latency: min = 0.201904 ms, max = 0.237549 ms, mean = 0.202861 ms, median = 0.202698 ms, percentile(90%) = 0.203125 ms, percentile(95%) = 0.203369 ms, percentile(99%) = 0.205811 ms
[09/02/2024-17:47:04] [I] GPU Compute Time: min = 0.626678 ms, max = 1.15527 ms, mean = 0.632278 ms, median = 0.631836 ms, percentile(90%) = 0.633789 ms, percentile(95%) = 0.633911 ms, percentile(99%) = 0.634888 ms
[09/02/2024-17:47:04] [I] D2H Latency: min = 0.112061 ms, max = 0.156982 ms, mean = 0.114173 ms, median = 0.113556 ms, percentile(90%) = 0.115723 ms, percentile(95%) = 0.115967 ms, percentile(99%) = 0.116699 ms
[09/02/2024-17:47:04] [I] Total Host Walltime: 3.0019 s
[09/02/2024-17:47:04] [I] Total GPU Compute Time: 2.98815 s
[09/02/2024-17:47:04] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[09/02/2024-17:47:04] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[09/02/2024-17:47:04] [W] * GPU compute time is unstable, with coefficient of variance = 1.22015%.
[09/02/2024-17:47:04] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/02/2024-17:47:04] [I] Explanations of the performance metrics are printed in the verbose logs.
```