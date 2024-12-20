```
[09/07/2024-21:22:23] [I] === Performance summary ===
[09/07/2024-21:22:23] [I] Throughput: 687.676 qps
[09/07/2024-21:22:23] [I] Latency: min = 1.68785 ms, max = 10.8534 ms, mean = 1.75215 ms, median = 1.69934 ms, percentile(90%) = 1.74597 ms, percentile(95%) = 1.78418 ms, percentile(99%) = 2.92371 ms
[09/07/2024-21:22:23] [I] Enqueue Time: min = 1.00952 ms, max = 10.7031 ms, mean = 1.29381 ms, median = 1.20312 ms, percentile(90%) = 1.4812 ms, percentile(95%) = 1.64844 ms, percentile(99%) = 2.86414 ms
[09/07/2024-21:22:23] [I] H2D Latency: min = 0.201416 ms, max = 0.246094 ms, mean = 0.20623 ms, median = 0.202393 ms, percentile(90%) = 0.221924 ms, percentile(95%) = 0.234863 ms, percentile(99%) = 0.24231 ms
[09/07/2024-21:22:23] [I] GPU Compute Time: min = 1.37114 ms, max = 10.4899 ms, mean = 1.42454 ms, median = 1.38147 ms, percentile(90%) = 1.3855 ms, percentile(95%) = 1.3894 ms, percentile(99%) = 2.47192 ms
[09/07/2024-21:22:23] [I] D2H Latency: min = 0.112793 ms, max = 4.37329 ms, mean = 0.121365 ms, median = 0.114746 ms, percentile(90%) = 0.139648 ms, percentile(95%) = 0.154541 ms, percentile(99%) = 0.166626 ms
[09/07/2024-21:22:23] [I] Total Host Walltime: 3.00432 s
[09/07/2024-21:22:23] [I] Total GPU Compute Time: 2.94311 s
[09/07/2024-21:22:23] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[09/07/2024-21:22:23] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[09/07/2024-21:22:23] [W] * GPU compute time is unstable, with coefficient of variance = 26.399%.
[09/07/2024-21:22:23] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[09/07/2024-21:22:23] [I] Explanations of the performance metrics are printed in the verbose logs.
```