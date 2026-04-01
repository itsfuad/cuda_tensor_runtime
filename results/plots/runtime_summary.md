# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000612 | 0.535818 | 0.001 |
| add | 1024 | 0.001123 | 0.520464 | 0.002 |
| add | 4096 | 0.004142 | 0.534859 | 0.008 |
| add | 16384 | 0.015365 | 0.533819 | 0.029 |
| add | 65536 | 0.060037 | 0.621207 | 0.097 |
| compiled_graph | 64 | 0.203610 | 2.543008 | 0.080 |
| compiled_graph | 128 | 1.308572 | 1.462186 | 0.895 |
| compiled_graph | 256 | 10.929284 | 1.949850 | 5.605 |
| compiled_graph | 512 | 125.967056 | 3.256380 | 38.683 |
| matmul | 64 | 0.164446 | 1.644316 | 0.100 |
| matmul | 128 | 1.285662 | 0.525456 | 2.447 |
| matmul | 256 | 10.472746 | 0.620348 | 16.882 |
| matmul | 512 | 156.000968 | 1.279798 | 121.895 |

## Highlights

- `matmul` at size `512`: CUDA is `121.89x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `38.68x` faster than CPU.
- `matmul` at size `256`: CUDA is `16.88x` faster than CPU.
