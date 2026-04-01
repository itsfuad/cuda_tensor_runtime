# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000525 | 0.547704 | 0.001 |
| add | 1024 | 0.001141 | 0.544134 | 0.002 |
| add | 4096 | 0.004477 | 0.540020 | 0.008 |
| add | 16384 | 0.015418 | 0.561654 | 0.027 |
| add | 65536 | 0.061074 | 0.652463 | 0.094 |
| compiled_graph | 64 | 0.205530 | 2.597126 | 0.079 |
| compiled_graph | 128 | 1.272806 | 1.578170 | 0.807 |
| compiled_graph | 256 | 11.083180 | 1.910466 | 5.801 |
| compiled_graph | 512 | 128.673652 | 3.448996 | 37.308 |
| matmul | 64 | 0.164922 | 1.677832 | 0.098 |
| matmul | 128 | 1.298256 | 0.605442 | 2.144 |
| matmul | 256 | 10.466214 | 0.630752 | 16.593 |
| matmul | 512 | 122.246872 | 1.444018 | 84.657 |

## Highlights

- `matmul` at size `512`: CUDA is `84.66x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `37.31x` faster than CPU.
- `matmul` at size `256`: CUDA is `16.59x` faster than CPU.
