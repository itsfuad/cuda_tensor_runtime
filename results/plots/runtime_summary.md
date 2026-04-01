# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000513 | 0.538972 | 0.001 |
| add | 1024 | 0.001348 | 0.508349 | 0.003 |
| add | 4096 | 0.003845 | 0.529972 | 0.007 |
| add | 16384 | 0.015339 | 0.547743 | 0.028 |
| add | 65536 | 0.059427 | 0.606317 | 0.098 |
| compiled_graph | 64 | 0.197106 | 2.439940 | 0.081 |
| compiled_graph | 128 | 1.299888 | 1.484848 | 0.875 |
| compiled_graph | 256 | 11.083942 | 1.880488 | 5.894 |
| compiled_graph | 512 | 124.610030 | 3.201812 | 38.919 |
| matmul | 64 | 0.165006 | 1.701832 | 0.097 |
| matmul | 128 | 1.284692 | 0.532846 | 2.411 |
| matmul | 256 | 10.601314 | 0.620552 | 17.084 |
| matmul | 512 | 118.262260 | 1.239406 | 95.418 |

## Highlights

- `matmul` at size `512`: CUDA is `95.42x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `38.92x` faster than CPU.
- `matmul` at size `256`: CUDA is `17.08x` faster than CPU.
