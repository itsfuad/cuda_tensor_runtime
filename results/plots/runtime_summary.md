# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000519 | 0.400001 | 0.001 |
| add | 1024 | 0.001145 | 0.392789 | 0.003 |
| add | 4096 | 0.004120 | 0.395359 | 0.010 |
| add | 16384 | 0.016636 | 0.406547 | 0.041 |
| add | 65536 | 0.059972 | 0.485175 | 0.124 |
| compiled_graph | 64 | 0.186250 | 2.127750 | 0.088 |
| compiled_graph | 128 | 1.275066 | 1.093128 | 1.166 |
| compiled_graph | 256 | 11.061124 | 1.697858 | 6.515 |
| compiled_graph | 512 | 126.526526 | 2.859004 | 44.255 |
| matmul | 64 | 0.168772 | 1.526508 | 0.111 |
| matmul | 128 | 1.281860 | 0.401620 | 3.192 |
| matmul | 256 | 10.970678 | 0.488030 | 22.480 |
| matmul | 512 | 114.105612 | 1.136088 | 100.437 |

## Highlights

- `matmul` at size `512`: CUDA is `100.44x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `44.26x` faster than CPU.
- `matmul` at size `256`: CUDA is `22.48x` faster than CPU.
