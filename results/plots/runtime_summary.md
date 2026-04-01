# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000618 | 0.416106 | 0.001 |
| add | 1024 | 0.001029 | 0.391178 | 0.003 |
| add | 4096 | 0.003920 | 0.397448 | 0.010 |
| add | 16384 | 0.015620 | 0.404300 | 0.039 |
| add | 65536 | 0.060115 | 0.495045 | 0.121 |
| compiled_graph | 64 | 0.184884 | 2.245968 | 0.082 |
| compiled_graph | 128 | 1.306614 | 1.189244 | 1.099 |
| compiled_graph | 256 | 10.773128 | 1.519000 | 7.092 |
| compiled_graph | 512 | 119.210186 | 2.870614 | 41.528 |
| matmul | 64 | 0.177996 | 1.327242 | 0.134 |
| matmul | 128 | 1.250396 | 0.409182 | 3.056 |
| matmul | 256 | 10.936358 | 0.478956 | 22.834 |
| matmul | 512 | 110.503462 | 1.106592 | 99.859 |

## Highlights

- `matmul` at size `512`: CUDA is `99.86x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `41.53x` faster than CPU.
- `matmul` at size `256`: CUDA is `22.83x` faster than CPU.
