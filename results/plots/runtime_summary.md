# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000620 | 0.536403 | 0.001 |
| add | 1024 | 0.001143 | 0.512228 | 0.002 |
| add | 4096 | 0.004179 | 0.513793 | 0.008 |
| add | 16384 | 0.020070 | 0.505672 | 0.040 |
| add | 65536 | 0.077373 | 0.608118 | 0.127 |
| compiled_graph | 64 | 0.192162 | 2.745010 | 0.070 |
| compiled_graph | 128 | 1.261614 | 1.564644 | 0.806 |
| compiled_graph | 256 | 10.977058 | 2.019040 | 5.437 |
| compiled_graph | 512 | 120.541706 | 3.368334 | 35.787 |
| matmul | 64 | 0.168696 | 1.726596 | 0.098 |
| matmul | 128 | 1.292440 | 0.537446 | 2.405 |
| matmul | 256 | 11.153386 | 0.630682 | 17.685 |
| matmul | 512 | 120.869002 | 1.255598 | 96.264 |

## Highlights

- `matmul` at size `512`: CUDA is `96.26x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `35.79x` faster than CPU.
- `matmul` at size `256`: CUDA is `17.68x` faster than CPU.
