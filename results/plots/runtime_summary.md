# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000614 | 0.518904 | 0.001 |
| add | 1024 | 0.001258 | 0.528874 | 0.002 |
| add | 4096 | 0.003923 | 0.538540 | 0.007 |
| add | 16384 | 0.015735 | 0.553022 | 0.028 |
| add | 65536 | 0.058974 | 0.658556 | 0.090 |
| compiled_graph | 64 | 0.186466 | 2.657970 | 0.070 |
| compiled_graph | 128 | 1.319820 | 1.477208 | 0.893 |
| compiled_graph | 256 | 10.574798 | 1.864864 | 5.671 |
| compiled_graph | 512 | 130.106772 | 3.251646 | 40.013 |
| matmul | 64 | 0.164786 | 1.547938 | 0.106 |
| matmul | 128 | 1.276542 | 0.528880 | 2.414 |
| matmul | 256 | 10.883182 | 0.621290 | 17.517 |
| matmul | 512 | 122.901096 | 1.262790 | 97.325 |

## Highlights

- `matmul` at size `512`: CUDA is `97.33x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `40.01x` faster than CPU.
- `matmul` at size `256`: CUDA is `17.52x` faster than CPU.
