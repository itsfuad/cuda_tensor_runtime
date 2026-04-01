# Runtime Benchmark Summary

| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |
| --- | ---: | ---: | ---: | ---: |
| add | 256 | 0.000521 | 0.527923 | 0.001 |
| add | 1024 | 0.001208 | 0.508444 | 0.002 |
| add | 4096 | 0.003868 | 0.518011 | 0.007 |
| add | 16384 | 0.015538 | 0.524918 | 0.030 |
| add | 65536 | 0.060919 | 0.601804 | 0.101 |
| compiled_graph | 64 | 0.187484 | 2.611296 | 0.072 |
| compiled_graph | 128 | 1.291864 | 1.488696 | 0.868 |
| compiled_graph | 256 | 10.945924 | 1.886144 | 5.803 |
| compiled_graph | 512 | 123.506324 | 3.359236 | 36.766 |
| matmul | 64 | 0.169148 | 1.691784 | 0.100 |
| matmul | 128 | 1.248330 | 0.736754 | 1.694 |
| matmul | 256 | 11.492128 | 0.572160 | 20.086 |
| matmul | 512 | 115.292576 | 1.265140 | 91.130 |

## Highlights

- `matmul` at size `512`: CUDA is `91.13x` faster than CPU.
- `compiled_graph` at size `512`: CUDA is `36.77x` faster than CPU.
- `matmul` at size `256`: CUDA is `20.09x` faster than CPU.
