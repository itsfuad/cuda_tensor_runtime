# Planner Benchmark Summary

| workload | size | planner | avg_ms |
| --- | ---: | --- | ---: |
| add | 256 | adaptive | 0.010308 |
| add | 256 | adaptive_cold | 1.069238 |
| add | 256 | measured | 0.012564 |
| add | 256 | threshold | 0.012394 |
| add | 1024 | adaptive | 0.010442 |
| add | 1024 | adaptive_cold | 0.030952 |
| add | 1024 | measured | 0.012835 |
| add | 1024 | threshold | 0.013435 |
| add | 4096 | adaptive | 0.020476 |
| add | 4096 | adaptive_cold | 0.041368 |
| add | 4096 | measured | 0.016771 |
| add | 4096 | threshold | 0.546590 |
| add | 16384 | adaptive | 0.041780 |
| add | 16384 | adaptive_cold | 0.062016 |
| add | 16384 | measured | 0.027048 |
| add | 16384 | threshold | 0.526325 |
| add | 65536 | adaptive | 0.073906 |
| add | 65536 | adaptive_cold | 0.114456 |
| add | 65536 | measured | 0.074517 |
| add | 65536 | threshold | 0.609859 |
| compiled_graph | 64 | adaptive | 0.227054 |
| compiled_graph | 64 | adaptive_cold | 1.324670 |
| compiled_graph | 64 | measured | 2.277588 |
| compiled_graph | 64 | threshold | 2.603356 |
| compiled_graph | 128 | adaptive | 0.583708 |
| compiled_graph | 128 | adaptive_cold | 0.652648 |
| compiled_graph | 128 | measured | 1.515240 |
| compiled_graph | 128 | threshold | 1.480824 |
| compiled_graph | 256 | adaptive | 0.783338 |
| compiled_graph | 256 | adaptive_cold | 1.232578 |
| compiled_graph | 256 | measured | 1.870770 |
| compiled_graph | 256 | threshold | 1.905996 |
| compiled_graph | 512 | adaptive | 1.552970 |
| compiled_graph | 512 | adaptive_cold | 6.413446 |
| compiled_graph | 512 | measured | 3.200332 |
| compiled_graph | 512 | threshold | 3.513830 |
| matmul | 64 | adaptive | 0.174682 |
| matmul | 64 | adaptive_cold | 1.249340 |
| matmul | 64 | measured | 0.184362 |
| matmul | 64 | threshold | 1.661746 |
| matmul | 128 | adaptive | 0.516004 |
| matmul | 128 | adaptive_cold | 0.558132 |
| matmul | 128 | measured | 1.474700 |
| matmul | 128 | threshold | 0.548654 |
| matmul | 256 | adaptive | 0.630134 |
| matmul | 256 | adaptive_cold | 1.052454 |
| matmul | 256 | measured | 0.632566 |
| matmul | 256 | threshold | 0.619750 |
| matmul | 512 | adaptive | 1.171344 |
| matmul | 512 | adaptive_cold | 5.913636 |
| matmul | 512 | measured | 1.226504 |
| matmul | 512 | threshold | 1.239668 |

## Adaptive Backend Mix

| workload | size | cuda_ratio |
| --- | ---: | ---: |
| add | 256 | 0.040 |
| add | 1024 | 0.040 |
| add | 4096 | 0.040 |
| add | 16384 | 0.040 |
| add | 65536 | 0.040 |
| compiled_graph | 64 | 0.040 |
| compiled_graph | 128 | 0.490 |
| compiled_graph | 256 | 0.500 |
| compiled_graph | 512 | 0.500 |
| matmul | 64 | 0.040 |
| matmul | 128 | 0.960 |
| matmul | 256 | 0.960 |
| matmul | 512 | 0.960 |

## Measured Planner vs Baselines

| workload | size | oracle_backend | measured_vs_oracle |
| --- | ---: | --- | ---: |
| add | 256 | cpu | 24.129 |
| add | 1024 | cpu | 10.625 |
| add | 4096 | cpu | 4.336 |
| add | 16384 | cpu | 1.741 |
| add | 65536 | cpu | 1.223 |
| compiled_graph | 64 | cpu | 12.148 |
| compiled_graph | 128 | cpu | 1.173 |
| compiled_graph | 256 | cuda | 0.992 |
| compiled_graph | 512 | cuda | 0.953 |
| matmul | 64 | cpu | 1.090 |
| matmul | 128 | cuda | 2.002 |
| matmul | 256 | cuda | 1.106 |
| matmul | 512 | cuda | 0.969 |

## Warmed Adaptive vs Threshold

| workload | size | adaptive_vs_threshold |
| --- | ---: | ---: |
| add | 256 | 1.202 |
| add | 1024 | 1.287 |
| add | 4096 | 26.694 |
| add | 16384 | 12.598 |
| add | 65536 | 8.252 |
| compiled_graph | 64 | 11.466 |
| compiled_graph | 128 | 2.537 |
| compiled_graph | 256 | 2.433 |
| compiled_graph | 512 | 2.263 |
| matmul | 64 | 9.513 |
| matmul | 128 | 1.063 |
| matmul | 256 | 0.984 |
| matmul | 512 | 1.058 |
