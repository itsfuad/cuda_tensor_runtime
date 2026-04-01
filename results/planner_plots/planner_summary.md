# Planner Benchmark Summary

| workload | size | planner | avg_ms |
| --- | ---: | --- | ---: |
| add | 256 | adaptive | 0.010350 |
| add | 256 | adaptive_cold | 1.133716 |
| add | 256 | measured | 0.012024 |
| add | 256 | threshold | 0.012054 |
| add | 1024 | adaptive | 0.020784 |
| add | 1024 | adaptive_cold | 0.042180 |
| add | 1024 | measured | 0.012658 |
| add | 1024 | threshold | 0.012817 |
| add | 4096 | adaptive | 0.031310 |
| add | 4096 | adaptive_cold | 0.042354 |
| add | 4096 | measured | 0.016073 |
| add | 4096 | threshold | 0.528846 |
| add | 16384 | adaptive | 0.030804 |
| add | 16384 | adaptive_cold | 0.062240 |
| add | 16384 | measured | 0.026719 |
| add | 16384 | threshold | 0.543851 |
| add | 65536 | adaptive | 0.083106 |
| add | 65536 | adaptive_cold | 0.113468 |
| add | 65536 | measured | 0.073866 |
| add | 65536 | threshold | 0.640352 |
| compiled_graph | 64 | adaptive | 0.217446 |
| compiled_graph | 64 | adaptive_cold | 1.414126 |
| compiled_graph | 64 | measured | 2.267512 |
| compiled_graph | 64 | threshold | 2.545544 |
| compiled_graph | 128 | adaptive | 0.569178 |
| compiled_graph | 128 | adaptive_cold | 0.677866 |
| compiled_graph | 128 | measured | 1.484236 |
| compiled_graph | 128 | threshold | 1.471400 |
| compiled_graph | 256 | adaptive | 0.808762 |
| compiled_graph | 256 | adaptive_cold | 1.221152 |
| compiled_graph | 256 | measured | 1.880318 |
| compiled_graph | 256 | threshold | 1.833794 |
| compiled_graph | 512 | adaptive | 1.542390 |
| compiled_graph | 512 | adaptive_cold | 6.380542 |
| compiled_graph | 512 | measured | 3.220566 |
| compiled_graph | 512 | threshold | 3.392938 |
| matmul | 64 | adaptive | 0.175592 |
| matmul | 64 | adaptive_cold | 1.313858 |
| matmul | 64 | measured | 0.175598 |
| matmul | 64 | threshold | 1.531386 |
| matmul | 128 | adaptive | 0.533160 |
| matmul | 128 | adaptive_cold | 0.660796 |
| matmul | 128 | measured | 1.922884 |
| matmul | 128 | threshold | 0.568880 |
| matmul | 256 | adaptive | 0.621450 |
| matmul | 256 | adaptive_cold | 1.033162 |
| matmul | 256 | measured | 0.594450 |
| matmul | 256 | threshold | 0.625196 |
| matmul | 512 | adaptive | 1.220130 |
| matmul | 512 | adaptive_cold | 5.686418 |
| matmul | 512 | measured | 1.243966 |
| matmul | 512 | threshold | 1.240514 |

## Adaptive Backend Mix

| workload | size | cuda_ratio |
| --- | ---: | ---: |
| add | 256 | 0.040 |
| add | 1024 | 0.040 |
| add | 4096 | 0.040 |
| add | 16384 | 0.040 |
| add | 65536 | 0.040 |
| compiled_graph | 64 | 0.040 |
| compiled_graph | 128 | 0.500 |
| compiled_graph | 256 | 0.500 |
| compiled_graph | 512 | 0.500 |
| matmul | 64 | 0.040 |
| matmul | 128 | 0.880 |
| matmul | 256 | 0.960 |
| matmul | 512 | 0.960 |

## Measured Planner vs Baselines

| workload | size | oracle_backend | measured_vs_oracle |
| --- | ---: | --- | ---: |
| add | 256 | cpu | 19.645 |
| add | 1024 | cpu | 11.269 |
| add | 4096 | cpu | 3.881 |
| add | 16384 | cpu | 1.739 |
| add | 65536 | cpu | 1.230 |
| compiled_graph | 64 | cpu | 11.137 |
| compiled_graph | 128 | cpu | 1.134 |
| compiled_graph | 256 | cuda | 0.964 |
| compiled_graph | 512 | cuda | 0.989 |
| matmul | 64 | cpu | 1.068 |
| matmul | 128 | cuda | 3.659 |
| matmul | 256 | cuda | 0.958 |
| matmul | 512 | cuda | 0.972 |

## Warmed Adaptive vs Threshold

| workload | size | adaptive_vs_threshold |
| --- | ---: | ---: |
| add | 256 | 1.165 |
| add | 1024 | 0.617 |
| add | 4096 | 16.891 |
| add | 16384 | 17.655 |
| add | 65536 | 7.705 |
| compiled_graph | 64 | 11.707 |
| compiled_graph | 128 | 2.585 |
| compiled_graph | 256 | 2.267 |
| compiled_graph | 512 | 2.200 |
| matmul | 64 | 8.721 |
| matmul | 128 | 1.067 |
| matmul | 256 | 1.006 |
| matmul | 512 | 1.017 |
