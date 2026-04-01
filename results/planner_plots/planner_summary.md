# Planner Benchmark Summary

| workload | size | planner | avg_ms |
| --- | ---: | --- | ---: |
| add | 256 | adaptive | 0.010428 |
| add | 256 | adaptive_cold | 1.149174 |
| add | 256 | measured | 0.012950 |
| add | 256 | threshold | 0.012458 |
| add | 1024 | adaptive | 0.021890 |
| add | 1024 | adaptive_cold | 0.030922 |
| add | 1024 | measured | 0.013352 |
| add | 1024 | threshold | 0.013232 |
| add | 4096 | adaptive | 0.020922 |
| add | 4096 | adaptive_cold | 0.052156 |
| add | 4096 | measured | 0.017168 |
| add | 4096 | threshold | 0.542232 |
| add | 16384 | adaptive | 0.041342 |
| add | 16384 | adaptive_cold | 0.061894 |
| add | 16384 | measured | 0.027967 |
| add | 16384 | threshold | 0.533245 |
| add | 65536 | adaptive | 0.083334 |
| add | 65536 | adaptive_cold | 0.105740 |
| add | 65536 | measured | 0.074329 |
| add | 65536 | threshold | 0.619852 |
| compiled_graph | 64 | adaptive | 0.205906 |
| compiled_graph | 64 | adaptive_cold | 1.301288 |
| compiled_graph | 64 | measured | 2.201200 |
| compiled_graph | 64 | threshold | 2.842886 |
| compiled_graph | 128 | adaptive | 0.577520 |
| compiled_graph | 128 | adaptive_cold | 0.649488 |
| compiled_graph | 128 | measured | 1.478410 |
| compiled_graph | 128 | threshold | 1.556704 |
| compiled_graph | 256 | adaptive | 0.782780 |
| compiled_graph | 256 | adaptive_cold | 1.241814 |
| compiled_graph | 256 | measured | 1.835730 |
| compiled_graph | 256 | threshold | 1.934986 |
| compiled_graph | 512 | adaptive | 1.634264 |
| compiled_graph | 512 | adaptive_cold | 6.444582 |
| compiled_graph | 512 | measured | 3.463700 |
| compiled_graph | 512 | threshold | 3.348918 |
| matmul | 64 | adaptive | 0.177002 |
| matmul | 64 | adaptive_cold | 1.139968 |
| matmul | 64 | measured | 0.175572 |
| matmul | 64 | threshold | 1.726868 |
| matmul | 128 | adaptive | 0.655594 |
| matmul | 128 | adaptive_cold | 0.554992 |
| matmul | 128 | measured | 1.563492 |
| matmul | 128 | threshold | 0.561908 |
| matmul | 256 | adaptive | 0.651676 |
| matmul | 256 | adaptive_cold | 1.078494 |
| matmul | 256 | measured | 0.793434 |
| matmul | 256 | threshold | 0.623896 |
| matmul | 512 | adaptive | 1.210344 |
| matmul | 512 | adaptive_cold | 5.816658 |
| matmul | 512 | measured | 1.371052 |
| matmul | 512 | threshold | 1.265220 |

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
| matmul | 128 | 0.960 |
| matmul | 256 | 0.960 |
| matmul | 512 | 0.960 |

## Measured Planner vs Baselines

| workload | size | oracle_backend | measured_vs_oracle |
| --- | ---: | --- | ---: |
| add | 256 | cpu | 25.240 |
| add | 1024 | cpu | 9.902 |
| add | 4096 | cpu | 4.465 |
| add | 16384 | cpu | 1.823 |
| add | 65536 | cpu | 1.251 |
| compiled_graph | 64 | cpu | 11.168 |
| compiled_graph | 128 | cpu | 1.137 |
| compiled_graph | 256 | cuda | 0.976 |
| compiled_graph | 512 | cuda | 1.082 |
| matmul | 64 | cpu | 1.064 |
| matmul | 128 | cuda | 2.934 |
| matmul | 256 | cuda | 1.279 |
| matmul | 512 | cuda | 1.106 |

## Warmed Adaptive vs Threshold

| workload | size | adaptive_vs_threshold |
| --- | ---: | ---: |
| add | 256 | 1.195 |
| add | 1024 | 0.604 |
| add | 4096 | 25.917 |
| add | 16384 | 12.898 |
| add | 65536 | 7.438 |
| compiled_graph | 64 | 13.807 |
| compiled_graph | 128 | 2.695 |
| compiled_graph | 256 | 2.472 |
| compiled_graph | 512 | 2.049 |
| matmul | 64 | 9.756 |
| matmul | 128 | 0.857 |
| matmul | 256 | 0.957 |
| matmul | 512 | 1.045 |
