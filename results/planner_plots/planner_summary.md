# Planner Benchmark Summary

| workload | size | planner | avg_ms |
| --- | ---: | --- | ---: |
| add | 256 | adaptive | 0.021558 |
| add | 256 | adaptive_cold | 1.086982 |
| add | 256 | measured | 0.012216 |
| add | 256 | threshold | 0.012430 |
| add | 1024 | adaptive | 0.020754 |
| add | 1024 | adaptive_cold | 0.041196 |
| add | 1024 | measured | 0.012734 |
| add | 1024 | threshold | 0.013422 |
| add | 4096 | adaptive | 0.020698 |
| add | 4096 | adaptive_cold | 0.041516 |
| add | 4096 | measured | 0.016478 |
| add | 4096 | threshold | 0.550944 |
| add | 16384 | adaptive | 0.041354 |
| add | 16384 | adaptive_cold | 0.062350 |
| add | 16384 | measured | 0.027432 |
| add | 16384 | threshold | 0.562302 |
| add | 65536 | adaptive | 0.082876 |
| add | 65536 | adaptive_cold | 0.104770 |
| add | 65536 | measured | 0.076252 |
| add | 65536 | threshold | 0.653792 |
| compiled_graph | 64 | adaptive | 0.197360 |
| compiled_graph | 64 | adaptive_cold | 1.273030 |
| compiled_graph | 64 | measured | 2.144526 |
| compiled_graph | 64 | threshold | 2.434770 |
| compiled_graph | 128 | adaptive | 0.595434 |
| compiled_graph | 128 | adaptive_cold | 0.656670 |
| compiled_graph | 128 | measured | 1.735216 |
| compiled_graph | 128 | threshold | 1.505968 |
| compiled_graph | 256 | adaptive | 0.795708 |
| compiled_graph | 256 | adaptive_cold | 1.254126 |
| compiled_graph | 256 | measured | 1.882120 |
| compiled_graph | 256 | threshold | 1.931836 |
| compiled_graph | 512 | adaptive | 1.774644 |
| compiled_graph | 512 | adaptive_cold | 6.433400 |
| compiled_graph | 512 | measured | 3.549462 |
| compiled_graph | 512 | threshold | 3.252204 |
| matmul | 64 | adaptive | 0.176548 |
| matmul | 64 | adaptive_cold | 1.294726 |
| matmul | 64 | measured | 0.194506 |
| matmul | 64 | threshold | 1.737280 |
| matmul | 128 | adaptive | 0.568102 |
| matmul | 128 | adaptive_cold | 0.725242 |
| matmul | 128 | measured | 1.609224 |
| matmul | 128 | threshold | 0.563536 |
| matmul | 256 | adaptive | 0.666922 |
| matmul | 256 | adaptive_cold | 1.114740 |
| matmul | 256 | measured | 0.636622 |
| matmul | 256 | threshold | 0.697982 |
| matmul | 512 | adaptive | 1.271486 |
| matmul | 512 | adaptive_cold | 6.080898 |
| matmul | 512 | measured | 1.277164 |
| matmul | 512 | threshold | 1.261974 |

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
| matmul | 128 | 0.860 |
| matmul | 256 | 0.960 |
| matmul | 512 | 0.960 |

## Measured Planner vs Baselines

| workload | size | oracle_backend | measured_vs_oracle |
| --- | ---: | --- | ---: |
| add | 256 | cpu | 23.283 |
| add | 1024 | cpu | 11.162 |
| add | 4096 | cpu | 3.680 |
| add | 16384 | cpu | 1.779 |
| add | 65536 | cpu | 1.249 |
| compiled_graph | 64 | cpu | 10.434 |
| compiled_graph | 128 | cpu | 1.363 |
| compiled_graph | 256 | cuda | 0.985 |
| compiled_graph | 512 | cuda | 1.029 |
| matmul | 64 | cpu | 1.179 |
| matmul | 128 | cuda | 2.658 |
| matmul | 256 | cuda | 1.009 |
| matmul | 512 | cuda | 0.884 |

## Warmed Adaptive vs Threshold

| workload | size | adaptive_vs_threshold |
| --- | ---: | ---: |
| add | 256 | 0.577 |
| add | 1024 | 0.647 |
| add | 4096 | 26.618 |
| add | 16384 | 13.597 |
| add | 65536 | 7.889 |
| compiled_graph | 64 | 12.337 |
| compiled_graph | 128 | 2.529 |
| compiled_graph | 256 | 2.428 |
| compiled_graph | 512 | 1.833 |
| matmul | 64 | 9.840 |
| matmul | 128 | 0.992 |
| matmul | 256 | 1.047 |
| matmul | 512 | 0.993 |
