# Planner Benchmark Summary

| workload | size | planner | avg_ms |
| --- | ---: | --- | ---: |
| add | 256 | adaptive | 1.123622 |
| add | 256 | measured | 0.012467 |
| add | 256 | threshold | 0.012670 |
| add | 1024 | adaptive | 0.103048 |
| add | 1024 | measured | 0.012886 |
| add | 1024 | threshold | 0.012837 |
| add | 4096 | adaptive | 0.113660 |
| add | 4096 | measured | 0.527202 |
| add | 4096 | threshold | 0.548936 |
| add | 16384 | adaptive | 0.127256 |
| add | 16384 | measured | 0.523863 |
| add | 16384 | threshold | 0.536420 |
| add | 65536 | adaptive | 0.177438 |
| add | 65536 | measured | 0.601921 |
| add | 65536 | threshold | 0.618675 |
| compiled_graph | 64 | adaptive | 1.572282 |
| compiled_graph | 64 | measured | 2.316314 |
| compiled_graph | 64 | threshold | 2.452710 |
| compiled_graph | 128 | adaptive | 0.866910 |
| compiled_graph | 128 | measured | 1.627290 |
| compiled_graph | 128 | threshold | 1.645706 |
| compiled_graph | 256 | adaptive | 2.602962 |
| compiled_graph | 256 | measured | 1.948182 |
| compiled_graph | 256 | threshold | 1.815586 |
| compiled_graph | 512 | adaptive | 21.501282 |
| compiled_graph | 512 | measured | 3.301862 |
| compiled_graph | 512 | threshold | 3.225250 |
| matmul | 64 | adaptive | 1.330642 |
| matmul | 64 | measured | 0.195318 |
| matmul | 64 | threshold | 1.740066 |
| matmul | 128 | adaptive | 0.680382 |
| matmul | 128 | measured | 1.619728 |
| matmul | 128 | threshold | 0.531052 |
| matmul | 256 | adaptive | 2.318700 |
| matmul | 256 | measured | 0.606076 |
| matmul | 256 | threshold | 0.860282 |
| matmul | 512 | adaptive | 20.675590 |
| matmul | 512 | measured | 1.220902 |
| matmul | 512 | threshold | 1.220288 |

## Adaptive Backend Mix

| workload | size | cuda_ratio |
| --- | ---: | ---: |
| add | 256 | 0.160 |
| add | 1024 | 0.160 |
| add | 4096 | 0.160 |
| add | 16384 | 0.160 |
| add | 65536 | 0.160 |
| compiled_graph | 64 | 0.160 |
| compiled_graph | 128 | 0.500 |
| compiled_graph | 256 | 0.500 |
| compiled_graph | 512 | 0.500 |
| matmul | 64 | 0.160 |
| matmul | 128 | 0.840 |
| matmul | 256 | 0.840 |
| matmul | 512 | 0.840 |

## Measured Planner vs Baselines

| workload | size | oracle_backend | measured_vs_oracle |
| --- | ---: | --- | ---: |
| add | 256 | cpu | 20.107 |
| add | 1024 | cpu | 11.273 |
| add | 4096 | cpu | 126.149 |
| add | 16384 | cpu | 26.102 |
| add | 65536 | cpu | 7.779 |
| compiled_graph | 64 | cpu | 12.054 |
| compiled_graph | 128 | cpu | 1.290 |
| compiled_graph | 256 | cuda | 0.965 |
| compiled_graph | 512 | cuda | 0.980 |
| matmul | 64 | cpu | 1.158 |
| matmul | 128 | cuda | 3.014 |
| matmul | 256 | cuda | 0.961 |
| matmul | 512 | cuda | 0.972 |
