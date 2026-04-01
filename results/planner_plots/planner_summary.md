# Planner Benchmark Summary

| workload | size | planner | avg_ms |
| --- | ---: | --- | ---: |
| add | 256 | adaptive | 1.178288 |
| add | 256 | measured | 0.011907 |
| add | 256 | threshold | 0.013853 |
| add | 1024 | adaptive | 0.094792 |
| add | 1024 | measured | 0.012607 |
| add | 1024 | threshold | 0.012870 |
| add | 4096 | adaptive | 0.103212 |
| add | 4096 | measured | 0.528871 |
| add | 4096 | threshold | 0.526183 |
| add | 16384 | adaptive | 0.124050 |
| add | 16384 | measured | 0.553960 |
| add | 16384 | threshold | 0.552967 |
| add | 65536 | adaptive | 0.175546 |
| add | 65536 | measured | 0.657468 |
| add | 65536 | threshold | 0.650447 |
| compiled_graph | 64 | adaptive | 1.499126 |
| compiled_graph | 64 | measured | 2.496700 |
| compiled_graph | 64 | threshold | 2.651436 |
| compiled_graph | 128 | adaptive | 0.927296 |
| compiled_graph | 128 | measured | 1.466950 |
| compiled_graph | 128 | threshold | 1.478604 |
| compiled_graph | 256 | adaptive | 2.708734 |
| compiled_graph | 256 | measured | 2.063886 |
| compiled_graph | 256 | threshold | 1.821896 |
| compiled_graph | 512 | adaptive | 21.249758 |
| compiled_graph | 512 | measured | 3.274218 |
| compiled_graph | 512 | threshold | 3.241324 |
| matmul | 64 | adaptive | 1.281472 |
| matmul | 64 | measured | 0.186370 |
| matmul | 64 | threshold | 1.626806 |
| matmul | 128 | adaptive | 0.680496 |
| matmul | 128 | measured | 1.747874 |
| matmul | 128 | threshold | 0.529448 |
| matmul | 256 | adaptive | 2.361860 |
| matmul | 256 | measured | 0.610044 |
| matmul | 256 | threshold | 0.637072 |
| matmul | 512 | adaptive | 20.387650 |
| matmul | 512 | measured | 1.235552 |
| matmul | 512 | threshold | 1.238400 |

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
