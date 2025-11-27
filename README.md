# NVLink 实验说明

本实验有两个程序，分别测试nvlink的带宽和延迟。

## 算法简介
- `nvlink_bandwidth_test.cu`: 在 **GPU A** 上执行一个纯读 kernel，直接访问 **GPU B** 的 HBM 并累加大矩阵元素，只写回本地线程级部分和，借此测量跨卡 NVLink 读带宽。
- `nvlink_latency_test.cu`: 构造随机指针追逐链，分别驻留在本地和远端 HBM；单线程使用 `clock64()` 记录串行追逐次数的总周期数，计算平均访问延迟并对比本地/远端结果。

## 编译命令
```bash
nvcc -O1 -arch=sm_90a nvlink_bandwidth_test.cu -o nvlink_bandwidth_test
nvcc -O1 -arch=sm_90a nvlink_latency_test.cu -o nvlink_latency_test
```

请注意根据要运行的GPU架构更改-arch后的参数。例如H系列GPU为sm_90a。

## 运行参数
- `nvlink_bandwidth_test`
  - ./run_bandwidth.sh > out_bandwidth.log
- `nvlink_latency_test`
  - ./run_latency.sh > out_latency.log

