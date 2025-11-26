# NVLink 实验说明

## 算法简介
- `nvlink_peer_sum.cu`: 在 **GPU A** 上执行一个纯读 kernel，直接访问 **GPU B** 的 HBM 并累加大矩阵元素，只写回本地线程级部分和，借此测量跨卡 NVLink 读带宽。
- `nvlink_latency_test.cu`: 构造随机指针追逐链，分别驻留在本地和远端 HBM；单线程使用 `clock64()` 记录串行追逐次数的总周期数，计算平均访问延迟并对比本地/远端结果。

## 编译命令
```bash
cd /home/yunxuan/test-nvlink
nvcc -O1 -arch=sm_90a nvlink_peer_sum.cu -o nvlink_peer_sum
nvcc -O1 -arch=sm_90a nvlink_latency_test.cu -o nvlink_latency_test
```
如需同时兼容 V100/H100，可将 `-arch=sm_70` 替换为 `-gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90`。

## 运行参数
- `nvlink_peer_sum`
  - `rows cols`（可选）：矩阵尺寸，默认 `16384 16384`。
  - 使用示例：`./nvlink_peer_sum 32768 32768`
- `nvlink_latency_test`
  - `num_nodes`：指针链节点数，默认 `1<<20`。
  - `iterations`：单次试验追逐次数，默认 `1<<22`。
  - `trials`：试验重复次数，默认 `10`。
  - `compute_dev`：执行 kernel 的 GPU，默认 `0`。
  - `remote_dev`：存放远端链表的 GPU，默认 `1`。
  - 使用示例：`./nvlink_latency_test 1048576 4194304 10 0 1`

