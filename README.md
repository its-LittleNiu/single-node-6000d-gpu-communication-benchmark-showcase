# 6000d-runner 客户使用说明

`6000d-runner` 用于一键执行 6000D GPU 通信与算力测试。程序会自动按顺序运行环境检查、P2P 带宽、Host/GPU 带宽、NCCL 集合通信、两两 GPU 通信、socket 维度通信汇总和 GEMM 算力测试。

## 1. 运行方式

进入 `6000d-runner` 所在目录：

```bash
cd /path/to/6000d-runner-dir
chmod +x 6000d-runner
./6000d-runner
如文件在 dist 目录：
cd /path/to/project
chmod +x dist/6000d-runner
./dist/6000d-runner

## 2. 程序执行内容


默认执行以下流程：
1. 关闭 PCIe ACSCtl
2. 检查系统、驱动、CUDA、Docker、Python、存储等环境信息
3. 运行 CUDA p2pBandwidthLatencyTest，提取 P2P enabled bandwidth matrix
4. 运行 nvbandwidth，测试 Host to Device / Device to Host 带宽
5. 运行 nccl-tests，测试 2/4/8 卡 collective 通信
6. 运行 nccl-tests，测试两两 GPU broadcast / all-reduce
7. 运行 nccl-tests，测试 socket 内 / 跨 socket collective 通信
8. 通过 Docker 运行 GEMM 算力测试
## 3. 常用运行参数


跳过关闭 ACSCtl：
./6000d-runner --skip-acs
跳过环境检查：
./6000d-runner --skip-env-check
跳过 GEMM 算力测试：
./6000d-runner --skip-gemm
某一步失败后继续执行后续测试：
./6000d-runner --continue-on-error
设置每个测试步骤之间的等待时间，默认 2 秒：
./6000d-runner --sleep 5
## 4. GEMM 参数传递
-- 后面的参数会传给 GEMM 测试。
示例：
./6000d-runner -- \
  --dtype fp8 \
  --fp8-format e4m3 \
  --m 12032 \
  --n 12288 \
  --k 12288 \
  --warmup 10000 \
  --iters 1000 \
  --rounds 20 \
  --threshold 70
含义：
--dtype        数据类型，可选 fp8 / bf16 / fp16 / fp32
--fp8-format   FP8 格式，可选 e4m3 / e5m2
--m            GEMM 矩阵 M
--n            GEMM 矩阵 N
--k            GEMM 矩阵 K
--warmup       预热次数
--iters        每轮计时迭代次数
--rounds       测试轮数
--threshold    TFLOPS 阈值，低于该值会被标记
常用示例：
./6000d-runner -- \
  --dtype fp16 \
  --m 8192 \
  --n 24576 \
  --k 12288 \
  --warmup 20 \
  --iters 100 \
  --rounds 20
跳过环境检查和 ACSCtl，只按指定参数运行后续测试：
./6000d-runner \
  --skip-acs \
  --skip-env-check \
  -- \
  --dtype fp8 \
  --fp8-format e4m3 \
  --m 12032 \
  --n 12288 \
  --k 12288 \
  --warmup 10000 \
  --iters 1000 \
  --rounds 20 \
  --threshold 70
## 5. 输出文件
总日志会生成在当前运行目录：
run_all.log
各测试结果默认生成在：
<hostname>_nccl_result/
例如主机名为 node01，则结果目录为：
node01_nccl_result/
常见输出目录：
node01_nccl_result/
  host_device_bandwidth/
  pairwise_collective/
  socket_collective/
  gemm_benchmark/
其中：
host_device_bandwidth/      Host to Device / Device to Host 带宽日志和 CSV 汇总
pairwise_collective/        两两 GPU broadcast / all-reduce 日志、矩阵和 CSV 汇总
socket_collective/          socket 维度 collective 日志和 CSV 汇总
gemm_benchmark/             GEMM 算力测试日志
## 6. 默认依赖路径
程序会自动搜索常见路径。建议按以下默认目录安装依赖。
6.1 cuda-samples
用于 P2P 测试：
~/cuda-samples/
程序会搜索以下位置：
~/cuda-samples/build/cpp/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
~/cuda-samples/build/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
~/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
6.2 nvbandwidth
用于 Host/GPU 带宽测试：
~/nvbandwidth/build/nvbandwidth
6.3 nccl-tests
用于 NCCL collective、pairwise 和 socket 测试：
~/nccl-tests/build/
该目录下应包含：
all_reduce_perf
broadcast_perf
all_gather_perf
reduce_scatter_perf
reduce_perf
alltoall_perf
程序也会尝试搜索：
/root/nccl-tests/build/
当前目录上级路径中的 nccl-tests/build/
各用户 home 目录下的 nccl-tests/build/
## 7. Docker 镜像要求
GEMM 测试默认使用 NVIDIA PyTorch 镜像：
nvcr.io/nvidia/pytorch:25.05-py3
运行前请确认镜像已经存在：
docker image inspect nvcr.io/nvidia/pytorch:25.05-py3
如果没有镜像，需要提前拉取：
docker pull nvcr.io/nvidia/pytorch:25.05-py3
也可以指定其他镜像：
IMAGE=nvcr.io/nvidia/pytorch:25.05-py3 ./6000d-runner
## 8. 常用环境变量
指定 GEMM 测试的数据类型列表：
DTYPE_LIST="bf16 fp16" ./6000d-runner
指定 Docker 镜像：
IMAGE=nvcr.io/nvidia/pytorch:25.05-py3 ./6000d-runner
指定 GEMM 使用的进程数：
NPROC_PER_NODE=8 ./6000d-runner
指定 master port：
MASTER_PORT=4201 ./6000d-runner
指定结果目录：
HOST_RESULT_ROOT=/data/6000d_result ./6000d-runner
## 9. 推荐检查命令
运行前建议确认 GPU 状态：
nvidia-smi
确认 Docker 可用：
docker --version
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.05-py3 nvidia-smi
确认 nccl-tests：
ls ~/nccl-tests/build/
确认 nvbandwidth：
ls ~/nvbandwidth/build/nvbandwidth
确认 cuda-samples：
find ~/cuda-samples -name p2pBandwidthLatencyTest
## 10. 示例完整命令
默认完整测试：
./6000d-runner
跳过 GEMM，只做通信测试：
./6000d-runner --skip-gemm
失败后继续跑完整流程：
./6000d-runner --continue-on-error
指定 GEMM 参数：
./6000d-runner -- \
  --dtype fp8 \
  --fp8-format e4m3 \
  --m 12032 \
  --n 12288 \
  --k 12288 \
  --warmup 10000 \
  --iters 1000 \
  --rounds 20 \
  --threshold 70
指定结果目录并跳过 ACSCtl：
HOST_RESULT_ROOT=/data/6000d_result ./6000d-runner --skip-acs
