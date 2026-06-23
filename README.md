# 6000D Runner

`6000d-runner` 是 6000D 单机 GPU 通信与算力测试的一键执行程序，用于统一运行 P2P、H2D/D2H、NCCL collective、pairwise collective、socket collective、GEMM 以及环境检查流程。

该项目支持两种运行方式：

- 源码模式：使用 `6000d_runner.py` 统一调度各测试脚本。
- 二进制模式：使用 Nuitka 编译生成 `dist/6000d-runner`，交付时无需暴露原始 Python 脚本。
