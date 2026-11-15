# 输入、源码、运行条件与复核

## 1. 源码身份

基线 Git commit 是 `2c961072e0232ab1a41f85be481bb2f8876f6fc5`。
本分支包含内存与 communicator 改进、可关闭的诊断、固定势重建和 smart 下界候选。
27 个改动源码/测试文件的 SHA256 均与实际 GPU 发布包一致，见
[provenance.json](data/provenance.json)。

历史 `2c96107+memory-…`、`+qr-…`、`+restartfix-…`、`+restartbounds-…` 是
冻结源码快照的标签，不是可以 `git checkout` 的 Git commit。
最新测试包 SHA256 为
`9f332b0cb1b01756953c44d7f0940e01cf695aeacd4776975cfc7227ae5dd6c2`；
GB200 runtime SHA256 为
`59067e1e373f0a5e5114cc47c0ad6edcf708ac13ff3eae2248d241fb5ed8b4b8`。
不要把本分支所有改动同时开启后当作每一个历史版本。

| 阶段 | 内容 | 验证边界 |
|---|---|---|
| base | Git 基线 | 原有基准与 H100 无对称性 control |
| memory-v1 | 稠密临时量复用、sector communicator 等 | 完整 SCF 显存下降；时间 trade-off 待进一步分离 |
| memory-v2 | MPI Ritz 单分配布局复用 | 模块精度、容量与短 SCF 验证 |
| qr candidate | robust QR 的单分配候选 | 模块及短 SCF 通过，大体系验证未完成 |
| restart levels | 固定势额外重建 | 原精度门槛，最多 40 轮 |
| restart smart | 额外重建使用 smart 下界 | 95,648 电子重启收敛；匹配 A/B 与数组审查仍开放 |

## 2. 结构和输入

历史结构生成过程见 [structures/README.md](../structures/README.md)。
更大的 cluster 沿用 diamond 截球、补共享 dangling sites、H 饱和的生成器。
以下只生成结构/输入，不运行 SCF：

```bash
python examples/0d_nanodiamonds/structures/generate_nanodiamonds.py \
  --output-root /tmp/parsec_generated_clusters \
  --maximum-carbon-atoms 31000 --maximum-multiplier 21

python examples/3d_diamond_benchmark/make_diamond.py \
  13 13 13 /tmp/parsec_gamma13.in
python examples/3d_diamond_benchmark/make_diamond.py \
  10 10 10 /tmp/parsec_k2_10.in --kmesh 2 2 2
```

重新生成的默认 solver controls 不应被当成历史原件。精确复核请使用本目录
[inputs](inputs/) 中冻结的 `parsec.in`，并检查 [inputs.json](data/inputs.json)。
cluster k11 的表中 H100 对照通过 launcher 选择 `--symmetry off`；输入自身的
`Ignore_Symmetry: false` 不足以确定运行模式，必须同时记录 launcher 设置。

赝势直接使用仓库已有文件，不额外复制：

| 用途 | 文件 | SHA256 |
|---|---|---|
| cluster C | `examples/0d_nanodiamonds/C_POTRE.DAT` | `2e073cb81f66327a15d82a3955b8822c72f13494d31ec51799b2b3c122598c8c` |
| cluster H | `examples/0d_nanodiamonds/H_POTRE.DAT` | `202df2aba56d7aa99a515909add249b643cde27f1aff67bc79b4ceeebf2af191` |
| 周期 PBE C | `examples/3d_diamond_benchmark/C_POTRE.DAT` | `d6d64ac88cd8a72e6593f5cd8be7e78841d891abab9f758852e3727117844784` |

## 3. 模式开关

所有 MPI ranks 必须得到相同设置。安装运行依赖的方法沿用仓库 README；
site launcher 必须做到一 rank 一 GPU，不能让 64 个进程使用同一张卡。

| 变量 | 本次用途 | 注意事项 |
|---|---|---|
| `PARSEC_MPI_SHARED_RITZ=inplace` | 重用列/行基组分配 | `rounds`、`rows` control 仍保留，库 `auto` 默认不变 |
| `PARSEC_MPI_SHARED_QR=inplace` | QR 候选 | 默认 `rows`；只验证了实数 shared-basis 路径 |
| `PARSEC_DENSE_BUFFER_BYTES=67108864` | 分块 scratch | 每个缓冲的限额，不是 GPU 总峰值 |
| `PARSEC_MPI_SHARED_TRANSPORT=host` | 本次 host-staged MPI | cuda 路径需要独立环境与数值验证 |
| `PARSEC_DEBUG=off/summary/audit` | 可关闭的诊断 | audit 有额外 Hq、reduction、同步开销 |
| `PARSEC_DEBUG_RANKS=leaders` | sector/pool leader 日志 | 保存完整日志时记录各 leader |
| `PARSEC_RESTART_REFINE=off/levels` | 固定势重建 | 默认 off；必须确实经过 `load_restart` |
| `PARSEC_RESTART_REFINE_MAX_PASSES=40` | 有界重建 | 失败即停止，不接受未通过的密度 |
| `PARSEC_RESTART_REFINE_LOWER_BOUND=inherit/smart` | 额外重建下界 | 默认 inherit，原 SCF 精度不变 |

对同样已分配的 GPU，fresh cluster/Γ 计算的可移植入口是
`parsec_python.acceleration.benchmarks.mpi_full_scf`。例如在一个支持 Slurm GPU
绑定的 16×4 GB200 分配中，从仓库根目录运行：

```bash
export PYTHONPATH="$PWD/src"
export PARSEC_MPI_SHARED_RITZ=inplace
export PARSEC_MPI_SHARED_QR=inplace
export PARSEC_DENSE_BUFFER_BYTES=67108864
export PARSEC_MPI_SHARED_TRANSPORT=host
export PARSEC_DEBUG=summary

srun --nodes=16 --ntasks-per-node=4 --gpus-per-task=1 \
  python -m parsec_python.acceleration.benchmarks.mpi_full_scf \
  --input examples/0d_nanodiamonds/large_system_experiments/inputs/cluster/k19_C23191H2884/parsec.in \
  --output-dir /tmp/parsec_k19_run --devices 0 --symmetry on \
  --pp-dir examples/0d_nanodiamonds
```

此命令是 site-dependent 的启动示例，不是本次 TaskKit 提单命令，也不是完整
性能复现配置。生产计时应同时记录 native/GPU backend、BLAS/OpenMP threads、
CPU binding、CUDA-aware MPI、缓存位置和 checkpoint I/O。实际试验冻结了这些设置。
复杂 k 点的 MPI 入口见
[3d_diamond_benchmark](../../3d_diamond_benchmark/README.md)。

**密度/势 checkpoint 重启不能写成 `--restart-from parsec.dat`。**该 benchmark 的
`--restart-from` 用于坐标/几何重启。本次 diagnostic adapter 在准备好相同 system
后调用 `SCF.restart.load_restart`，检查父文件哈希、网格、输入、GPU/runtime 身份，
把得到的 `SCFWarmStart` 显式传给 SCF。API 的核心形式为：

```python
from parsec_python.SCF.restart import load_restart

fields, warm = load_restart(checkpoint_path, prepared_system)
assert not (fields.extrapolate or fields.missing_points or fields.dropped_points)
# Pass warm_start=warm to the matching prepared-system SCF driver.
# The MPI adapter must broadcast the same problem and policy to all ranks.
```

在环境里开启 `levels` 但没有经过这个恢复入口，不会凭空触发重建。
本目录不分发 GB 级父 checkpoint；它的 SHA256、原始科学输入和运行时身份可用于
核对本地持有的原文件。公开的独立 CPU fixtures 可在没有这些 checkpoint 的情况下运行。

## 4. 验证命令与已执行结果

发布前在与 GPU 冻结源码相同的 CPU 环境运行了 12 个模块，共 **161 项测试，
134 通过，27 跳过，0 失败**。模块列表和原始 unittest 末尾输出已保存在
[provenance.json](data/provenance.json)。这些跳过项不能算成 GPU 通过。

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m unittest \
  parsec_python.acceleration.tests.test_dense_memory \
  parsec_python.acceleration.tests.test_ritz_dense \
  parsec_python.acceleration.tests.test_mpi_shared_basis \
  parsec_python.acceleration.tests.test_mpi_shared_qr \
  parsec_python.acceleration.tests.test_mpi_scf \
  parsec_python.acceleration.tests.test_kpoint_pools \
  parsec_python.acceleration.tests.test_column_ritz \
  parsec_python.acceleration.tests.test_debug_audit \
  parsec_python.tests.unit.test_restart_refinement \
  parsec_python.tests.unit.test_diagnostics \
  parsec_python.tests.unit.test_strict_scf_eigensolver \
  parsec_python.tests.unit.test_parsec_dat_io

python examples/0d_nanodiamonds/large_system_experiments/checks/check_cpu_restart_bounds.py
python examples/0d_nanodiamonds/large_system_experiments/checks/check_cpu_kpoint_restart.py
python examples/0d_nanodiamonds/large_system_experiments/checks/validate_report.py
```

独立谱 fixture 为了比较收敛工作允许 80 轮；GPU 候选一直是 40 轮。
这两个不同上限已写入 JSON，不能把 CPU 的工作比直接当成 GPU 实验结果。

GPU 模块检查保留在 `acceleration/benchmarks/`：`check_dense_memory`、
`check_mpi_inplace_memory`、`check_mpi_shared_qr`、`check_debug_audit`。
已有 GPU 结果记录的是实际跑过的规模和 dtype，不表示所有开关组合已测试。

## 5. 抢占恢复与时间口径

科学程序与调度控制器分离。每次分配使用不可变源码/运行时/输入，提交前检查当前
资源与其他 pending 任务；有歧义的提交 intent 不重放。发生数值或身份检查失败时
停止对应路径，不能因为“资源可抢占”就自动重试错误计算。

恢复只使用已校验的健康父状态。诊断 fork 不把自己的输出自动替代原父 checkpoint。
当前重建中间态没有序列化，因此分配中断会丢失已经做完的重建轮数，这也是后续
优化 checkpoint 的动机。每次分配最多 8 小时，重提次数有上限；本轮原控制与新
候选共享既有 51 次分配预算，新候选从中划出 3 次，没有新增总预算。

`wall_seconds` 是一条运行的 launch-to-finish 时间；`reported_program_seconds`
是程序自身报告的时间；`restart_level_refinement.seconds` 只计额外重建。
队列、此前中断工作和跨尝试累计成本另算。不能把恢复后的一个短片段当成整个
从头计算的总耗时，或把诊断 residual 审计的开销忽略掉。

## 6. 图表和记录复核

```bash
python examples/0d_nanodiamonds/large_system_experiments/checks/plot_results.py
python examples/0d_nanodiamonds/large_system_experiments/checks/validate_report.py
```

图表只读本目录筛选后的 CSV/JSON，使用 Matplotlib。报告校验检查输入与源码哈希、
价电子数、关键比较的数值，以及本地 Markdown 链接。完整精度验收仍需要真正的
SCF 输出数组；报告校验不是量子力学计算的替代。
