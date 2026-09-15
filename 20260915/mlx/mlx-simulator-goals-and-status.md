# MLX 模拟器：最终目标说明与当前进度（2026-09-15）

依据：MLX 论文（ISCA 2026，`paper_secs/paper_isca26_full`）、课题组前作（Wu et al., Multilayer Dataflow, ISCA 2025）、`ref_arch/`（DFU3500：执行模型、SIMD/Tensor 指令、
GEMM 分块层次、DMA 双缓冲、算子模板与编译链、专题 7 的 n-loop × 模块时间线）。证据入口：台账 `docs/mlx-paper-performance-reproduction.md`、计划 `docs/mlx-gpgpu-toolchain-scheduling-gap-plan.md` §0.11。

四个最终目标，每个给出：定义与验收标准 → 论文/ref_arch 依据 → 当前进度 → 差距与下一步。

---

## 目标 1：性能加速比复现（由机制与架构保证）

**定义**。论文的加速比 = T_base / T_MLX。基线机器（Jetson Xavier NX、Orin、H100）本地不可得，基线只能用 4090 的 SM 绑定 + 资源模型折算，且执行方式在论文里有歧义；
因此加速比复现拆成两个可独立验收的部分：

1. **MLX 侧（机制与架构）**：模拟器给出的 T_MLX 与论文隐含的 T_MLX（= 算法算子数 / 论文报告的 roofline 利用率）在 10% 内。等价于复现论文 Fig. 22（计算利用率）、
   Fig. 23（8×8 扩展）、Fig. 25（利用率随 S）的每一格，以及 Fig. 20/21 用论文利用率反推的 MLX 时间。**这一部分与基线无关，是"模拟器实现了论文机制"的判据。**
2. **基线侧（读数）**：同一个 T_MLX 配不同基线给出不同加速比：论文推断基线（→ 论文数字）、4090 6-SM 折算 NX（Tensor Core / CUDA-core 两组假设）。报告时两种读数并列，不把基线假设当结论。

**验收标准**：Fig. 22 16 格、Fig. 23 15 格、Fig. 25 24 格、Fig. 20 8 格、Fig. 21 5 点，逐格 |误差| ≤ 10%（趋势类另加 Spearman ≥ 0.7）；不允许按目标调参（参数只能来自论文/前作/物理项目）。

**论文机制清单与实现状态**（每项 = 一个可换实现的模块或剖面变量；`docs/mlx-tagged-core-modules.md`）：

| 论文机制 | 论文位置 | 模拟器实现 | 状态 |
|---|---|---|---|
| 解耦四流水 PE（load/store/compute/xfer）、结果缓冲、写回端口、旁路 | Fig. 9c | `Pipeline`、`result_buffer`、`writeback_ports`、`bypass`、`ComputeUnit` | 完成，Fig. 22 16/16 |
| CDC（闭合依赖集）与常量驻留寄存器、块迭代（trip） | §IV、Fig. 10 | `Block.constants/persistent`、`trip_count`、`store_overlap` | 完成 |
| 标签仲裁的多上下文（4/8 在飞）、按流水线发射 | Fig. 9c、§V | `IssueGate`（7+1 种停顿原因）、`context_inflight`、`issue_per_pipeline` | 完成 |
| skip-hop 网格（±1/±2 跳）、接收队列 | §V、Table II | `Mesh`（skip_hop/unit_hop）、`receive_queue` | 完成，四类跳全部出现 |
| 每 PE SPM 端口（行向量×拍）、阵列级带宽/请求率/容量、DRAM 溢出 | Fig. 9、Fig. 23 | `ScratchpadPort`（shared/per_pe/queued + DebugPort）、`spm_bandwidth/request_rate/capacity`、`dram_*` | 完成，Fig. 23 14/15 |
| 多层折叠执行：跨层数据经 xfer 直达对端寄存器，不回全局内存 | §IV Fig. 7/8 | `layer`/上下文槽、xfer 邮箱寄存器；FFT 7 层、BSMM 6 层单 CDC | 完成（核内）；跨算子层折叠（QKV→FFT→注意力同驻）未做 |
| launch / CDC 配置 / 数据供应（论文：launch 开销 17%→12%，roofline 按算力与带宽归一） | §VII Fig. 22/25 | `LaunchModel`（none/cdc/cdc_dma 跨 CDC 预取）、多 CDC（wave）内核 | 第二十二轮新增；BSMM 小 S 进入 10%（B32 512/1K/8K），稳态小 S 仍高 20% |
| SRAM I/O 重排（FFT 级间） | §V | 未计时（程序 A/B 之间的重排为零代价） | 缺 |
| SWA（滑窗注意力）与压缩注意力 | Fig. 24/25 | 注意力五阶段内核（scores/max/exp/PV/scale，广播操作数、持久累加器、全 lane exp） | 完成；SWA 用窗口=键数近似 |
| 稠密 GEMM 折叠（8×8 tile、psum 沿前向传播） | §IV | 输出驻留 GEMM（K 块常量、bcast），无 psum 传播 | 部分 |

**ref_arch 的作用**：论文没写的部分按 DFU3500 的已知实现补：Instance 基址表 ↔ `trip_count/stride`；APP/Task/Subtask ↔ launch/CDC/块；DMA 双通道双缓冲 ↔ `cdc_dma` 跨 CDC 预取；
专题 7 的 n-loop × 模块时间线 ↔ `scripts/tagged_timeline.py` + `docs/mlx-operator-nloop-timelines.md`（区别：条来自模拟器事件而非估算）。

**当前进度（数字）**：

| 图 | 判据 | 当前 | 备注 |
|---|---|---|---|
| Fig. 22 计算利用率 | 16 格 | **16/16** | v6 剖面 |
| Fig. 23 8×8 扩展 | 15 格 | **14/15** | bw16 + req16 + 2 MB + DRAM 6 |
| Fig. 25 利用率-S | 24 格 | v6：9/24；v8（wave+DMA）：BSMM 12 格 **7/12**，FFT/SWA 扫描进行中 | 剩余：稳态小 S 高 20%、FFT 稳态 0.46 vs 0.58–0.84（算子口径）、SWA 与 S 无关 |
| Fig. 20 八内核（论文推断基线） | 8 格 | **8K 四格 4/4**，256 四格 0/4（快 1.3–1.5×，v8 待重跑） | 4090-NX 基线读数：TC 0/8、CUDA-FP32 投影类同量级 |
| Fig. 21 端到端（论文推断基线） | 5 点 | 0/5（MLX 快 1.5×，v8 待重跑）；内存条 8/10 | 论文 N 趋势另含 GPU 侧 N 无关项（NX 16 GB 换页假设） |
| Fig. 17/24（H100/Orin） | — | 未做（同法：4090 全卡 / 16-SM 折算） | |
| Fig. 18/19（先验加速器） | 引用比对 | 256 GOp/s 缩减设计可直接比对 | 未整理 |

**差距与下一步**：(1) 等 v8 扫描出来重跑 Fig. 20/21；(2) 每 CDC 剩余 ≈250 周期暴露量：DRAM 延迟（Fig. 23 剖面已用 100）、Y 回写、单通道带宽，逐项加；(3) FFT 算子口径（复数 FMA 计 4 op）与级间重排计时；
(4) 三项 ISA/调度候选（跨迭代 load-ahead、exp.acc、累加器末迭代 store）——注意这些会让 MLX 更快，只在论文机制要求时采用。

---

## 目标 2：系统级模拟正确性验证（含指令集与编译）

**定义**。同一负载在四个层级给出逐位相同的数值与相同的核心周期：数值仿真（FP16 逐步舍入合约）= 独立核心（C++ 周期模拟器）= 时钟设备（驱动级 RoCC 设备 + 系统剖面）
= Chipyard（RISC-V 主机 + RoCC + DPI 设备 + DRAM，Verilator）。指令集与编译是验证对象的一部分：程序由编译器/模板生成，经 wire 描述符（ISA 的线格式）下发，设备端解析后执行。

**验收标准**：每个论文内核（BSMM B16/32/64、FFT-CMP、注意力五阶段、GEMM、8×8 网格）在四层级逐位一致、周期一致；系统剖面（`tagged_options`）由设备端核心自己的解析器校验并回显；
模块级改动（端口/网格/计算单元/launch）在设备上重验。

**指令集**（`simulator_ext/tagged`，`system_sim/physical_device/tagged_wire.*`）：10 条 opcode（load/store/mul/add/fma/max/exp/div/xfer + 本地读），四流水线；wire v3：几何随核心上限、步长 ±32767、
local/bcast 位、常量段与持久段、绑定输入/输出地址（DRAM arena）；编码器在程序装得下 v2 剖面时仍发 v2（stage1 字节不变）。ref_arch 对应：DFU 的 HLDT/HSTT/COPY/HMMA 与 Instance 基址表 → 我们的 load/store/xfer/fma 与 trip 步长；
DFU 的 CSV 指令 + 图映射 + 控制 bin → 我们的 program.json + wire 描述符 + 系统剖面。

**编译**：算子模板生成器（`fig22/23/25_kernels.py`、`fig21_kernels.py`：CDC 模板、权重驻留、unroll、多 CDC wave、注意力五阶段、输出驻留 GEMM）+ 镜像构建（`paper_image.py`、`attention_image.py`：19 次 launch 经 DRAM arena 串接）。
不是通用编译器：没有从模型图到 MLX 程序的自动 lowering（计划 §1 的 LLVM/MLIR + 空间映射路线未开始）；模型级 C++ 张量执行器（Llama2-7B 6181 调用、logits 逐位）是功能语义执行，不经 MLX 指令。

**当前进度**：

| 层级 | 状态 | 证据 |
|---|---|---|
| 数值仿真 = 独立核心 | 全部论文内核逐位 | `fig25-cycle-runtime-reproduction.json` `all_bitwise`，`fig21-kernels-utilization.json` |
| 独立核心 = 时钟设备（驱动级） | BSMM 4×4/8×8、注意力五阶段（广播/持久/全 lane exp）、模块追踪逐周期相同、launch 模型 4-CDC 内核 | `tests/test_tagged_system_v3.py` 9 项通过 |
| = Chipyard（Verilator，宽内存 ELF 预载） | 论文 BSMM 镜像核心周期 966/1152/1152 = 独立核心；注意力 19 launch 12416 行逐位、周期相同 | `chipyard-paper-profile-v3-evidence.json`、`chipyard-attention-chain-v3-evidence.json` |
| 剖面校验 | 未知键/不一致组合被设备端拒绝并回显有效剖面 | `test_tagged_options_are_validated_and_echoed` |
| 回归 | 黄金周期 12 项（3 s）、系统 v3 9 项（≈14 min，含设备重建） | |

**差距与下一步**：(1) v8（launch 模型）的 Chipyard 论文镜像重跑（驱动级已过）；(2) SWA 内核与完整结构化 transformer block 镜像（QKV-BSMM → FFT-CMP → 注意力 → 输出 → FFN 同一 arena 串接）；
(3) 通用编译：至少把 Llama2 一层的算子图自动 lowering 到现有模板（模板参数化已有，缺图级调度）；(4) decode 路径（KV 增长、块 BSMM）未做。

---

## 目标 3：周期级事件性能模拟（核心）与误差控制

**定义**。核心模拟器按周期推进，每条指令的 issue/complete/route/receive/wait/wake 与每周期每模块占用都是事件；四个模块接口（端口/发射门/网格/计算单元）+ launch 模型可换实现、可注入调试激励；
时间线工具把事件画成 ref_arch 专题 7 的四张图并与 n-loop 行对照。误差控制的对象：(a) 与论文报告的利用率（目标 1 的格）；(b) 与 RTL 的周期（前作把模拟器校到 RTL 7% 内）。

**当前进度**：事件与模块追踪完成；`tagged_timeline.py`（A/A'/B/C）与六个内核的实测时间线；黄金周期回归保证重构不改周期；模块替换/注入测试通过；n-loop 文档定位了三项利用率损失。
RTL：`rtl/mlx` 只有单 PE 控制/共享 RF 前端通过组件检查，**核心周期尚未对 RTL 校准**（前作 7%）——这是目标 3 的主要缺口；周期校准前，"周期 = 论文周期"只能靠论文报告的利用率间接检验。

**下一步**：单 PE RTL 与核心的逐周期对照（先 load/compute/store 三流水，再 xfer），把 RTL 差异作为误差项进台账。

---

## 目标 4：端到端系统级性能模拟与误差控制

**定义**。Chipyard 路径给出的不只是核心周期，还有主机 launch、描述符取回、绑定输入装入、结果沉积、DRAM 排队——即论文 Fig. 21 端到端所需的系统项。误差控制：端到端时间分解为
核心计算（目标 3 控制）+ 数据供应/launch（目标 1 的 launch 模型 + 系统路径实测）+ 基线（假设集合）；每项单独报告误差来源。

**当前进度**：Chipyard 端到端 PASS 的同时也暴露了系统路径的数据供应远慢于论文：BSMM 镜像取描述符 148k 周期、结果沉积 58k 周期 vs 核心 966（TileLink 8 B/周期串行、描述符内嵌 SPM 镜像），
即当前 Chipyard 内存路径不代表论文的 DMA/DRAM 系统，只能用于正确性；端到端性能靠核心周期 + launch/DMA 模型（`cdc_dma`，前作 2×25.6 GB/s）。Fig. 21 的 MLX 侧时间 = Σ 层算子数 / 实测利用率（v8 待重跑）。
基线侧：6-SM 4090 端到端实测（真实权重、profiler 按类拆分）+ 按类折算 NX；论文隐含的 GPU 侧 N 无关项（14–47 s）不可在 4090 上复现。

**下一步**：(1) 系统路径的 DMA 通道模型（宽 DMA 取代 TileLink 逐字取回；描述符只带绑定地址），使 Chipyard 的供应时间与 `cdc_dma` 同量级——这样端到端性能也能在 Chipyard 上直接测；
(2) 多 launch 的 arena 串接已验证，把 launch 间隔计入；(3) 误差报告格式固定为三项分解。

---

## 总表

| 目标 | 完成度 | 主要证据 | 主要缺口 |
|---|---|---|---|
| 1 加速比复现（机制/架构） | Fig. 22 16/16、Fig. 23 14/15、Fig. 25 BSMM 7/12（v8）、Fig. 20 8K 4/4；Fig. 21 0/5 | 台账 §3–§6、§3.3.5–3.3.8 | 小 S 稳态 +20%、FFT 口径、Fig. 20/21 v8 重跑 |
| 2 系统级正确性（ISA/编译） | 四层级逐位/周期一致（BSMM、注意力链、8×8、launch 模型驱动级） | Chipyard 两份证据、系统 v3 测试 | 完整 block 镜像、通用 lowering、decode |
| 3 周期级事件模拟 | 模块化核心 + 事件/模块追踪 + 时间线 + 黄金回归 | `tagged_timeline.py`、`timelines/` | RTL 周期校准未做 |
| 4 端到端系统级性能 | 核心 + launch/DMA 模型可算；Chipyard 供应路径仅供正确性 | Chipyard 证据、`gpu-baseline-e2e` | Chipyard DMA 通道模型、误差三项分解 |
