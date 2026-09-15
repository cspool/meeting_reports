# R10 消融报告 — Qwen3-1.7B（agentix_core vs fcfs baseline，workflow05 合同视图）

日期：2026-09-14 · GPU 0 · 负载 `thr_mixed_r0.3.json`（sha256/seed 冻结，与复现运行同一份）·
唯一变量 `--policy`（agentix_core = 程序级 MLFQ per Algorithm 1；fcfs = vLLM 原生连续批）。
其余全同：kv-offload 8 GB、V1 runner + NVTX scopes、nsys 2026.3.1、同卡同引擎参数。

视图对：
- core：`qwen3_core_g06_g10/HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html` + `CONCURRENCY_UTILIZATION.html`
- baseline：`qwen3_fcfs_g06_g10/…`（同名两页）；分堆计划各自 `GROUPS.json`（含源 sqlite sha256）。

## 1. 端侧指标（run summary）

| 指标 | agentix_core | fcfs | 比值 |
|---|---|---|---|
| program token latency mean (ms) | 7.66 | 7.45 | 1.028× |
| p90 (ms) | 14.75 | 14.54 | 1.015× |
| p99 (ms) | 15.48 | 15.17 | 1.020× |
| program response p90 (s) | 37.5 | 37.1 | — |
| wall (s) / calls | 87 / 1629 | 86 / 1629 | — |

**r0.3 非排队制下 core≈fcfs**（差异 <3 %）。这与 h23 主结论一致：程序级调度只在排队制
有杠杆（LLaMA cap16 排队制下 PLAS 0.72× mean / 0.57× p90，见 results.md §2）；本消融
固定非排队制，用于隔离**机制本身的 trace 形态代价**。

## 2. 分堆计划对照（两侧同一政策：>10 % 入选、5 堆 Lloyd、全局排名）

| 全局# | 类型/堆 | core Σ(s) | n | HL | assoc | fcfs Σ(s) | n | HL | assoc | ΔΣ(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| #1 | forward/p4 | 16.44 | 1392 | 1392 | 886 | 14.81 | 1242 | 1242 | 791 | +1.63 |
| #2 | preprocess/p2 | 8.38 | 5552 | 0 | 1 | 9.74 | 6377 | 0 | 3 | -1.36 |
| #3 | preprocess/p1 | 6.62 | 5075 | 0 | 1 | 5.74 | 4364 | 0 | 0 | +0.88 |
| #4 | preprocess/p3 | 5.12 | 2976 | 288 | 5 | 5.22 | 2963 | 483 | 1 | -0.10 |
| #5 | forward/p1 | 1.33 | 6344 | 0 | 2 | 1.34 | 6377 | 0 | 1 | -0.00 |
| #6 | preprocess/p4 | 1.15 | 483 | 483 | 9 | 1.17 | 465 | 465 | 18 | -0.02 |
| #7 | forward/p0 | 0.89 | 4862 | 0 | 0 | 0.87 | 4744 | 0 | 0 | +0.02 |
| #8 | forward/p2 | 0.33 | 1325 | 255 | 0 | 0.38 | 1570 | 305 | 6 | -0.06 |
| #9 | forward/p3 | 0.13 | 159 | 159 | 1 | 0.17 | 233 | 233 | 6 | -0.04 |
| #10 | preprocess/p0 | 0.01 | 69 | 0 | 0 | 0.01 | 57 | 0 | 0 | +0.00 |

实例数 core 283,816 vs fcfs 284,979；折叠上限 2540 vs 2556 µs；
» 折叠点 12,995 vs 13,059。

## 3. 高延迟页读数（消融差异）

- **堆拓扑同构**：两侧入选类型相同（forward / preprocess），全局第 1 名都是 forward 最长
  时长堆（pile4），成员 100 % 高延迟注记——尾延迟的机器侧形态不因调度策略而变。
- **机制签名**：core 的重 forward 堆比 fcfs 多 **+1.63 s / +150 成员**
  （forward 类型合计 19.1 vs 17.6 s）。这是 MLFQ 量子切块的 chunked continuation：
  量子用尽的调用重提交（前缀缓存重命中但仍需调度步进与增量 prefill），把额外质量堆进最重
  forward 堆。r0.3 下这是纯开销（端侧无收益）；排队制下它是换取抢占能力的成本。
- **preprocess（host 侧分母）**：core 21.3 vs fcfs 21.9 s——host 开销对策略不敏感，
  仍是最大"隐形分母"。

## 4. 资源窗口页读数

- 两侧可关联堆同为 forward 重堆（core assoc 886 vs fcfs 791 成员含 gemm）；
  preprocess 各堆关联≈0，按合同隐藏、保留排名。
- family 级 NCU 中位数（`ncu_qwen3/focus_raw.csv`，采于同引擎 GEMM 家族，两侧共用并在页面声明）：
  L2 ≈76 %、SM/tensor ≈49 %、DRAM 14–20 %——资源墙在 L2/tensor，与策略无关。

## 5. 消融结论

1. 在 r0.3 非排队制，agentix_core 机制对端侧延迟**中性**（≤3 %），trace 侧唯一可检出差异
   是重 forward 堆的 +1.6 s continuation 质量与折叠空洞数变化。
2. 尾延迟根因（少数大 batch/prefill forward 步、L2/tensor 墙）在两种策略下同构——**优化
   这些步（更小 prefill chunk、长 prefill 降并发）对两侧同等有效**，且不依赖调度策略。
3. 机制的收益端不在本消融覆盖内：需要排队制（cap16）才可见（LLaMA 0.72×/0.57×）。

## 6. 记账

每堆每段最多绘 60 条成员线（全量在 GROUPS.json）；折叠时钟不声称无损；HL 阈值为类型
中位+3×MAD 注记；硬件关联 family 级非逐实例。baseline 采集命令与 core 逐 token 相同仅换
policy，可从两侧 `profile.log` 首行与 run summary 复核。
