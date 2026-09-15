# R10 消融报告 — LLaMA-3.1-8B（agentix_core vs fcfs baseline，workflow05 合同视图）

日期：2026-09-14 · GPU 1 · 负载 `thr_mixed_r0.3.json`（sha256/seed 冻结，与复现运行同一份）·
唯一变量 `--policy`（agentix_core = 程序级 MLFQ per Algorithm 1；fcfs = vLLM 原生连续批）。
其余全同：kv-offload 8 GB、V1 runner + NVTX scopes、nsys 2026.3.1、同卡同引擎参数。

视图对：
- core：`llama_core_g06_g10/HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html` + `CONCURRENCY_UTILIZATION.html`
- baseline：`llama_fcfs_g06_g10/…`（同名两页）；分堆计划各自 `GROUPS.json`（含源 sqlite sha256）。

## 1. 端侧指标（run summary）

| 指标 | agentix_core | fcfs | 比值 |
|---|---|---|---|
| program token latency mean (ms) | 17.35 | 17.07 | 1.017× |
| p90 (ms) | 29.09 | 28.81 | 1.010× |
| p99 (ms) | 29.69 | 29.33 | 1.012× |
| program response p90 (s) | 100.3 | 99.8 | — |
| wall (s) / calls | 188 / 1629 | 188 / 1629 | — |

**r0.3 非排队制下 core≈fcfs**（差异 <3 %）。这与 h23 主结论一致：程序级调度只在排队制
有杠杆（LLaMA cap16 排队制下 PLAS 0.72× mean / 0.57× p90，见 results.md §2）；本消融
固定非排队制，用于隔离**机制本身的 trace 形态代价**。

## 2. 分堆计划对照（两侧同一政策：>10 % 入选、5 堆 Lloyd、全局排名）

| 全局# | 类型/堆 | core Σ(s) | n | HL | assoc | fcfs Σ(s) | n | HL | assoc | ΔΣ(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| #1 | forward/p4 | 24.85 | 1186 | 1186 | 821 | 23.48 | 1139 | 1139 | 792 | +1.37 |
| #2 | preprocess/p3 | 18.55 | 5145 | 0 | 15 | 18.19 | 5058 | 0 | 14 | +0.36 |
| #3 | preprocess/p4 | 8.42 | 1905 | 206 | 2 | 8.13 | 1785 | 209 | 3 | +0.29 |
| #4 | preprocess/p2 | 4.77 | 1817 | 0 | 107 | 4.78 | 1837 | 0 | 90 | -0.01 |
| #5 | forward/p2 | 2.75 | 5441 | 0 | 9 | 1.79 | 3138 | 0 | 5 | +0.96 |
| #6 | preprocess/p1 | 2.24 | 1162 | 0 | 39 | 2.54 | 1357 | 0 | 25 | -0.31 |
| #7 | forward/p3 | 1.15 | 1407 | 1346 | 2 | 1.20 | 1326 | 902 | 4 | -0.05 |
| #8 | forward/p1 | 0.49 | 1210 | 0 | 5 | 1.55 | 3283 | 0 | 3 | -1.06 |
| #9 | forward/p0 | 0.22 | 782 | 0 | 7 | 0.34 | 1147 | 0 | 5 | -0.12 |
| #10 | preprocess/p0 | 0.03 | 62 | 0 | 0 | 0.03 | 59 | 0 | 0 | +0.00 |

实例数 core 235,220 vs fcfs 235,400；折叠上限 4494 vs 4244 µs；
» 折叠点 10,358 vs 11,029。

## 3. 高延迟页读数（消融差异）

- **堆拓扑同构**：两侧入选类型相同（forward / preprocess），全局第 1 名都是 forward 最长
  时长堆（pile4），成员 100 % 高延迟注记——尾延迟的机器侧形态不因调度策略而变。
- **机制签名**：core 的重 forward 堆比 fcfs 多 **+1.37 s / +47 成员**
  （forward 类型合计 29.5 vs 28.4 s）。这是 MLFQ 量子切块的 chunked continuation：
  量子用尽的调用重提交（前缀缓存重命中但仍需调度步进与增量 prefill），把额外质量堆进最重
  forward 堆。r0.3 下这是纯开销（端侧无收益）；排队制下它是换取抢占能力的成本。
- **preprocess（host 侧分母）**：core 34.0 vs fcfs 33.7 s——host 开销对策略不敏感，
  仍是最大"隐形分母"。

## 4. 资源窗口页读数

- 两侧可关联堆同为 forward 重堆（core assoc 821 vs fcfs 792 成员含 gemm）；
  preprocess 各堆关联≈0，按合同隐藏、保留排名。
- family 级 NCU 中位数（`ncu_focus/focus2_raw.csv`，采于同引擎 GEMM 家族，两侧共用并在页面声明）：
  L2 ≈76 %、SM/tensor ≈49 %、DRAM 14–20 %——资源墙在 L2/tensor，与策略无关。

## 5. 消融结论

1. 在 r0.3 非排队制，agentix_core 机制对端侧延迟**中性**（≤3 %），trace 侧唯一可检出差异
   是重 forward 堆的 +1.4 s continuation 质量与折叠空洞数变化。
2. 尾延迟根因（少数大 batch/prefill forward 步、L2/tensor 墙）在两种策略下同构——**优化
   这些步（更小 prefill chunk、长 prefill 降并发）对两侧同等有效**，且不依赖调度策略。
3. 机制的收益端不在本消融覆盖内：需要排队制（cap16）才可见（LLaMA 0.72×/0.57×）。

## 6. 记账

每堆每段最多绘 60 条成员线（全量在 GROUPS.json）；折叠时钟不声称无损；HL 阈值为类型
中位+3×MAD 注记；硬件关联 family 级非逐实例。baseline 采集命令与 core 逐 token 相同仅换
policy，可从两侧 `profile.log` 首行与 run summary 复核。
