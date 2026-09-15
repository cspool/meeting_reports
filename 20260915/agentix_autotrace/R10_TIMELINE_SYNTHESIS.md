# R10 时间线可视化综合报告（workflow05 合同视图 · 三模型 agentix_core 复现 trace）

日期：2026-09-14 · 对象：`{llama,qwen3,qwenvl}_core_g06_g10/` 下的
`HIGH_LATENCY_PROCESS_HARDWARE_TIMELINE.html`（高延迟分堆梯形时间线）与
`CONCURRENCY_UTILIZATION.html`（已关联资源窗口页），分堆计划见各自 `GROUPS.json`。
所有数字均可在 GROUPS.json（含源 sqlite sha256）复核。

## 1. 分堆计划总览（两页共用）

Process 宇宙 = worker 侧 scope 实例；入选阈值 >10 % 时长和分母（重叠计入）；每类型 5 个
非空连续堆（确定性对数时长 Lloyd 聚类），全局按堆时长和排名。

| 模型 | 实例数 | 总时长 | 入选类型 | forward 合计 | preprocess 合计 | 折叠上限(2×中位) | » 折叠点 |
|---|---|---|---|---|---|---|---|
| LLaMA-3.1-8B | 235,220 | 80.7 s | forward / preprocess | 29.5 s | 34.0 s | 4494 µs | 10,358 |
| Qwen3-1.7B | 283,816 | 51.6 s | forward / preprocess | 19.1 s | 21.3 s | 2540 µs | 12,995 |
| Qwen2.5-VL-3B | 257,158 | 63.6 s | forward / preprocess | 20.5 s | 30.8 s | 3817 µs | 11,095 |

三个模型入选类型完全一致，且**全局第 1 名堆在三个模型上都是同一个堆**：`forward` 的
pile4（最长时长簇）——LLaMA 24.85 s / 1,186 成员，Qwen3 16.44 s / 1,392，VL 16.80 s / 1,313。

## 2. 高延迟页读到什么

- **高延迟集中在 forward 的最重两堆，且是"少数长实例吃掉时间"的形态。** LLaMA 的
  forward-pile4 仅占 forward 实例的 12 %（1,186/10,026）却贡献 24.85/29.5 = 84 % 的
  forward 时长，且 **成员 100 % 被高延迟注记**（类型阈值 = 中位+3×MAD = 657 µs）；
  pile3 亦 96 % HL（1,346/1,407）。Qwen3/VL 同构（pile4 全 HL）。这些是 MLFQ 连续批
  下的**大 batch/含 prefill 的 forward 步**，对应 w01' 并发分区里 ≥8 并发桶占 GPU 74.6 %
  的同一批窗口。
- **preprocess 时长和更大但形态相反**：LLaMA preprocess 合计 34.0 s > forward 29.5 s，
  然而分散在 5,145–1,905 成员的中位堆里，HL 成员只有 pile4 的 206/1,905（阈值 4,827 µs）。
  这是 host 侧调度/输入准备的"宽而浅"占用——单实例不构成尾延迟，但作为分母项它解释了
  为何 GPU busy（72 %）低于墙钟：host 步进本身消耗了可观的每步时间（与 w04'' 并发条件化
  步模型的截距项一致）。
- **时间分布（十段制）**：LLaMA 最密段是第 9/10 段——高延迟 forward 成员在**运行末段堆积**，
  与 MLFQ 的行为吻合：β 反饥饿把长程序在尾段提回 Q0，长上下文 prefill 连续段（chunked
  continuation 重提交）集中出现在负载末端；两个 Qwen 的最密段在第 3/10 段（其 r0.3 负载
  峰值在前段，模型小、排队早消化）。折叠时钟压掉了 1 万+ 个空闲间隔，说明选中实例之间
  存在大量 >2×中位时长的空洞——即**排队/等待主导的间隙**，与 cap16 排队制的设计一致。

## 3. 资源窗口页读到什么（仅已关联窗口）

关联是 family 级：NCU 对 batched-GEMM 家族重放的中位数挂到含 gemm kernel 的成员窗口，
不做逐实例归属（页面已声明）。

- **可显示的堆几乎只剩 forward 重堆**：LLaMA forward-pile4 有 821/1,186 成员含 gemm
  kernel（69 %），Qwen3 886/1,392、VL 839/1,313；而 preprocess 各堆关联数为 0–141
  （host-only 窗口），按合同隐藏但保留原始排名（#2–#4、#6、#10 在隐藏清单中列出）。
  **资源页因此把"值得看硬件的窗口"自动收敛到大 batch forward 堆**——这正是合同"只显示
  成功关联窗口"要达到的筛选效果。
- **关联窗口的硬件形态**：L2 ≈ 76 %、SM/tensor ≈ 49 %、DRAM 仅 14–20 %（w03' NCU 中位）。
  高延迟堆的墙在 **L2/tensor**，不在 DRAM——与 h22 单请求 eager 的"DRAM 受限 decode
  gemv"结论相反，且三个模型一致：连续批把 decode 维持成 GEMM 形。

## 4. 综合判定

1. **三模型的时间线拓扑同构**（同一入选类型、同一全局第 1 堆、同一 HL 形态），说明形态由
   **agentix_core 机制 + 连续批**决定，而非模型规模；模型规模只改尺度（折叠上限 4494/2540/
   3817 µs 即中位步长之比）。
2. **尾延迟的机器侧根因**是少数大 batch/prefill forward 步（每模型 ~1.2–1.4 k 个实例、
   占 forward 时长 84 %+），其资源墙是 L2/tensor；**降尾方向是切小这些步**（更小的
   prefill chunk / 长 prefill 降并发），而不是提内存带宽。
3. **host 侧 preprocess 是最大的"隐形分母"**（时长和超过 forward），resource 页无法关联
   任何硬件指标恰好暴露它是纯 host 开销——对小模型（Qwen3：preprocess 21.3 s vs
   forward 19.1 s）这已是第一优化对象，与 h22 的"Python dispatch 瓶颈"结论在 serving
   语境下的对应物。
4. 与 w04'' 交叉验证：并发条件化步模型三模型误差 −4.48 % / −2.38 % / −8.32 %，其并发桶
   划分与本页 forward 重堆的时间位置一致，两条独立链互证。

## 5. 记账与边界

- 每堆每段最多绘制 60 条成员线，全量成员在 GROUPS.json 记账；折叠时钟与十段制意味着页面
  不声称全量无损时间轴。
- 硬件关联为 family 级中位数（NCU 重放），非逐实例 PMC；preprocess 堆的"无关联"是证据性
  结论（窗口内无 gemm kernel），不是缺测。
- 高延迟阈值为类型内中位+3×MAD 注记，仅标色不参与分堆。
