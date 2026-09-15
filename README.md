记录

auto-trace， 算子生成， 编译优化， rsl model自我调优。model级别优化， 细粒度报告能直接作用于优化（算子）。粗粒度报告能做什么，快速自我调优，需要映射到process，快速定位process优化。offline能否消除online插庄的开销，精度？

模型自我提升的baseline比较差，gap，paper。

process粒度，runtime， 引擎。指导model优化代码！！！！

agent负载的特点。





空间架构的核心意义：权重保存到PE，核心是数据复用。PE局部存储。最大的数据固定，输入输出流动。侧重process并行在PE之间。

SM之间隔离，竞争， L2cache hit。SM内pipeline并行。

模拟器难创新，trace去预测已有架构的性能，修改runtime/软件模块后的性能预测，不要硬件模拟器。

硬件模拟器是为了优化硬件。性能trace是为了优化软件。用trace去做实验，GPU实际执行，返回软件设计，GPU结果反馈trace预测性能。

系统，软件优化。从算法出发。

找找别人的性能优化工作。
