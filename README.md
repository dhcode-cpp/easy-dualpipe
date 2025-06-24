# easy-dualpipe
Pipeline-Parallel Lecture: Simplest Dualpipe Implementation.

Inspired by `DeepSeek-V3` and [DualPipe](https://github.com/deepseek-ai/DualPipe), We designed and implemented a simplest dualpipe-like schedule.

Blog: [【手撕DualPipe】让我们一步步把 MoE EP 通信 "消除"（附代码）](https://zhuanlan.zhihu.com/p/1910995677451912435)

![easy-dualpipe](img/easy_dualpipe.png)

## Run

run `dualpipe_step.py` and `dualpipe.py` 

----

## Lecture: [手撕LLM]

完整课程包含：手撕LLM/RLHF/VLM/推理加速，新增手撕分布式。

### 🔥 手撕分布式 Overview

minimal distribution training implementation from scratch

受 [open-infra-index](https://github.com/deepseek-ai/open-infra-index) 启发，为了掌握其所涉及的分布式训练技术栈，独立系统手撕关键的并行算法，简化实现以帮助更快理解 算法原理。

- 纯 `Pytorch` 从零实现 `5` 大类并行算法：`DP`、`TP`、`PP`、`CP`、`EP`分布式训练算法。不依赖 `DeepSpeed` 和 `Megatron` 框架
- 硬核手撕关键算法 `Backward` ，手撕分布式`gradient`和`ZeRO-ADAM`， 硬核实现 MoE EP 1F1B 下的 **通信-计算重叠** 
- Step-by-step 手撕 `DP:ZeRO-3`、`TP:Llama`、`CP: RingAttention`、`PP: DualPipe`、`EP: Gshard`
- 不需要多卡环境，纯CPU GLOO backend可运行所有实例，无须 triton和cuda等基础

实现关键算法

| DP             | TP                | PP                 | CP                   | EP             |
| -------------- | ----------------- | ------------------ | -------------------- | -------------- |
| 分布式数据     | **col-parallel**  | **梯度检查点**     | Ring-AllReduce       | All2All        |
| **DP梯度**     | **row-parallel**  | **PP-basic**       | **Softmax**          | 异步All2All    |
| **分布式Adam** | **SwiGLU**        | **Gpipe**          | Online-Softmax       | **TopK 梯度**  |
| **ZeRO-1**     | **🔥GQA**          | **PipeDream**      | Ring-Softmax         | **SMoE**       |
| **ZeRO-2**     | **LMhead**        | **Zero-Bubble**    | **FlashAttention-2** | **🔥GShard**    |
| **🔥ZeRO-3**    | **Embedding**     | 🔥**Easy-Dualpipe** | **🔥Ring-Attention**  | **1F1B Basic** |
| 混合精度Adam   | **RMS-Norm**      | **DualPipe**       |                      | **🔥1F1B 重叠** |
| IO-load-save   | **🔥CrossEntropy** |                    |                      | V3-MoE结构     |

### 联系

[[完整课程目录]](https://mp.weixin.qq.com/s/Jrtgt67Eh77jNk4cmcixFg), 课程学员上岸 `OpenAI`, `Meta`, `SEED` 等

微信：xiaodongguaAIGC

<img src="./README.assets/IMG_8606.JPG" alt="IMG_8606" width="300"  />

## About

this repo is part of lecture: "LLM from scratch". This repo Not to be used for any commercial purposes without permission.

License: CC-BY-NC-ND-4.0

## Reference

[DualPipe](https://github.com/deepseek-ai/DualPipe)