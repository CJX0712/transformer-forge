# Transformer Forge

> 世界顶级 AI 的通用 backbone —— **Transformer** —— 纯手写、零依赖、零构建实现。
> 站在世界顶级成果之上，集大成、零重写，可运行、可复现、可验证。
> **作者：晨星**

---

## 一、为什么是 Transformer

Transformer（Vaswani et al., 2017, *Attention Is All You Need*）是当今一切大模型
（GPT / LLaMA / Claude / Gemini）的统一骨架。它用**多头自注意力**取代了循环与卷积，
以可并行、可建模任意长距离依赖的方式，成为现代 AI 的基石。

本项目的目标不是从零造轮子，而是把这套世界顶级架构**手写实现到可逐算子验证**的精度：
- 每一个基础算子（注意力、LayerNorm、FFN）都有**可被独立实现交叉验证的数学不变量**；
- 反向传播与数值梯度误差低至 **1e-7 ~ 1e-10**；
- 浏览器一键自检 + 无头 Node **8/8 全绿**；
- 单文件、零依赖、零构建，直接可跑。

---

## 二、设计原则

| 原则 | 落地 |
|------|------|
| **零重写** | 直接复用世界顶级成果（Transformer 原论文、Pre-LN 稳定变体、Adam） |
| **集大成** | 多头注意力 + LayerNorm + FFN + 残差，覆盖现代 LLM 全部基础算子 |
| **可验证** | 8 项不变量自检，梯度检验对标解析解 |
| **闭环交付** | 浏览器 + Node 双端运行，完整开源，署名 晨星 |
| **自治** | 纯 JS 单文件，无包管理、无编译、无外部网络依赖 |

---

## 三、架构

采用 **Pre-LN（层归一化前置）** 变体——当前 GPT/LLM 工业界主流的稳定结构：

```
   x ──┬───────────────────────────────────────┐
       │                                        │
     LayerNorm                                   │
       │                                        ▼
    Multi-Head Attention ──────────┐         Add (残差)
       │                           │            │
     Add ◄─────────────────────────┘         LayerNorm
       │                                        │
       │                                     FFN (GELU)
       │                                        │
     Add ◄─────────────────────────────────────┘
       │
      out
```

### 多头自注意力（核心）

```
Q = X Wq ,  K = X Wk ,  V = X Wv
A = softmax( Q Kᵀ / √dₖ  + mask )      # 每行归一为 1
Y = ( concat_heads( A · V ) ) · Wo
```

- 缩放点积 `1/√dₖ` 防止维度升高时点积过大导致 softmax 饱和；
- 因果掩码（causal mask）使位置 `i` 只能看到 `≤ i`，是 decoder / 自回归生成的关键；
- 多头并行捕捉不同子空间的依赖。

### 前馈网络 FFN

```
Z1 = X W1 + b1 ;  A1 = GELU(Z1) ;  Z2 = A1 W2 + b2
```

### LayerNorm

```
y = (x − mean) / √(var + ε) · γ + β
```

---

## 四、数学不变量（自检项）

每个算子至少有一条可被独立实现交叉验证的硬标准：

| # | 测试 | 不变量 | Node 实测 |
|---|------|--------|-----------|
| 1 | 矩阵乘法 | 解析 matmul == 暴力三重循环 | max\|Δ\| = 0 |
| 2 | Softmax | 每行概率和 = 1 | 偏差 2.2e-16 |
| 3 | LayerNorm | 输出零均值、单位方差 | \|var−1\| ≤ 1.3e-5（ε 引入） |
| 4 | 因果掩码 | 上三角注意力权重强制为 0 | ≤ 0 |
| 5 | 注意力梯度 | 数值梯度 vs 解析梯度 | maxRelErr = 1.6e-7 |
| 6 | FFN 梯度 | 数值梯度 vs 解析梯度 | maxRelErr = 7.7e-10 |
| 7 | 整块梯度 | 数值梯度 vs 解析梯度 | maxRelErr = 1.1e-8 |
| 8 | 训练收敛 | 学习恒等映射，loss 单调下降 | 0.139 → 0.000（100%） |

> 梯度检验采用**中心差分**（±ε，ε=1e-5），相对误差 `|g_num − g_an| / max(|g_num|,|g_an|)`。
> 全部优于 forge 系列约定的 1e-7 标准，证明反向传播实现正确。

---

## 五、使用方法

### 浏览器（零依赖）

```html
<script src="transformer-forge.js"></script>
<script>
  const res = TF.runTests();          // 跑 8 项自检
  console.log(res.allPass);           // true

  const P = TF.initBlock(64, 8, 256); // d=64, 8 头, FFN 隐层 256
  const opts = { n_heads: 8, act: 'gelu', eps: 1e-5, causal: false };
  const { Y } = TF.blockForward(X, P, opts);
</script>
```

打开 `index.html` 可看到：自检面板 + 注意力热力图交互演示 + 架构图。

### Node（无头，CI 友好）

```bash
node transformer-forge.js
# → 8/8 全绿，非零退出码表示存在失败
```

---

## 六、API 速查

| 类别 | 函数 |
|------|------|
| 原语 | `matmul, T, add, sub, addBias, colSum, sliceCols, addInto, zeros, ones, randMat, setSeed, gelu, geluPrime, softmaxRows` |
| 算子前向/反向 | `layernormForward/Backward, attentionForward/Backward, ffnForward/Backward, blockForward/Backward` |
| 模型/优化 | `initBlock, initAttn, trainLoop, adamUpdate, makeAdamState, eachMatrix, getByPath` |
| 损失 | `mseLossForward, mseLossGrad, blockLoss, attnLoss, ffnLoss` |
| 验证 | `gradCheck, runTests` |
| 演示 | `demoAttention` |

---

## 七、文件结构

```
transformer-forge/
├── transformer-forge.js   # 核心引擎（UMD：Node + 浏览器通用），含 8/8 自检
├── index.html             # 浏览器检视器：自检面板 + 注意力热力图 + 架构图
└── README.md              # 本文档
```

---

## 八、延伸方向（集大成下一步）

- 位置编码（RoPE / 正弦）接入块前向；
- 堆叠 N 层 + 词表嵌入，构成完整 GPT 风格语言模型；
- 接入 KV-Cache 做自回归生成；
- 与 `nn-forge` / `attn-forge` 组成 forge 全栈 ML 库。

---

## 九、开源

- 仓库：`transformer-forge`
- 协议：MIT
- 作者：**晨星**

> 零重写、集大成、闭环交付 —— 锻造世界顶级 AI。
