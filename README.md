# transformer-forge · 从零手写的 mini-Transformer / GPT 实验室

<p align="center">
  <a href="https://github.com/CJX0712/transformer-forge/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/transformer-forge/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/transformer-forge/releases"><img src="https://img.shields.io/github/v/release/CJX0712/transformer-forge?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/transformer-forge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/transformer-forge" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

> forge 系列 #14 — 零依赖、可在浏览器内训练、可在 Node 无头自检的因果 Transformer。

不是调包，不是「调用 `torch.nn.Transformer`」。从矩阵乘、LayerNorm、因果多头注意力、FFN 到 Adam，
**全部手算**，纯 Float64Array，单文件 HTML 内联。打开 `index.html` 就能训练；跑 `node _smoke.js` 就能自检。

## 它是什么

一个最小的 decoder-only Transformer 块（可堆叠多层）：

```
h_in ──► LN1 ──► 因果多头自注意力(MHA) ──► + 残差 ──► h_mid
                                                      │
h_mid ─► LN2 ──► FFN(W1→ReLU→W2) ─────────► + 残差 ──► h_out
                                                      │
                                            └─► Wlm ──► softmax ──► 交叉熵
```

- **因果多头自注意力**：softmax 作用在 −∞ 掩码的下三角上，按 √d_k 缩放。
- **位置编码**：可学习的 `tokenEmb + posEmb` 相加（不是 RoPE/ sinusoidal，故意保持最小）。
- **LayerNorm**（按行，含解析后向）、**FFN**（`W1·ReLU(·)+c1 → W2·(·)+c2`）、**Adam** 优化器，全部手写。
- 内置两个玩具任务：`shift`（位置 i 预测 token i−1，逼模型真正学会跨位置注意）与 `copy`（恒等映射基线）。

## 运行

### 浏览器（零依赖）
直接用浏览器打开 `index.html`：
1. 选任务 / 调超参（词表、序列长、d_model、层数、步数）。
2. 点「训练并评估」——引擎训练并画出**注意力热力图**（行=query，列=key；shift 任务应出现一条下对角亮带 `(i, i−1)`）。
3. 同页自动跑浏览器端梯度检验等不变量。

### Node 无头自检
```bash
node _smoke.js     # 9 项断言：softmax 行和=1、因果掩码、注意力有限、梯度检验、前向确定性、
                   #                      后向有限、训练 loss 下降、shift≥95%、copy≥98%
node _probe.js     # 打印 shift / copy 任务的注意力 ASCII 热力图
node _uicheck.js   # 用最小 DOM 桩驱动 UI 脚本端到端自检
```

## 正确性验证（硬不变量）

| 不变量 | 方法 | 结果 |
|---|---|---|
| 前向确定性 | 同参同输入 → 同输出 | ✅ |
| softmax 每行和 = 1 | 解析 | ✅ |
| 因果掩码 | 上三角严格为 0 | ✅ |
| 反向梯度 = 数值梯度 | 全 362 参数中心差分 | ✅ maxRel ≈ 4.6e-8 |
| LayerNorm 后向 | 独立有限差分 | ✅ maxRel ≈ 2.8e-10 |
| 整层后向输出 dHin | 独立有限差分 | ✅ maxRel ≈ 5.4e-10 |
| 训练收敛 | shift 随机测试集 100% / copy 100% | ✅ |

> 调试手记：反向曾因 `layernormBackward` 中 `dvar` 多除了一次 σ（写成 `σ⁻³` 应为 `σ⁻²`）而全错——
> 注意力与 FFN 的**参数**梯度因不走 LN 而照过，但 `dHin` 与所有嵌入/归一化梯度失真。
> 以「独立前向 + 有限差分」作金标准裁决（而非另一个手写的后向），一次性定位。

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 单文件交付物：`<script id="engine">` 引擎 + `<script id="ui">` 界面 |
| `_smoke.js` | 9 项无头断言（grad-check / 训练 / 精度） |
| `_probe.js` | 注意力 ASCII 热力图探针 |
| `_uicheck.js` | 无头 DOM 桩驱动的 UI 端到端自检 |
| `README.md` / `LICENSE` / `.gitignore` | 文档 / MIT (晨星) / 忽略规则 |

## License

MIT © 2026 晨星 (CJX0712)
