/* =============================================================================
 * transformer-forge.js
 * -----------------------------------------------------------------------------
 * 世界顶级 AI 的通用 backbone —— Transformer —— 纯手写、零依赖、零构建实现。
 *
 * 设计原则（继承 forge 系列）：
 *   - 零重写：站在世界顶级成果之上（Vaswani et al. 2017, Pre-LN 变体, Adam）。
 *   - 集大成：多头注意力 + LayerNorm + FFN + 残差，覆盖现代 LLM 的全部基础算子。
 *   - 可验证：每条算子都有可被独立实现交叉验证的「不变量」。
 *   - 闭环：浏览器一键自检 + 无头 Node 8/8 全绿；完整开源，署名 晨星。
 *
 * 运行：
 *   node transformer-forge.js        -> 跑 8 项自检并打印
 *   浏览器：<script src="transformer-forge.js"></script> 后访问 window.TF
 *
 * 作者：晨星
 * ===========================================================================*/

(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.TF = api;
  if (typeof root !== 'undefined') root.TransformerForge = api;
})(typeof self !== 'undefined' ? self : this, function () {

  'use strict';

  /* ---------------------------------------------------------------------------
   * 0. 确定性随机（可复现）
   * -------------------------------------------------------------------------*/
  let _seed = 0x9e3779b9;
  function srand() {
    // xorshift32
    _seed ^= _seed << 13; _seed >>>= 0;
    _seed ^= _seed >> 17;
    _seed ^= _seed << 5; _seed >>>= 0;
    return _seed / 4294967296;
  }
  function setSeed(s) { _seed = (s >>> 0) || 1; }
  function randMat(r, c, scale) {
    scale = scale == null ? 0.1 : scale;
    const m = new Array(r);
    for (let i = 0; i < r; i++) {
      const row = new Array(c);
      for (let j = 0; j < c; j++) row[j] = (srand() * 2 - 1) * scale;
      m[i] = row;
    }
    return m;
  }
  function zeros(r, c) {
    const m = new Array(r);
    for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0);
    return m;
  }
  function ones(n) { return new Array(n).fill(1); }

  /* ---------------------------------------------------------------------------
   * 1. 矩阵 / 张量原语（行主序 [rows][cols]）
   * -------------------------------------------------------------------------*/
  const M = {
    matmul(A, B) {
      // A[r][k] * B[k][c] -> [r][c]
      const r = A.length, k = B.length, c = B[0].length;
      const C = new Array(r);
      for (let i = 0; i < r; i++) {
        const Ci = new Array(c).fill(0);
        const Ai = A[i];
        for (let p = 0; p < k; p++) {
          const a = Ai[p];
          if (a === 0) continue;
          const Bp = B[p];
          for (let j = 0; j < c; j++) Ci[j] += a * Bp[j];
        }
        C[i] = Ci;
      }
      return C;
    },
    T(A) {
      const r = A.length, c = A[0].length, B = new Array(c);
      for (let j = 0; j < c; j++) {
        const Bj = new Array(r);
        for (let i = 0; i < r; i++) Bj[i] = A[i][j];
        B[j] = Bj;
      }
      return B;
    },
    add(A, B) {
      const r = A.length, c = A[0].length, C = new Array(r);
      for (let i = 0; i < r; i++) {
        const Ci = new Array(c);
        for (let j = 0; j < c; j++) Ci[j] = A[i][j] + B[i][j];
        C[i] = Ci;
      }
      return C;
    },
    sub(A, B) {
      const r = A.length, c = A[0].length, C = new Array(r);
      for (let i = 0; i < r; i++) {
        const Ci = new Array(c);
        for (let j = 0; j < c; j++) Ci[j] = A[i][j] - B[i][j];
        C[i] = Ci;
      }
      return C;
    },
    addBias(A, b) {
      return A.map(row => row.map((v, j) => v + b[j]));
    },
    colSum(A) {
      const c = A[0].length, s = new Array(c).fill(0);
      for (const row of A) for (let j = 0; j < c; j++) s[j] += row[j];
      return s;
    },
    // 将 src 加到 target 的 (rowOff, colOff) 处（用于多头拼接）
    addInto(target, src, colOff, rowOff) {
      colOff = colOff || 0; rowOff = rowOff || 0;
      for (let i = 0; i < src.length; i++)
        for (let j = 0; j < src[0].length; j++)
          target[rowOff + i][colOff + j] += src[i][j];
    },
    sliceCols(mat, c0, c1) {
      return mat.map(row => {
        const out = new Array(c1 - c0);
        for (let j = c0; j < c1; j++) out[j - c0] = row[j];
        return out;
      });
    }
  };

  /* ---------------------------------------------------------------------------
   * 2. 激活函数（GELU 近似 + 导数）
   * -------------------------------------------------------------------------*/
  const GELU_C = Math.sqrt(2 / Math.PI);
  function gelu(x) {
    const u = GELU_C * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1 + Math.tanh(u));
  }
  function geluPrime(x) {
    const u = GELU_C * (x + 0.044715 * x * x * x);
    const up = GELU_C * (1 + 3 * 0.044715 * x * x);
    const t = Math.tanh(u);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * up;
  }
  function applyAct(mat, act) {
    if (act === 'relu') return mat.map(r => r.map(v => (v > 0 ? v : 0)));
    if (act === 'gelu') return mat.map(r => r.map(gelu));
    return mat; // linear
  }

  /* ---------------------------------------------------------------------------
   * 3. Softmax（按行，数值稳定 logsumexp 风格减去最大值）
   * -------------------------------------------------------------------------*/
  function softmaxRows(mat) {
    const r = mat.length, c = mat[0].length, out = new Array(r);
    for (let i = 0; i < r; i++) {
      let mx = -Infinity;
      for (let j = 0; j < c; j++) if (mat[i][j] > mx) mx = mat[i][j];
      let sum = 0;
      const row = new Array(c);
      for (let j = 0; j < c; j++) { row[j] = Math.exp(mat[i][j] - mx); sum += row[j]; }
      for (let j = 0; j < c; j++) row[j] /= sum;
      out[i] = row;
    }
    return out;
  }

  /* ---------------------------------------------------------------------------
   * 4. LayerNorm（对每个 token 在特征维做归一化）
   *     y = (x - mean) / std * gamma + beta,  std = sqrt(var + eps)
   * -------------------------------------------------------------------------*/
  function layernormForward(X, g, b, eps) {
    eps = eps == null ? 1e-5 : eps;
    const Tn = X.length, d = X[0].length;
    const Y = new Array(Tn), mean = new Array(Tn), std = new Array(Tn);
    for (let i = 0; i < Tn; i++) {
      let m = 0;
      for (let j = 0; j < d; j++) m += X[i][j];
      m /= d;
      let v = 0;
      for (let j = 0; j < d; j++) { const u = X[i][j] - m; v += u * u; }
      v /= d;
      const s = Math.sqrt(v + eps);
      const row = new Array(d);
      for (let j = 0; j < d; j++) row[j] = (X[i][j] - m) / s * g[j] + b[j];
      Y[i] = row; mean[i] = m; std[i] = s;
    }
    return { Y, cache: { X, g, b, eps, mean, std } };
  }
  function layernormBackward(dY, cache) {
    const { X, g, b, eps, mean, std } = cache;
    const Tn = X.length, d = X[0].length;
    const dg = new Array(d).fill(0), db = new Array(d).fill(0), dX = new Array(Tn);
    for (let i = 0; i < Tn; i++) {
      const xhat = new Array(d);
      for (let j = 0; j < d; j++) xhat[j] = (X[i][j] - mean[i]) / std[i];
      let md = 0, mdx = 0;
      for (let j = 0; j < d; j++) {
        const dhat = dY[i][j] * g[j];
        dg[j] += dY[i][j] * xhat[j];
        db[j] += dY[i][j];
        md += dhat; mdx += dhat * xhat[j];
      }
      md /= d; mdx /= d;
      const dx = new Array(d);
      for (let j = 0; j < d; j++)
        dx[j] = (dY[i][j] * g[j] - md - mdx * xhat[j]) / std[i];
      dX[i] = dx;
    }
    return { dX, dg, db };
  }

  /* ---------------------------------------------------------------------------
   * 5. 多头自注意力（scaled dot-product, 可选因果掩码）
   *    Q = X Wq, K = X Wk, V = X Wv
   *    A = softmax( Q K^T / sqrt(dk) (+ mask) )
   *    C = A V  ->  concat heads -> Y = C Wo
   * -------------------------------------------------------------------------*/
  function attentionForward(X, P, opts) {
    const Tn = X.length, d = X[0].length, H = opts.n_heads, dk = d / H;
    const { Wq, Wk, Wv, Wo } = P;
    const Q = M.matmul(X, Wq), K = M.matmul(X, Wk), V = M.matmul(X, Wv);
    const Cfull = zeros(Tn, H * dk);
    const headCaches = [];
    for (let h = 0; h < H; h++) {
      const qh = M.sliceCols(Q, h * dk, (h + 1) * dk);
      const kh = M.sliceCols(K, h * dk, (h + 1) * dk);
      const vh = M.sliceCols(V, h * dk, (h + 1) * dk);
      const scores = M.matmul(qh, M.T(kh)); // [Tn][Tn]
      const scale = 1 / Math.sqrt(dk);
      for (let i = 0; i < Tn; i++)
        for (let j = 0; j < Tn; j++) {
          if (opts.causal && j > i) scores[i][j] = -Infinity;
          else scores[i][j] *= scale;
        }
      const A = softmaxRows(scores);
      const C = M.matmul(A, vh);
      for (let i = 0; i < Tn; i++)
        for (let j = 0; j < dk; j++) Cfull[i][h * dk + j] = C[i][j];
      headCaches.push({ A, qh, kh, vh, dk });
    }
    const Y = M.matmul(Cfull, Wo);
    return { Y, cache: { X, P, opts, Cfull, headCaches } };
  }
  function attentionBackward(dY, cache) {
    const { X, P, opts, Cfull, headCaches } = cache;
    const Tn = X.length, d = X[0].length, H = opts.n_heads, dk = d / H;
    const { Wq, Wk, Wv, Wo } = P;
    const dWo = M.matmul(M.T(Cfull), dY);            // [H*dk][d]
    const dCfull = M.matmul(dY, M.T(Wo));            // [Tn][H*dk]
    const dX = zeros(Tn, d);
    const dWq = zeros(d, H * dk), dWk = zeros(d, H * dk), dWv = zeros(d, H * dk);
    for (let hi = 0; hi < headCaches.length; hi++) {
      const hc = headCaches[hi];
      const { A, qh, kh, vh, dk } = hc;
      const dCh = zeros(Tn, dk);
      for (let i = 0; i < Tn; i++)
        for (let j = 0; j < dk; j++) dCh[i][j] = dCfull[i][hi * dk + j];
      const dVh = M.matmul(M.T(A), dCh);             // [Tn][dk]
      const dA = M.matmul(dCh, M.T(vh));             // [Tn][Tn]
      const dscores = zeros(Tn, Tn);
      const scale = 1 / Math.sqrt(dk);
      for (let i = 0; i < Tn; i++) {
        let dot = 0;
        for (let j = 0; j < Tn; j++) dot += A[i][j] * dA[i][j];
        for (let j = 0; j < Tn; j++) {
          let g = A[i][j] * (dA[i][j] - dot);
          if (opts.causal && j > i) g = 0;
          dscores[i][j] = g;
        }
      }
      const dQh = M.matmul(dscores, kh);             // [Tn][dk]
      const dKh = M.matmul(M.T(dscores), qh);        // [Tn][dk]
      for (let i = 0; i < Tn; i++)
        for (let j = 0; j < dk; j++) { dQh[i][j] *= scale; dKh[i][j] *= scale; }
      M.addInto(dWq, M.matmul(M.T(X), dQh), hi * dk, 0);
      M.addInto(dWk, M.matmul(M.T(X), dKh), hi * dk, 0);
      M.addInto(dWv, M.matmul(M.T(X), dVh), hi * dk, 0);
      const Wqh = M.sliceCols(Wq, hi * dk, (hi + 1) * dk);
      const Wkh = M.sliceCols(Wk, hi * dk, (hi + 1) * dk);
      const Wvh = M.sliceCols(Wv, hi * dk, (hi + 1) * dk);
      M.addInto(dX, M.matmul(dQh, M.T(Wqh)));
      M.addInto(dX, M.matmul(dKh, M.T(Wkh)));
      M.addInto(dX, M.matmul(dVh, M.T(Wvh)));
    }
    return { dX, dWq, dWk, dWv, dWo };
  }

  /* ---------------------------------------------------------------------------
   * 6. 前馈网络 FFN（两层线性 + GELU）
   *    Z1 = X W1 + b1 ; A1 = act(Z1) ; Z2 = A1 W2 + b2
   * -------------------------------------------------------------------------*/
  function ffnForward(X, W1, b1, W2, b2, act) {
    const Z1 = M.addBias(M.matmul(X, W1), b1);
    const A1 = applyAct(Z1, act);
    const Z2 = M.addBias(M.matmul(A1, W2), b2);
    return { Y: Z2, cache: { X, W1, b1, W2, b2, A1, Z1, act } };
  }
  function ffnBackward(dY, cache) {
    const { X, W1, b1, W2, b2, A1, Z1, act } = cache;
    const dW2 = M.matmul(M.T(A1), dY);
    const db2 = M.colSum(dY);
    const dA1 = M.matmul(dY, M.T(W2));
    const dZ1 = dA1.map((row, i) => row.map((v, j) =>
      v * (act === 'gelu' ? geluPrime(Z1[i][j]) : (Z1[i][j] > 0 ? 1 : 0))));
    const dW1 = M.matmul(M.T(X), dZ1);
    const db1 = M.colSum(dZ1);
    const dX = M.matmul(dZ1, M.T(W1));
    return { dX, dW1, db1, dW2, db2 };
  }

  /* ---------------------------------------------------------------------------
   * 7. Pre-LN Transformer 块（现代稳定变体：残差包在 LayerNorm 外）
   *    a = X + Attention(LN(X))
   *    Y = a + FFN(LN(a))
   * -------------------------------------------------------------------------*/
  function blockForward(X, P, opts) {
    const ln1 = layernormForward(X, P.ln1g, P.ln1b, opts.eps);
    const att = attentionForward(ln1.Y, P.attn, opts);
    const a = M.add(X, att.Y);
    const ln2 = layernormForward(a, P.ln2g, P.ln2b, opts.eps);
    const ff = ffnForward(ln2.Y, P.ffn.W1, P.ffn.b1, P.ffn.W2, P.ffn.b2, opts.act);
    const Y = M.add(a, ff.Y);
    return { Y, cache: { ln1, att, a, ln2, ff, X } };
  }
  function blockBackward(dY, cache, opts) {
    const { ln1, att, a, ln2, ff } = cache;
    const ffb = ffnBackward(dY, ff.cache);                 // d(ln2.Y) = ffb.dX
    const ln2b = layernormBackward(ffb.dX, ln2.cache);     // d(a)
    const dA = M.add(dY, ln2b.dX);                         // a = X + att.Y
    const ab = attentionBackward(dA, att.cache);           // d(ln1.Y) = ab.dX
    const ln1b = layernormBackward(ab.dX, ln1.cache);      // d(X) via attn
    const dX = M.add(dA, ln1b.dX);                         // 残差 + 注意力路径
    return {
      dX,
      ln1g: ln1b.dg, ln1b: ln1b.db,
      ln2g: ln2b.dg, ln2b: ln2b.db,
      attn: { Wq: ab.dWq, Wk: ab.dWk, Wv: ab.dWv, Wo: ab.dWo },
      ffn: { W1: ffb.dW1, b1: ffb.db1, W2: ffb.dW2, b2: ffb.db2 }
    };
  }

  /* ---------------------------------------------------------------------------
   * 8. 参数初始化 + Adam 优化器
   * -------------------------------------------------------------------------*/
  function initBlock(d, n_heads, d_ff, scale) {
    scale = scale == null ? 0.08 : scale;
    const H = n_heads, dk = d / H;
    return {
      ln1g: ones(d), ln1b: new Array(d).fill(0),
      attn: {
        Wq: randMat(d, H * dk, scale), Wk: randMat(d, H * dk, scale),
        Wv: randMat(d, H * dk, scale), Wo: randMat(H * dk, d, scale)
      },
      ln2g: ones(d), ln2b: new Array(d).fill(0),
      ffn: {
        W1: randMat(d, d_ff, scale), b1: new Array(d_ff).fill(0),
        W2: randMat(d_ff, d, scale), b2: new Array(d).fill(0)
      }
    };
  }
  function initAttn(d, n_heads, scale) {
    scale = scale == null ? 0.1 : scale;
    const H = n_heads, dk = d / H;
    return {
      Wq: randMat(d, H * dk, scale), Wk: randMat(d, H * dk, scale),
      Wv: randMat(d, H * dk, scale), Wo: randMat(H * dk, d, scale)
    };
  }
  // 遍历参数树中所有「矩阵」叶子，回调 (matrix, path)
  function eachMatrix(obj, fn) {
    (function rec(o, path) {
      if (!o || typeof o !== 'object') return;
      if (Array.isArray(o) && o.length > 0 && Array.isArray(o[0]) && typeof o[0][0] === 'number') {
        fn(o, path); return;
      }
      for (const k in o) rec(o[k], path + '.' + k);
    })(obj, '');
  }
  function getByPath(o, path) {
    const ks = path.split('.').filter(s => s.length);
    let cur = o;
    for (const k of ks) cur = cur[k];
    return cur;
  }
  function makeAdamState(P) {
    const s = {};
    eachMatrix(P, (m, p) => { s[p] = { m: m.map(r => r.map(() => 0)), v: m.map(r => r.map(() => 0)) }; });
    return s;
  }
  function adamUpdate(P, grads, state, t, lr, b1, b2, eps) {
    b1 = b1 == null ? 0.9 : b1; b2 = b2 == null ? 0.999 : b2; eps = eps == null ? 1e-8 : eps;
    eachMatrix(P, (m, p) => {
      const G = getByPath(grads, p), st = state[p];
      for (let i = 0; i < m.length; i++)
        for (let j = 0; j < m[0].length; j++) {
          const g = G[i][j];
          st.m[i][j] = b1 * st.m[i][j] + (1 - b1) * g;
          st.v[i][j] = b2 * st.v[i][j] + (1 - b2) * g * g;
          const mh = st.m[i][j] / (1 - Math.pow(b1, t));
          const vh = st.v[i][j] / (1 - Math.pow(b2, t));
          m[i][j] -= lr * mh / (Math.sqrt(vh) + eps);
        }
    });
  }

  // 通用训练循环：lossFn(P, X, Yt) -> {loss, grads}
  function trainLoop(P, X, Yt, lossFn, opts) {
    const state = makeAdamState(P);
    let t = 0; const losses = [];
    for (let e = 0; e < opts.epochs; e++) {
      const { loss, grads } = lossFn(P, X, Yt);
      losses.push(loss);
      t++;
      adamUpdate(P, grads, state, t, opts.lr);
    }
    return losses;
  }

  /* ---------------------------------------------------------------------------
   * 9. 损失函数（MSE）与对应反向
   * -------------------------------------------------------------------------*/
  function mseLossForward(Y, Yt) {
    let loss = 0;
    for (let i = 0; i < Y.length; i++)
      for (let j = 0; j < Y[0].length; j++) { const d = Y[i][j] - Yt[i][j]; loss += 0.5 * d * d; }
    loss /= (Y.length * Y[0].length);
    return loss;
  }
  function mseLossGrad(Y, Yt) {
    const N = Y.length * Y[0].length;
    return Y.map((row, i) => row.map((v, j) => (v - Yt[i][j]) / N));
  }

  function blockLoss(P, X, Yt, opts) {
    const { Y, cache } = blockForward(X, P, opts);
    const loss = mseLossForward(Y, Yt);
    const dY = mseLossGrad(Y, Yt);
    const grads = blockBackward(dY, cache, opts);
    return { loss, grads };
  }
  function attnLoss(P, X, Yt, opts) {
    const { Y, cache } = attentionForward(X, P, opts);
    const loss = mseLossForward(Y, Yt);
    const dY = mseLossGrad(Y, Yt);
    const g = attentionBackward(dY, cache);
    return { loss, grads: { Wq: g.dWq, Wk: g.dWk, Wv: g.dWv, Wo: g.dWo } };
  }
  function ffnLoss(P, X, Yt, opts) {
    const { Y, cache } = ffnForward(X, P.W1, P.b1, P.W2, P.b2, opts.act);
    const loss = mseLossForward(Y, Yt);
    const dY = mseLossGrad(Y, Yt);
    const g = ffnBackward(dY, cache);
    return { loss, grads: { W1: g.dW1, b1: g.db1, W2: g.dW2, b2: g.db2 } };
  }

  /* ---------------------------------------------------------------------------
   * 10. 梯度检验（数值梯度 vs 解析梯度，中心差分）
   * -------------------------------------------------------------------------*/
  function gradCheck(P, X, lossFn, eps) {
    eps = eps == null ? 1e-5 : eps;
    const paths = [];
    eachMatrix(P, (m, p) => paths.push(p));
    const an = lossFn(P, X).grads;
    let maxRel = 0, count = 0;
    for (const p of paths) {
      const m = getByPath(P, p), Ag = getByPath(an, p);
      for (let i = 0; i < m.length; i++)
        for (let j = 0; j < m[0].length; j++) {
          const old = m[i][j];
          m[i][j] = old + eps; const lp = lossFn(P, X).loss;
          m[i][j] = old - eps; const lm = lossFn(P, X).loss;
          m[i][j] = old;
          const num = (lp - lm) / (2 * eps);
          const a = Ag[i][j];
          const denom = Math.max(Math.abs(num), Math.abs(a), 1e-12) + 1e-12;
          const rel = Math.abs(num - a) / denom;
          if (rel > maxRel) maxRel = rel;
          count++;
        }
    }
    return { maxRel, count };
  }

  /* ---------------------------------------------------------------------------
   * 11. 自检套件（8/8）
   * -------------------------------------------------------------------------*/
  function approxEq(a, b, tol) { return Math.abs(a - b) <= tol; }
  function maxAbsMatDiff(A, B) {
    let mx = 0;
    for (let i = 0; i < A.length; i++)
      for (let j = 0; j < A[0].length; j++) mx = Math.max(mx, Math.abs(A[i][j] - B[i][j]));
    return mx;
  }

  function runTests() {
    const tests = [];
    const opts = { n_heads: 2, eps: 1e-5, act: 'gelu', causal: true };

    // --- T1: 矩阵乘法 vs 暴力三重循环 ---
    (function () {
      setSeed(11);
      const A = randMat(3, 2, 1), B = randMat(2, 4, 1);
      const C = M.matmul(A, B);
      const ref = zeros(3, 4);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) {
        let s = 0; for (let k = 0; k < 2; k++) s += A[i][k] * B[k][j]; ref[i][j] = s;
      }
      tests.push({ name: '矩阵乘法 matmul', pass: maxAbsMatDiff(C, ref) < 1e-12,
        info: 'max|Δ|=' + maxAbsMatDiff(C, ref).toExponential(2) + ' (对照三重循环)' });
    })();

    // --- T2: softmax 行和为 1 ---
    (function () {
      setSeed(22);
      const mat = randMat(5, 6, 3);
      const S = softmaxRows(mat);
      let ok = true, worst = 0;
      for (let i = 0; i < 5; i++) {
        let s = 0; for (let j = 0; j < 6; j++) s += S[i][j];
        worst = Math.max(worst, Math.abs(s - 1));
        if (Math.abs(s - 1) > 1e-12) ok = false;
      }
      tests.push({ name: 'Softmax 归一', pass: ok, info: '行和最大偏差=' + worst.toExponential(2) });
    })();

    // --- T3: LayerNorm 零均值 / 单位方差（g=1,b=0） ---
    (function () {
      setSeed(33);
      const d = 8, X = randMat(4, d, 2);
      const { Y } = layernormForward(X, ones(d), new Array(d).fill(0), 1e-5);
      let ok = true, worstMean = 0, worstVar = 0;
      for (let i = 0; i < 4; i++) {
        let m = 0; for (let j = 0; j < d; j++) m += Y[i][j]; m /= d;
        let v = 0; for (let j = 0; j < d; j++) { const u = Y[i][j] - m; v += u * u; } v /= d;
        worstMean = Math.max(worstMean, Math.abs(m));
        worstVar = Math.max(worstVar, Math.abs(v - 1));
        if (Math.abs(m) > 1e-9 || Math.abs(v - 1) > 1e-4) ok = false;
      }
      tests.push({ name: 'LayerNorm 统计', pass: ok,
        info: '|mean|≤' + worstMean.toExponential(2) + ', |var-1|≤' + worstVar.toExponential(2) + ' (eps 引入)' });
    })();

    // --- T4: 因果掩码（位置 i 只可见 ≤ i） ---
    (function () {
      setSeed(44);
      const d = 4, H = 2, Tn = 5;
      const P = initAttn(d, H, 0.3);
      const X = randMat(Tn, d, 1);
      const { Y, cache } = attentionForward(X, P, { n_heads: H, causal: true });
      const hc = cache.headCaches[0];
      let ok = true, worst = 0;
      for (let i = 0; i < Tn; i++)
        for (let j = 0; j < Tn; j++)
          if (j > i) { worst = Math.max(worst, hc.A[i][j]); if (hc.A[i][j] > 1e-12) ok = false; }
      tests.push({ name: '因果掩码', pass: ok, info: '上三角最大注意力权重=' + worst.toExponential(2) });
    })();

    // --- T5: 注意力梯度检验 ---
    (function () {
      setSeed(55);
      const d = 4, H = 2, Tn = 3;
      const P = initAttn(d, H, 0.4);
      const X = randMat(Tn, d, 1);
      const Yt = randMat(Tn, d, 1);
      const { maxRel, count } = gradCheck(P, X, (p, x) => attnLoss(p, x, Yt, { n_heads: H, causal: false }));
      tests.push({ name: '注意力梯度检验', pass: maxRel < 1e-4,
        info: 'maxRelErr=' + maxRel.toExponential(2) + ' / ' + count + ' 参数' });
    })();

    // --- T6: FFN 梯度检验 ---
    (function () {
      setSeed(66);
      const d = 4, d_ff = 6, Tn = 3;
      const P = { W1: randMat(d, d_ff, 0.4), b1: new Array(d_ff).fill(0),
                  W2: randMat(d_ff, d, 0.4), b2: new Array(d).fill(0) };
      const X = randMat(Tn, d, 1);
      const Yt = randMat(Tn, d, 1);
      const { maxRel, count } = gradCheck(P, X, (p, x) => ffnLoss(p, x, Yt, { act: 'gelu' }));
      tests.push({ name: 'FFN 梯度检验', pass: maxRel < 1e-4,
        info: 'maxRelErr=' + maxRel.toExponential(2) + ' / ' + count + ' 参数' });
    })();

    // --- T7: 整块（Pre-LN）梯度检验 ---
    (function () {
      setSeed(77);
      const d = 4, H = 2, d_ff = 6, Tn = 3;
      const P = initBlock(d, H, d_ff, 0.3);
      const X = randMat(Tn, d, 1);
      const Yt = randMat(Tn, d, 1);
      const opts2 = { n_heads: H, eps: 1e-5, act: 'gelu', causal: false };
      const { maxRel, count } = gradCheck(P, X, (p, x) => blockLoss(p, x, Yt, opts2));
      tests.push({ name: '整块梯度检验 (Pre-LN)', pass: maxRel < 1e-4,
        info: 'maxRelErr=' + maxRel.toExponential(2) + ' / ' + count + ' 参数' });
    })();

    // --- T8: 训练收敛性（无残差单层注意力学习恒等映射） ---
    (function () {
      setSeed(88);
      const d = 6, H = 2, Tn = 5;
      const P = initAttn(d, H, 0.5);
      setSeed(99);
      const X = randMat(Tn, d, 1);
      const Yt = X.map(r => r.slice()); // 目标 = 输入（copy）
      const losses = trainLoop(P, X, Yt, (p, x, y) => attnLoss(p, x, y, { n_heads: H, causal: false }),
        { epochs: 1500, lr: 0.05 });
      const drop = (losses[0] - losses[losses.length - 1]) / (losses[0] + 1e-12);
      tests.push({ name: '训练收敛性', pass: drop > 0.5,
        info: 'loss ' + losses[0].toFixed(4) + ' → ' + losses[losses.length - 1].toFixed(4)
            + ' (下降 ' + (drop * 100).toFixed(1) + '%)' });
    })();

    const allPass = tests.every(t => t.pass);
    return { tests, allPass };
  }

  /* ---------------------------------------------------------------------------
   * 12. 浏览器交互演示辅助：小样本前向 + 注意力热力图数据
   * -------------------------------------------------------------------------*/
  function demoAttention(d, n_heads, causal, seed) {
    setSeed(seed == null ? 123 : seed);
    const Tn = 5;
    const P = initAttn(d, n_heads, 0.6);
    const X = randMat(Tn, d, 1);
    const { cache } = attentionForward(X, P, { n_heads, causal });
    return {
      Tn, n_heads, causal,
      heads: cache.headCaches.map(h => h.A),
      X
    };
  }

  /* ---------------------------------------------------------------------------
   * 13. 公共 API
   * -------------------------------------------------------------------------*/
  return {
    // 原语
    matmul: M.matmul, T: M.T, add: M.add, sub: M.sub, addBias: M.addBias,
    colSum: M.colSum, sliceCols: M.sliceCols, addInto: M.addInto,
    zeros, ones, randMat, setSeed, srand,
    gelu, geluPrime, applyAct, softmaxRows,
    // 算子
    layernormForward, layernormBackward,
    attentionForward, attentionBackward,
    ffnForward, ffnBackward,
    blockForward, blockBackward,
    // 模型 / 优化
    initBlock, initAttn, trainLoop, adamUpdate, makeAdamState,
    eachMatrix, getByPath,
    mseLossForward, mseLossGrad, blockLoss, attnLoss, ffnLoss,
    // 验证
    gradCheck, runTests,
    // 演示
    demoAttention
  };
});

/* Node 直接运行：跑自检 */
if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
  const TF = module.exports;
  const res = TF.runTests();
  console.log('============================================');
  console.log(' transformer-forge 自检 (晨星)');
  console.log('============================================');
  for (const t of res.tests) {
    console.log((t.pass ? '  ✅ ' : '  ❌ ') + t.name + '  —  ' + t.info);
  }
  console.log('--------------------------------------------');
  console.log(res.allPass ? '  结果: 8/8 全绿 ✅' : '  结果: 存在失败 ❌');
  console.log('============================================');
  if (!res.allPass) process.exitCode = 1;
}
