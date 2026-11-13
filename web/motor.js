// Motor de inferencia do MiniT2I em JavaScript puro. Mesmo calculo do treino/dit.py,
// conferido contra o PyTorch pelo caso de referencia do modelo.json.

export function decodificarF16(b64) {
  const bin = atob(b64.trim());
  const n = bin.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const geluErf = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
const geluTanh = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
const silu = (x) => x / (1 + Math.exp(-x));

// out[i,o] = sum_k x[i,k] W[o,k] + b[o]
function linear(x, n, din, W, b, dout, wOfs = 0, bOfs = 0) {
  const out = new Float32Array(n * dout);
  for (let i = 0; i < n; i++) {
    const xi = i * din;
    for (let o = 0; o < dout; o++) {
      const wo = wOfs + o * din;
      let s = b ? b[bOfs + o] : 0;
      for (let k = 0; k < din; k++) s += x[xi + k] * W[wo + k];
      out[i * dout + o] = s;
    }
  }
  return out;
}

function layerNorm(x, n, d, eps, g, b) {
  const out = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    let m = 0, v = 0;
    for (let k = 0; k < d; k++) m += x[i * d + k];
    m /= d;
    for (let k = 0; k < d; k++) { const z = x[i * d + k] - m; v += z * z; }
    const r = 1 / Math.sqrt(v / d + eps);
    for (let k = 0; k < d; k++) {
      const z = (x[i * d + k] - m) * r;
      out[i * d + k] = g ? z * g[k] + b[k] : z;
    }
  }
  return out;
}

// nn.MultiheadAttention, consulta xq (nq,d), chave e valor xk (nk,d), mask[j] true ignora a chave j
function atencao(xq, nq, xk, nk, d, H, P, mask) {
  const W = P.in_proj_weight, B = P.in_proj_bias;
  const Q = linear(xq, nq, d, W, B, d, 0, 0);
  const K = linear(xk, nk, d, W, B, d, d * d, d);
  const V = linear(xk, nk, d, W, B, d, 2 * d * d, 2 * d);
  const hd = d / H, esc = 1 / Math.sqrt(hd);
  const O = new Float32Array(nq * d), sc = new Float32Array(nk);
  for (let h = 0; h < H; h++) {
    const ho = h * hd;
    for (let i = 0; i < nq; i++) {
      let mx = -Infinity;
      for (let j = 0; j < nk; j++) {
        if (mask && mask[j]) { sc[j] = -Infinity; continue; }
        let s = 0;
        for (let k = 0; k < hd; k++) s += Q[i * d + ho + k] * K[j * d + ho + k];
        sc[j] = s * esc; if (sc[j] > mx) mx = sc[j];
      }
      let soma = 0;
      for (let j = 0; j < nk; j++) { sc[j] = sc[j] === -Infinity ? 0 : Math.exp(sc[j] - mx); soma += sc[j]; }
      for (let j = 0; j < nk; j++) {
        const p = sc[j] / soma; if (!p) continue;
        for (let k = 0; k < hd; k++) O[i * d + ho + k] += p * V[j * d + ho + k];
      }
    }
  }
  return linear(O, nq, d, P.out_proj_weight, P.out_proj_bias, d);
}

export class MiniT2I {
  constructor(cfg, pesos) {
    this.cfg = cfg;
    this.w = {};
    for (const t of cfg.tensores) this.w[t.nome] = pesos.subarray(t.ofs, t.ofs + t.n);
    this.stoi = Object.fromEntries(cfg.vocab.map((p, i) => [p, i]));
    const tira = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "");
    this.semAcento = Object.fromEntries(cfg.vocab.map((p, i) => [tira(p), i]));
    this.tira = tira;
  }
  att(prefixo) {
    const w = this.w;
    return { in_proj_weight: w[prefixo + ".in_proj_weight"], in_proj_bias: w[prefixo + ".in_proj_bias"],
             out_proj_weight: w[prefixo + ".out_proj.weight"], out_proj_bias: w[prefixo + ".out_proj.bias"] };
  }

  // devolve os tokens e as palavras que o modelo nao conhece
  tokenizar(texto) {
    const L = this.cfg.maxlen, tok = new Int32Array(L), ignoradas = [];
    let n = 0;
    for (const p of texto.toLowerCase().split(/\s+/).filter(Boolean)) {
      const id = this.stoi[p] ?? this.semAcento[this.tira(p)];
      if (id === undefined || id < 2) { ignoradas.push(p); continue; }
      if (n < L) tok[n++] = id;
    }
    if (n === 0) tok[0] = 1;
    return { tok, ignoradas };
  }

  codificarTexto(tok) {
    const { dim: d, heads: H, txt_depth, maxlen: L } = this.cfg, w = this.w;
    const mask = Array.from(tok, (t) => t === 0);
    let x = new Float32Array(L * d);
    for (let i = 0; i < L; i++)
      for (let k = 0; k < d; k++) x[i * d + k] = w["text.emb.weight"][tok[i] * d + k] + w["text.pos"][i * d + k];
    for (let l = 0; l < txt_depth; l++) {
      const p = `text.enc.layers.${l}.`;
      let h = layerNorm(x, L, d, 1e-5, w[p + "norm1.weight"], w[p + "norm1.bias"]);
      h = atencao(h, L, h, L, d, H, this.att(p + "self_attn"), mask);
      for (let i = 0; i < x.length; i++) x[i] += h[i];
      h = layerNorm(x, L, d, 1e-5, w[p + "norm2.weight"], w[p + "norm2.bias"]);
      h = linear(h, L, d, w[p + "linear1.weight"], w[p + "linear1.bias"], 4 * d);
      for (let i = 0; i < h.length; i++) h[i] = geluErf(h[i]);
      h = linear(h, L, 4 * d, w[p + "linear2.weight"], w[p + "linear2.bias"], d);
      for (let i = 0; i < x.length; i++) x[i] += h[i];
    }
    x = layerNorm(x, L, d, 1e-5, w["text.norm.weight"], w["text.norm.bias"]);
    const pooled = new Float32Array(d); let n = 0;
    for (let i = 0; i < L; i++) if (!mask[i]) { n++; for (let k = 0; k < d; k++) pooled[k] += x[i * d + k]; }
    for (let k = 0; k < d; k++) pooled[k] /= n;
    return { txt: x, mask, pooled };
  }

  // x em (3, img, img), t escalar em [0,1], ctx de codificarTexto. Devolve a velocidade prevista.
  forward(x, t, ctx) {
    const { img, ch: C, patch: p, dim: d, depth, heads: H, maxlen: L } = this.cfg, w = this.w;
    const g = img / p, N = g * g, half = d / 2;
    // embedding do tempo
    const te = new Float32Array(d);
    for (let k = 0; k < half; k++) {
      const a = t * 1000 * Math.exp(-Math.log(10000) * k / half);
      te[k] = Math.cos(a); te[half + k] = Math.sin(a);
    }
    let c = linear(te, 1, d, w["t_mlp.0.weight"], w["t_mlp.0.bias"], d).map(silu);
    c = linear(c, 1, d, w["t_mlp.2.weight"], w["t_mlp.2.bias"], d);
    for (let k = 0; k < d; k++) c[k] += ctx.pooled[k];
    const cs = c.map(silu);
    // patches
    const pv = new Float32Array(N * C * p * p);
    for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) {
      const base = (i * g + j) * C * p * p;
      for (let cc = 0; cc < C; cc++) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++)
        pv[base + cc * p * p + a * p + b] = x[cc * img * img + (i * p + a) * img + j * p + b];
    }
    let h = linear(pv, N, C * p * p, w["embed.weight"], w["embed.bias"], d);
    for (let i = 0; i < h.length; i++) h[i] += w["pos"][i];
    for (let l = 0; l < depth; l++) {
      const pr = `blocks.${l}.`;
      const m = linear(cs, 1, d, w[pr + "ada.1.weight"], w[pr + "ada.1.bias"], 6 * d);
      let u = layerNorm(h, N, d, 1e-6);
      for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) u[i * d + k] = u[i * d + k] * (1 + m[d + k]) + m[k];
      u = atencao(u, N, u, N, d, H, this.att(pr + "attn"), null);
      for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) h[i * d + k] += m[2 * d + k] * u[i * d + k];
      u = layerNorm(h, N, d, 1e-5, w[pr + "n2.weight"], w[pr + "n2.bias"]);
      u = atencao(u, N, ctx.txt, L, d, H, this.att(pr + "cross"), ctx.mask);
      for (let i = 0; i < h.length; i++) h[i] += u[i];
      u = layerNorm(h, N, d, 1e-6);
      for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) u[i * d + k] = u[i * d + k] * (1 + m[4 * d + k]) + m[3 * d + k];
      u = linear(u, N, d, w[pr + "mlp.0.weight"], w[pr + "mlp.0.bias"], 4 * d);
      for (let i = 0; i < u.length; i++) u[i] = geluTanh(u[i]);
      u = linear(u, N, 4 * d, w[pr + "mlp.2.weight"], w[pr + "mlp.2.bias"], d);
      for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) h[i * d + k] += m[5 * d + k] * u[i * d + k];
    }
    const f = linear(cs, 1, d, w["fada.1.weight"], w["fada.1.bias"], 2 * d);
    let u = layerNorm(h, N, d, 1e-6);
    for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) u[i * d + k] = u[i * d + k] * (1 + f[d + k]) + f[k];
    u = linear(u, N, d, w["out.weight"], w["out.bias"], p * p * C);
    const v = new Float32Array(C * img * img);
    for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++)
      for (let cc = 0; cc < C; cc++)
        v[cc * img * img + (i * p + a) * img + j * p + b] = u[(i * g + j) * p * p * C + (a * p + b) * C + cc];
    return v;
  }
}

// gerador pseudoaleatorio com semente, ruido normal por Box Muller
export function ruido(n, semente) {
  let s = semente >>> 0 || 1;
  const r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = r(), v = r(), m = Math.sqrt(-2 * Math.log(u));
    out[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return out;
}
