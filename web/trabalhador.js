// Web Worker, gera uma imagem por vez. Recebe os pesos uma vez e depois pedidos de geracao.
// montar.py junta este arquivo ao motor.js num unico script.
import { MiniT2I, ruido } from "./motor.js";

let modelo = null;

function paraRGBA(x, S) {
  const out = new Uint8ClampedArray(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    for (let c = 0; c < 3; c++) out[i * 4 + c] = Math.round((Math.max(-1, Math.min(1, x[c * S * S + i])) + 1) * 127.5);
    out[i * 4 + 3] = 255;
  }
  return out;
}

self.onmessage = (e) => {
  const d = e.data;
  if (d.tipo === "iniciar") {
    modelo = new MiniT2I(d.cfg, d.pesos);
    self.postMessage({ tipo: "pronto" });
    return;
  }
  if (d.tipo !== "gerar") return;
  const S = modelo.cfg.img, n = 3 * S * S;
  const cond = modelo.codificarTexto(Int32Array.from(d.tok));
  const vazio = new Int32Array(modelo.cfg.maxlen); vazio[0] = 1;
  const inc = modelo.codificarTexto(vazio);
  const z = ruido(n, d.semente);
  // geracao do zero comeca em t = 1. Variacao parte de uma imagem pronta misturada com ruido.
  const t0 = d.base ? d.forca : 1;
  let x = new Float32Array(n);
  if (d.base) {
    for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) x[c * S * S + i] = d.base[i * 4 + c] / 127.5 - 1;
    for (let k = 0; k < n; k++) x[k] = (1 - t0) * x[k] + t0 * z[k];
  } else x = z;
  const passos = Math.max(1, Math.round(d.passos * t0));
  const inicio = performance.now();
  for (let i = 0; i < passos; i++) {
    const t = t0 * (1 - i / passos), t2 = t0 * (1 - (i + 1) / passos);
    const vc = modelo.forward(x, t, cond);
    const vu = d.guidance === 1 ? vc : modelo.forward(x, t, inc);
    const prev = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const v = vu[k] + d.guidance * (vc[k] - vu[k]);
      prev[k] = x[k] - t * v;
      x[k] += (t2 - t) * v;
    }
    const fim = i === passos - 1;
    self.postMessage({ tipo: "passo", id: d.id, idx: d.idx, passo: i + 1, total: passos,
                       rgba: paraRGBA(fim ? x : prev, S), fim, ms: performance.now() - inicio });
  }
};
