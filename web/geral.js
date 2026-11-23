// Motor do MiniT2I geral no navegador. Baixa os tres modelos ONNX do Hugging Face,
// guarda no cache do navegador e gera com o ONNX Runtime Web, na placa de video por WebGPU.
const MiniGeral = (() => {
  const ORT = window.MINIT2I_ORT || "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
  let ort = null, sess = {}, meta = null, vocab = null, pronto = false, base = "";

  function carregarScript(src) {
    return new Promise((ok, erro) => {
      const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => erro(new Error("falha ao carregar " + src));
      document.head.appendChild(s);
    });
  }

  // quando ja existe uma versao antiga no cache, usa ela na hora e baixa a nova em segundo plano para a proxima visita
  async function baixar(url, versao, aoProgresso, aceitarAntiga, aoUsarAntiga) {
    const chave = url + "?v=" + encodeURIComponent(versao);
    let cache = null;
    try {
      cache = await caches.open("minit2i-geral");
      const hit = await cache.match(chave);
      if (hit) { const b = await hit.arrayBuffer(); aoProgresso(b.byteLength, b.byteLength); return b; }
      if (aceitarAntiga) {
        for (const k of await cache.keys()) {
          if (!k.url.startsWith(url + "?v=")) continue;
          const velho = await cache.match(k); if (!velho) continue;
          const b = await velho.arrayBuffer(); aoProgresso(b.byteLength, b.byteLength);
          if (aoUsarAntiga) aoUsarAntiga(decodeURIComponent(k.url.split("?v=")[1] || ""));
          baixar(url, versao, () => {}, false).catch(() => {});
          return b;
        }
      }
    } catch (e) { cache = null; }
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} while downloading ${url.split("/").pop()}`);
    const total = +r.headers.get("content-length") || 0;
    const leitor = r.body.getReader(); const partes = []; let n = 0;
    for (;;) { const { done, value } = await leitor.read(); if (done) break; partes.push(value); n += value.length; aoProgresso(n, total); }
    const buf = new Uint8Array(n); let o = 0; for (const p of partes) { buf.set(p, o); o += p.length; }
    try { if (cache) { await cache.put(chave, new Response(buf)); for (const k of await cache.keys()) if (k.url.startsWith(url) && k.url !== new Request(chave).url) await cache.delete(k); } } catch (e) { /* sem cache, segue */ }
    return buf.buffer;
  }

  // tokenizador BERT multilingue, mesmo algoritmo do original: limpeza, separacao de CJK e pontuacao, WordPiece
  function ehPont(cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(String.fromCodePoint(cp));
  }
  function ehCJK(cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
  }
  function ehEspaco(ch) { return ch === " " || ch === "\t" || ch === "\n" || ch === "\r" || /\p{Zs}/u.test(ch); }
  function ehControle(ch) { if (ch === "\t" || ch === "\n" || ch === "\r") return false; return /[\p{Cc}\p{Cf}]/u.test(ch); }

  function palavrasBasicas(texto, minusculas) {
    let limpo = "";
    for (const ch of texto) {
      const cp = ch.codePointAt(0);
      if (cp === 0 || cp === 0xFFFD || ehControle(ch)) continue;
      if (ehEspaco(ch)) limpo += " ";
      else if (ehCJK(cp)) limpo += " " + ch + " ";
      else limpo += ch;
    }
    const out = [];
    for (let tok of limpo.trim().split(/ +/).filter(Boolean)) {
      if (minusculas) tok = tok.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
      let atual = "";
      for (const ch of tok) {
        if (ehPont(ch.codePointAt(0))) { if (atual) out.push(atual); out.push(ch); atual = ""; } else atual += ch;
      }
      if (atual) out.push(atual);
    }
    return out;
  }

  function wordpiece(palavra) {
    const chars = Array.from(palavra);
    if (chars.length > 100) return [meta.tokenizador.unk];
    const ids = []; let ini = 0;
    while (ini < chars.length) {
      let fim = chars.length, achou = -1;
      while (ini < fim) {
        let sub = chars.slice(ini, fim).join(""); if (ini > 0) sub = "##" + sub;
        const id = vocab.get(sub);
        if (id !== undefined) { achou = id; break; }
        fim--;
      }
      if (achou < 0) return [meta.tokenizador.unk];
      ids.push(achou); ini = fim;
    }
    return ids;
  }

  function tokenizar(texto) {
    const tk = meta.tokenizador, L = meta.maxlen;
    let ids = [tk.cls];
    for (const p of palavrasBasicas(texto, tk.minusculas)) ids.push(...wordpiece(p));
    ids = ids.slice(0, L - 1); ids.push(tk.sep);
    const out = new BigInt64Array(L), am = new BigInt64Array(L);
    ids.forEach((v, i) => { out[i] = BigInt(v); am[i] = 1n; });
    for (let i = ids.length; i < L; i++) out[i] = BigInt(tk.pad);
    return { ids: out, am, n: ids.length };
  }

  async function iniciar(repo, aoEstado) {
    if (pronto) return meta;
    base = /^(https?:|\/)/.test(repo) ? repo : `https://huggingface.co/${repo}/resolve/main/web/`;
    aoEstado("loading ONNX Runtime");
    if (!window.ort) await carregarScript(ORT + "ort.webgpu.min.js");
    ort = window.ort; ort.env.wasm.wasmPaths = ORT;
    meta = await (await fetch(base + "meta.json?t=" + Date.now(), { cache: "no-store" })).json();
    const gpu = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    if (!gpu && meta.dtype === "fp16") throw new Error("the general model needs WebGPU. Use an up to date Chrome or Edge on a computer.");
    const arquivos = [["texto.onnx", "texto"], ["dit.onnx", "dit"], ["vae.onnx", "vae"]];
    const tam = meta.tamanhos || {}; const totalMB = Object.values(tam).reduce((a, b) => a + b, 0) / 1e6;
    const feito = {};
    const prog = (nome) => (n, t) => { feito[nome] = n; const mb = Object.values(feito).reduce((a, b) => a + b, 0) / 1e6;
      aoEstado(`downloading the general model, ${Math.round(mb)} of ${Math.round(totalMB || mb)} MB, first visit only`); };
    const [vtxt, ...bufs] = await Promise.all([
      fetch(base + "vocab.txt?v=" + meta.tokenizador_versao).then((r) => r.text()),
      ...arquivos.map(([f]) => f === "dit.onnx"
        ? baixar(base + f, meta.versao, prog(f), true, (v) => { if (/^\d+$/.test(v)) meta.passo = +v; })
        : baixar(base + f, "fixo", prog(f))),
    ]);
    vocab = new Map(vtxt.split("\n").map((w, i) => [w, i]));
    aoEstado("preparing the network on your GPU");
    const ep = gpu ? ["webgpu", "wasm"] : ["wasm"];
    sess.texto = await ort.InferenceSession.create(bufs[0], { executionProviders: ["wasm"] });
    sess.dit = await ort.InferenceSession.create(bufs[1], { executionProviders: ep });
    sess.vae = await ort.InferenceSession.create(bufs[2], { executionProviders: ep });
    pronto = true;
    return meta;
  }

  async function codificar(textos) {
    const L = meta.maxlen, D = meta.cfg.txt_dim, n = textos.length;
    const ids = new BigInt64Array(n * L), am = new BigInt64Array(n * L), mask = new Uint8Array(n * L), palavras = [];
    textos.forEach((t, i) => { const r = tokenizar(t); ids.set(r.ids, i * L); am.set(r.am, i * L); for (let j = 0; j < L; j++) mask[i * L + j] = r.am[j] === 0n ? 1 : 0; });
    const o = await sess.texto.run({ ids: new ort.Tensor("int64", ids, [n, L]), am: new ort.Tensor("int64", am, [n, L]) });
    return { h: o.h.data, pooled: o.pooled.data, mask, L, D };
  }

  function ruido(n, semente) {
    let s = (semente >>> 0) || 1;
    const r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; };
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) { const u = r(), v = r(), m = Math.sqrt(-2 * Math.log(u)); out[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v); }
    return out;
  }

  async function decodificar(z, n) {
    const c = meta.cfg; const o = await sess.vae.run({ z: new ort.Tensor("float32", z, [n, c.lat_ch, c.lat_res, c.lat_res]) });
    const img = o.img.data, S = o.img.dims[2], imgs = [];
    for (let b = 0; b < n; b++) {
      const rgba = new Uint8ClampedArray(S * S * 4);
      for (let i = 0; i < S * S; i++) { for (let k = 0; k < 3; k++) rgba[i * 4 + k] = (img[((b * 3 + k) * S * S) + i] + 1) * 127.5; rgba[i * 4 + 3] = 255; }
      imgs.push(rgba);
    }
    return { imgs, S };
  }

  // gera n imagens ao mesmo tempo, com previa periodica da estimativa da imagem final
  async function gerar(prompt, { n = 4, passos = 25, cfg = 4.5, semente = 0 }, aoPasso) {
    const c = meta.cfg, Z = c.lat_ch * c.lat_res * c.lat_res;
    const enc = await codificar([prompt, ""]);
    const { L, D } = enc, B = 2 * n;
    const txt = new Float32Array(B * L * D), pooled = new Float32Array(B * D), tmask = new Uint8Array(B * L);
    for (let b = 0; b < B; b++) {
      const src = b < n ? 0 : 1;
      txt.set(enc.h.subarray(src * L * D, (src + 1) * L * D), b * L * D);
      pooled.set(enc.pooled.subarray(src * D, (src + 1) * D), b * D);
      tmask.set(enc.mask.subarray(src * L, (src + 1) * L), b * L);
    }
    const tTxt = new ort.Tensor("float32", txt, [B, L, D]), tPool = new ort.Tensor("float32", pooled, [B, D]), tMask = new ort.Tensor("bool", tmask, [B, L]);
    let x = new Float32Array(n * Z);
    for (let b = 0; b < n; b++) x.set(ruido(Z, semente + b * 7919), b * Z);
    const cada = Math.max(1, Math.round(passos / 6));
    for (let i = 0; i < passos; i++) {
      const t = 1 - i / passos, t2 = 1 - (i + 1) / passos;
      const xx = new Float32Array(B * Z); xx.set(x); xx.set(x, n * Z);
      const o = await sess.dit.run({ x: new ort.Tensor("float32", xx, [B, c.lat_ch, c.lat_res, c.lat_res]), t: new ort.Tensor("float32", new Float32Array(B).fill(t), [B]), txt: tTxt, tmask: tMask, pooled: tPool });
      const v = o.v.data, prev = new Float32Array(n * Z);
      for (let k = 0; k < n * Z; k++) { const vv = v[n * Z + k] + cfg * (v[k] - v[n * Z + k]); prev[k] = x[k] - t * vv; x[k] += (t2 - t) * vv; }
      const fim = i === passos - 1;
      if (fim || (i + 1) % cada === 0) { const d = await decodificar(fim ? x : prev, n); await aoPasso(i + 1, passos, d.imgs, d.S, fim); }
      else await aoPasso(i + 1, passos, null, 0, false);
    }
  }

  return { iniciar, gerar, tokenizar: (t) => tokenizar(t), get meta() { return meta; }, get pronto() { return pronto; }, _palavras: palavrasBasicas };
})();
