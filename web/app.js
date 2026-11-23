// Interface. Carrega os pesos, distribui as imagens entre os nucleos e monta o feed de trabalhos.
(() => {
  const $ = (id) => document.getElementById(id);
  const NUC = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2)));
  const CHAVE = "minit2i.historico.v1";
  let cfg = null, tokenizador = null, trabalhadores = [], livres = [], fila = [], trabalhos = new Map(), contador = 0;

  // tokenizacao igual a do motor, repetida aqui para mostrar as palavras ignoradas antes de gerar
  const tira = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "");
  function tokenizar(texto) {
    const L = cfg.maxlen, tok = new Array(L).fill(0), palavras = [];
    let n = 0;
    for (const p of texto.toLowerCase().split(/\s+/).filter(Boolean)) {
      const id = tokenizador.stoi[p] ?? tokenizador.sem[tira(p)];
      const ok = id !== undefined && id >= 2 && n < L;
      if (ok) tok[n++] = id;
      palavras.push({ p, ok });
    }
    if (n === 0) tok[0] = 1;
    return { tok, palavras };
  }

  function estado(txt, pulsando = true) {
    $("estado-txt").textContent = txt;
    $("estado").querySelector(".ponto").style.visibility = pulsando ? "visible" : "hidden";
  }

  // vocabulario em grupos, cada palavra vira um botao que entra no prompt
  function montarVocab() {
    const nomes = { formas: "shapes", cores: "colors", tamanhos: "sizes", posicoes: "positions" };
    $("vocab").innerHTML = "";
    for (const [k, lista] of Object.entries(cfg.grupos)) {
      const div = document.createElement("div");
      div.innerHTML = `<span class="olho">${nomes[k] || k}</span><div class="chips"></div>`;
      for (const w of lista) {
        const b = document.createElement("button");
        b.className = "chip"; b.type = "button"; b.textContent = w;
        b.onclick = () => { const i = $("prompt"); i.value = (i.value.trim() + " " + w).trim(); i.focus(); };
        div.querySelector(".chips").appendChild(b);
      }
      $("vocab").appendChild(div);
    }
  }

  function montarSugestoes() {
    const exemplos = modo === "geral" ? SUG_GERAL : [["um triângulo azul grande no centro", "Never seen"], ["um losango roxo pequeno no topo", "New"],
                      ["um círculo laranja grande à direita", "Popular"]];
    const box = $("sugestoes"); box.innerHTML = "";
    for (const [p, selo] of exemplos) {
      const b = document.createElement("button"); b.type = "button"; b.className = "sug";
      b.innerHTML = `<span class="t">/ ${esc(p)}</span><span class="selo${selo === "Popular" ? " quente" : ""}">${selo}</span>`;
      b.onclick = () => { $("prompt").value = p; $("prompt").focus(); };
      box.appendChild(b);
    }
    const mais = document.createElement("button"); mais.type = "button"; mais.className = "sug";
    mais.innerHTML = `<span class="t">More words</span><span aria-hidden="true">›</span>`;
    mais.onclick = () => { $("vocab").hidden = !$("vocab").hidden; };
    if (modo !== "geral") box.appendChild(mais);
  }

  function lembrarSessao(prompt) {
    const box = $("sessoes"); box.querySelector(".vazio-s")?.remove();
    const b = document.createElement("button"); b.type = "button"; b.textContent = prompt; b.title = prompt;
    b.onclick = () => { $("prompt").value = prompt; $("prompt").focus(); window.scrollTo({ top: 0, behavior: "smooth" }); };
    box.prepend(b);
    while (box.children.length > 14) box.lastChild.remove();
  }

  function surpresa() {
    if (modo === "geral") return SURPRESA_GERAL[Math.floor(Math.random() * SURPRESA_GERAL.length)];
    const g = cfg.grupos, r = (a) => a[Math.floor(Math.random() * a.length)];
    return `um ${r(g.formas)} ${r(g.cores)} ${r(g.tamanhos)} ${r(g.posicoes)}`;
  }

  // fila de imagens, cada trabalhador pega a proxima quando fica livre
  function despachar() {
    while (livres.length && fila.length) {
      const w = livres.pop(), job = fila.shift();
      w.ocupado = job; w.postMessage(job.msg);
    }
  }

  function aoMensagem(w, e) {
    const d = e.data;
    if (d.tipo === "pronto") { livres.push(w); despachar(); return; }
    if (d.tipo !== "passo") return;
    const t = trabalhos.get(d.id); if (!t) return;
    const cel = t.cels[d.idx];
    cel.ctx.putImageData(new ImageData(d.rgba, cfg.img, cfg.img), 0, 0);
    cel.pct.textContent = d.fim ? "" : Math.round(100 * d.passo / d.total) + "%";
    t.prog[d.idx] = d.passo / d.total;
    t.barra.style.width = (100 * t.prog.reduce((a, b) => a + b, 0) / 4) + "%";
    if (d.fim) {
      cel.el.classList.remove("gerando"); cel.pronta = true;
      w.ocupado = null; livres.push(w); despachar();
      if (t.cels.every((c) => c.pronta)) concluir(t);
    }
  }

  function concluir(t) {
    const s = ((performance.now() - t.inicio) / 1000).toFixed(1).replace(".", ",");
    t.tempo.textContent = `done in ${s} s`;
    t.barra.parentElement.hidden = true;
    t.botoes.forEach((b) => (b.disabled = false));
    const m = $("miniatura"); m.textContent = "";
    const c = document.createElement("canvas"); c.width = c.height = t.S;
    c.getContext("2d").drawImage(t.cels[0].canvas, 0, 0); m.appendChild(c);
    salvarHistorico();
  }

  function criarCartao(dados) {
    $("vazio-feed").hidden = true;
    const el = document.createElement("article");
    el.className = "trab";
    const promptHTML = dados.palavras.map(({ p, ok }) => ok ? esc(p) : `<span class="ign" title="word the model does not know yet">${esc(p)}</span>`).join(" ");
    const ign = 0;
    el.innerHTML = `
      <div class="bolha">${promptHTML || "<span class='ign'>empty prompt</span>"}</div>
      <div class="resposta">
      <div class="grade"></div>
      <div class="info">
        <div class="meta num"><span>${dados.tipo}</span><span>seed ${dados.sementes[0]}</span><span>${dados.passos} steps</span><span>guidance ${dados.guidance}</span><span class="tempo">queued</span></div>
        ${ign ? `<div class="nota">Crossed out words are not in the vocabulary yet and were ignored.</div>` : ""}
        <div class="barra"><i style="width:0"></i></div>
        <div class="botoes">
          <button class="bt" data-a="refazer" disabled>Redo</button>
          <button class="bt" data-a="editar" disabled>Edit</button>
        </div>
      </div>
      </div>`;
    const grade = el.querySelector(".grade");
    const S = dados.tam || cfg.img;
    const t = { id: ++contador, el, dados, S, cels: [], prog: [0, 0, 0, 0], inicio: performance.now(),
                barra: el.querySelector(".barra i"), tempo: el.querySelector(".tempo"), botoes: [...el.querySelectorAll(".botoes .bt")] };
    for (let i = 0; i < 4; i++) {
      const c = document.createElement("div"); c.className = "cel gerando";
      c.innerHTML = `<canvas width="${S}" height="${S}"${dados.modelo === "geral" ? ' style="image-rendering:auto"' : ""}></canvas><span class="pct"></span>`;
      const cv = c.querySelector("canvas");
      const cel = { el: c, ctx: cv.getContext("2d"), canvas: cv, pct: c.querySelector(".pct"), pronta: false };
      c.onclick = () => cel.pronta && abrir(t, i);
      t.cels.push(cel); grade.appendChild(c);
    }
    el.querySelector('[data-a="refazer"]').onclick = () => dados.modelo === "geral" ? novoTrabalhoGeral(dados.prompt) : novoTrabalho(dados.prompt);
    el.querySelector('[data-a="editar"]').onclick = () => { $("prompt").value = dados.prompt; $("prompt").focus(); };
    $("feed").append(el);
    trabalhos.set(t.id, t);
    lembrarSessao(dados.prompt);
    return t;
  }

  function lerAjustes() {
    const s = parseInt($("semente").value, 10);
    return { passos: +$("passos").value, guidance: +$("guidance").value, semente: Number.isFinite(s) ? s : Math.floor(Math.random() * 1e9) };
  }

  function novoTrabalho(prompt, base = null) {
    const { tok, palavras } = tokenizar(prompt);
    const a = lerAjustes();
    const sementes = [0, 1, 2, 3].map((k) => a.semente + k * 7919);
    const t = criarCartao({ prompt, palavras, sementes, passos: a.passos, guidance: a.guidance, tipo: base ? "variation" : "generation" });
    t.tempo.textContent = "generating";
    t.cels.forEach((cel, i) => {
      if (base) cel.ctx.putImageData(new ImageData(new Uint8ClampedArray(base), cfg.img, cfg.img), 0, 0);
      fila.push({ msg: { tipo: "gerar", id: t.id, idx: i, tok, semente: sementes[i], passos: a.passos, guidance: a.guidance,
                         base: base ? new Uint8ClampedArray(base) : null, forca: 0.6 } });
    });
    despachar();
  }

  // modelo geral, roda na placa de video, um trabalho por vez
  let filaGeral = Promise.resolve(), modo = "formas";
  async function garantirGeral() {
    if (MiniGeral.pronto) return;
    const m = await MiniGeral.iniciar(GERAL_REPO, (txt) => estado(txt));
    estado(`training step ${m.passo.toLocaleString("en-US")}`, false);
  }
  function novoTrabalhoGeral(prompt) {
    const a = lerAjustes();
    const palavras = prompt.split(/\s+/).filter(Boolean).map((p) => ({ p, ok: true }));
    const t = criarCartao({ prompt, palavras, sementes: [a.semente], passos: a.passos, guidance: a.guidance, tipo: "general, preview", modelo: "geral", tam: 128 });
    t.tempo.textContent = "queued";
    filaGeral = filaGeral.then(async () => {
      try {
        t.tempo.textContent = "loading the model";
        await garantirGeral();
        t.inicio = performance.now(); t.tempo.textContent = "generating";
        await MiniGeral.gerar(prompt, { n: 4, passos: a.passos, cfg: a.guidance, semente: a.semente }, async (p, total, imgs, S, fim) => {
          t.barra.style.width = (100 * p / total) + "%";
          t.cels.forEach((c) => (c.pct.textContent = fim ? "" : Math.round(100 * p / total) + "%"));
          if (imgs) imgs.forEach((rgba, i) => {
            const c = t.cels[i];
            if (c.canvas.width !== S) { c.canvas.width = c.canvas.height = S; t.S = S; }
            c.ctx.putImageData(new ImageData(rgba, S, S), 0, 0);
          });
          if (fim) { t.cels.forEach((c) => { c.pronta = true; c.el.classList.remove("gerando"); }); concluir(t); }
          await new Promise((r) => setTimeout(r, 0));
        });
      } catch (e) {
        console.error(e);
        t.tempo.textContent = "could not generate"; t.barra.parentElement.hidden = true;
        t.botoes.forEach((b) => (b.disabled = false));
        estado(e.message || String(e), false);
      }
    });
  }

  const SUG_GERAL = [["a dog running on the grass", "New"], ["a beach with palm trees at sunset", "Popular"], ["um campo de flores amarelas", "Portuguese"]];
  const SURPRESA_GERAL = ["a wooden house in the forest", "mountains covered in snow", "a cat sleeping on the sofa", "a red car on a city street",
    "a plate of tropical fruits", "a lighthouse at night", "a field of yellow flowers", "a bowl of ramen", "a city at night with lights", "a boat on the lake"];
  function mudarModo(novo) {
    if (novo === "geral" && !GERAL_REPO) return;
    modo = novo;
    $("aba-formas").classList.toggle("on", modo === "formas"); $("aba-geral").classList.toggle("on", modo === "geral");
    $("modelo").value = modo;
    $("prompt").placeholder = modo === "geral" ? "Describe an image" : "um triângulo azul grande no centro";
    montarSugestoes();
    if (modo === "geral") $("vocab").hidden = true;
  }

  // ampliacao, download e variacoes
  let aberto = null;
  function abrir(t, i) {
    aberto = { t, i };
    const gc = $("grande"); gc.width = gc.height = t.S;
    gc.style.imageRendering = t.dados.modelo === "geral" ? "auto" : "pixelated";
    gc.getContext("2d").putImageData(t.cels[i].ctx.getImageData(0, 0, t.S, t.S), 0, 0);
    $("caixa-txt").textContent = t.dados.prompt;
    $("caixa").hidden = false;
  }
  $("fechar").onclick = () => ($("caixa").hidden = true);
  $("caixa").onclick = (e) => { if (e.target === $("caixa")) $("caixa").hidden = true; };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("caixa").hidden = true; });
  $("baixar").onclick = () => {
    const c = document.createElement("canvas"); c.width = c.height = 512;
    const g = c.getContext("2d"); g.imageSmoothingEnabled = false;
    g.drawImage($("grande"), 0, 0, 512, 512);
    const a = document.createElement("a");
    a.download = "minit2i " + aberto.t.dados.prompt.slice(0, 40) + ".png";
    a.href = c.toDataURL("image/png"); a.click();
  };
  $("variar").onclick = () => {
    const { t, i } = aberto; $("caixa").hidden = true;
    if (t.dados.modelo === "geral") novoTrabalhoGeral(t.dados.prompt);
    else novoTrabalho(t.dados.prompt, t.cels[i].ctx.getImageData(0, 0, cfg.img, cfg.img).data);
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };

  // historico local, so as imagens prontas, como PNG pequeno
  function salvarHistorico() {
    try {
      const itens = [...trabalhos.values()].filter((t) => t.cels.every((c) => c.pronta)).slice(-24).map((t) => ({
        d: t.dados, img: t.cels.map((c) => c.canvas.toDataURL()) }));
      localStorage.setItem(CHAVE, JSON.stringify(itens));
    } catch (e) { /* armazenamento indisponivel, segue sem historico */ }
  }
  function carregarHistorico() {
    let itens = [];
    try { itens = JSON.parse(localStorage.getItem(CHAVE) || "[]"); } catch (e) { return 0; }
    for (const it of itens) {
      const t = criarCartao(it.d);
      it.img.forEach((src, i) => {
        const im = new Image();
        im.onload = () => t.cels[i].ctx.drawImage(im, 0, 0);
        im.src = src; t.cels[i].pronta = true; t.cels[i].el.classList.remove("gerando");
      });
      t.tempo.textContent = "from your history"; t.barra.parentElement.hidden = true;
      t.botoes.forEach((b) => (b.disabled = false));
    }
    return itens.length;
  }

  function esc(s) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

  $("passos").oninput = () => ($("passos-v").textContent = $("passos").value);
  $("guidance").oninput = () => ($("guidance-v").textContent = $("guidance").value.replace(".", ","));
  $("prompt").addEventListener("input", () => { const p = $("prompt"); p.style.height = "auto"; p.style.height = p.scrollHeight + "px"; });
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!$("gerar").disabled) $("form").requestSubmit(); }
  });
  $("bt-ajustes").onclick = (e) => {
    e.stopPropagation();
    const ab = $("painel").hidden; $("painel").hidden = !ab; $("bt-ajustes").setAttribute("aria-expanded", String(ab));
  };
  $("painel").onclick = (e) => e.stopPropagation();
  document.addEventListener("click", () => { $("painel").hidden = true; $("bt-ajustes").setAttribute("aria-expanded", "false"); });
  $("aba-galeria").onclick = () => $("galeria").scrollIntoView({ behavior: "smooth" });
  $("aba-gerar").onclick = () => { window.scrollTo({ top: 0, behavior: "smooth" }); $("prompt").focus(); };
  document.querySelector("[data-foco]").onclick = (e) => { e.preventDefault(); $("prompt").focus(); };
  $("aba-formas").onclick = () => mudarModo("formas");
  $("aba-geral").onclick = () => mudarModo("geral");
  $("modelo").onchange = () => mudarModo($("modelo").value);
  if (GERAL_REPO) { $("aba-geral").disabled = false; $("modelo").querySelector('option[value="geral"]').disabled = false; }
  $("surpresa").onclick = () => { $("prompt").value = surpresa(); $("prompt").focus(); };
  $("form").onsubmit = (e) => {
    e.preventDefault();
    const p = $("prompt").value.trim() || surpresa();
    modo === "geral" ? novoTrabalhoGeral(p) : novoTrabalho(p);
    $("prompt").value = ""; $("prompt").style.height = "auto";
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };

  async function iniciar() {
    try {
      const [rc, rp] = await Promise.all([fetch("modelo.json"), fetch("pesos.b64.txt")]);
      cfg = await rc.json();
      estado("decoding the weights");
      const b64 = await rp.text();
      const fonte = document.getElementById("fonte-trabalhador").textContent;
      const url = URL.createObjectURL(new Blob([fonte], { type: "text/javascript" }));
      tokenizador = { stoi: Object.fromEntries(cfg.vocab.map((p, i) => [p, i])), sem: Object.fromEntries(cfg.vocab.map((p, i) => [tira(p), i])) };
      montarVocab(); montarSugestoes();
      // decodifica uma vez aqui e manda uma copia para cada nucleo
      const bin = atob(b64.trim()), u16 = new Uint16Array(bin.length >> 1);
      for (let i = 0; i < u16.length; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      const pesos = f16ParaF32(u16);
      for (let k = 0; k < NUC; k++) {
        const w = new Worker(url);
        w.onmessage = (e) => aoMensagem(w, e);
        w.postMessage({ tipo: "iniciar", cfg, pesos: pesos.slice() });
        trabalhadores.push(w);
      }
      $("nucleos").textContent = `${NUC} ${NUC > 1 ? "cores" : "core"} in use`;
      $("prompt").disabled = false; $("gerar").disabled = false;
      estado("", false);
      if (GERAL_REPO) mudarModo("geral"); else mudarModo("formas");
      carregarHistorico();
      window.scrollTo({ top: document.body.scrollHeight });
    } catch (e) {
      estado("could not load the model. Serve the page over HTTP instead of opening it as a local file.", false);
      console.error(e);
    }
  }

  function f16ParaF32(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  iniciar();
})();
