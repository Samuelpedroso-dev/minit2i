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
    const nomes = { formas: "formas", cores: "cores", tamanhos: "tamanhos", posicoes: "posições" };
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
    const exemplos = [["um triângulo azul grande no centro", "Nunca visto"], ["um losango roxo pequeno no topo", "Novo"],
                      ["um círculo laranja grande à direita", "Popular"]];
    const box = $("sugestoes"); box.innerHTML = "";
    for (const [p, selo] of exemplos) {
      const b = document.createElement("button"); b.type = "button"; b.className = "sug";
      b.innerHTML = `<span class="t">/ ${esc(p)}</span><span class="selo${selo === "Popular" ? " quente" : ""}">${selo}</span>`;
      b.onclick = () => { $("prompt").value = p; $("prompt").focus(); };
      box.appendChild(b);
    }
    const mais = document.createElement("button"); mais.type = "button"; mais.className = "sug";
    mais.innerHTML = `<span class="t">Mais palavras</span><span aria-hidden="true">›</span>`;
    mais.onclick = () => { $("vocab").hidden = !$("vocab").hidden; };
    box.appendChild(mais);
  }

  function lembrarSessao(prompt) {
    const box = $("sessoes"); box.querySelector(".vazio-s")?.remove();
    const b = document.createElement("button"); b.type = "button"; b.textContent = prompt; b.title = prompt;
    b.onclick = () => { $("prompt").value = prompt; $("prompt").focus(); window.scrollTo({ top: 0, behavior: "smooth" }); };
    box.prepend(b);
    while (box.children.length > 14) box.lastChild.remove();
  }

  function surpresa() {
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
    t.tempo.textContent = `pronto em ${s} s`;
    t.barra.parentElement.hidden = true;
    t.botoes.forEach((b) => (b.disabled = false));
    const m = $("miniatura"); m.textContent = "";
    const c = document.createElement("canvas"); c.width = c.height = cfg.img;
    c.getContext("2d").drawImage(t.cels[0].canvas, 0, 0); m.appendChild(c);
    salvarHistorico();
  }

  function criarCartao(dados) {
    $("vazio-feed").hidden = true;
    const el = document.createElement("article");
    el.className = "trab";
    const promptHTML = dados.palavras.map(({ p, ok }) => ok ? esc(p) : `<span class="ign" title="palavra que o modelo ainda não conhece">${esc(p)}</span>`).join(" ");
    const ign = dados.palavras.filter((x) => !x.ok).length;
    el.innerHTML = `
      <div class="grade"></div>
      <div class="info">
        <div class="prompt">${promptHTML || "<span class='ign'>prompt vazio</span>"}</div>
        <div class="meta num"><span>${dados.tipo}</span><span>semente ${dados.sementes[0]}</span><span>${dados.passos} passos</span><span>guidance ${String(dados.guidance).replace(".", ",")}</span><span class="tempo">na fila</span></div>
        ${ign ? `<div class="nota">As palavras riscadas ainda não fazem parte do vocabulário e foram ignoradas.</div>` : ""}
        <div class="barra"><i style="width:0"></i></div>
        <div class="botoes">
          <button class="bt" data-a="refazer" disabled>Refazer</button>
          <button class="bt" data-a="editar" disabled>Editar prompt</button>
        </div>
      </div>`;
    const grade = el.querySelector(".grade");
    const t = { id: ++contador, el, dados, cels: [], prog: [0, 0, 0, 0], inicio: performance.now(),
                barra: el.querySelector(".barra i"), tempo: el.querySelector(".tempo"), botoes: [...el.querySelectorAll(".botoes .bt")] };
    for (let i = 0; i < 4; i++) {
      const c = document.createElement("div"); c.className = "cel gerando";
      c.innerHTML = `<canvas width="${cfg.img}" height="${cfg.img}"></canvas><span class="pct"></span>`;
      const cv = c.querySelector("canvas");
      const cel = { el: c, ctx: cv.getContext("2d"), canvas: cv, pct: c.querySelector(".pct"), pronta: false };
      c.onclick = () => cel.pronta && abrir(t, i);
      t.cels.push(cel); grade.appendChild(c);
    }
    el.querySelector('[data-a="refazer"]').onclick = () => novoTrabalho(dados.prompt);
    el.querySelector('[data-a="editar"]').onclick = () => { $("prompt").value = dados.prompt; $("prompt").focus(); };
    $("feed").prepend(el);
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
    const t = criarCartao({ prompt, palavras, sementes, passos: a.passos, guidance: a.guidance, tipo: base ? "variação" : "geração" });
    t.tempo.textContent = "gerando";
    t.cels.forEach((cel, i) => {
      if (base) cel.ctx.putImageData(new ImageData(new Uint8ClampedArray(base), cfg.img, cfg.img), 0, 0);
      fila.push({ msg: { tipo: "gerar", id: t.id, idx: i, tok, semente: sementes[i], passos: a.passos, guidance: a.guidance,
                         base: base ? new Uint8ClampedArray(base) : null, forca: 0.6 } });
    });
    despachar();
  }

  // ampliacao, download e variacoes
  let aberto = null;
  function abrir(t, i) {
    aberto = { t, i };
    const g = $("grande").getContext("2d");
    g.putImageData(t.cels[i].ctx.getImageData(0, 0, cfg.img, cfg.img), 0, 0);
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
    novoTrabalho(t.dados.prompt, t.cels[i].ctx.getImageData(0, 0, cfg.img, cfg.img).data);
    $("galeria").scrollIntoView({ behavior: "smooth", block: "start" });
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
      t.tempo.textContent = "do seu histórico"; t.barra.parentElement.hidden = true;
      t.botoes.forEach((b) => (b.disabled = false));
    }
    return itens.length;
  }

  function esc(s) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

  $("passos").oninput = () => ($("passos-v").textContent = $("passos").value);
  $("guidance").oninput = () => ($("guidance-v").textContent = $("guidance").value.replace(".", ","));
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
  $("surpresa").onclick = () => { $("prompt").value = surpresa(); $("prompt").focus(); };
  $("form").onsubmit = (e) => {
    e.preventDefault();
    const p = $("prompt").value.trim() || surpresa();
    $("prompt").value = p; novoTrabalho(p);
    $("galeria").scrollIntoView({ behavior: "smooth", block: "start" });
  };

  async function iniciar() {
    try {
      const [rc, rp] = await Promise.all([fetch("modelo.json"), fetch("pesos.b64.txt")]);
      cfg = await rc.json();
      estado("decodificando os pesos");
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
      $("nucleos").textContent = `${NUC} ${NUC > 1 ? "núcleos" : "núcleo"} em uso`;
      $("prompt").disabled = false; $("gerar").disabled = false;
      estado(`modelo pronto, ${cfg.parametros.toLocaleString("pt-BR")} parâmetros carregados`, false);
      if (!carregarHistorico()) novoTrabalho("um círculo vermelho grande à esquerda");
    } catch (e) {
      estado("não foi possível carregar o modelo. Abra a página por um servidor, não como arquivo local.", false);
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
