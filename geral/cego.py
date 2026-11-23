"""Monta uma página para avaliação humana às cegas entre dois experimentos.

Pega duas pastas do avaliar.py, junta as imagens do mesmo prompt e da mesma semente, sorteia qual
fica na esquerda e gera um único .html que funciona offline. Quem avalia não sabe qual é qual.
No fim a página mostra o placar por categoria e baixa um .json com os votos.

    python cego.py avaliacao/e0 avaliacao/e1 --saida cego_e0_e1.html
"""
import argparse, base64, json, pathlib, random

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--saida", default="cego.html")
ap.add_argument("--semente", type=int, default=0)
ap.add_argument("--max", type=int, default=0, help="limita o número de pares, 0 usa todos")
a = ap.parse_args()

pa, pb = pathlib.Path(a.a), pathlib.Path(a.b)
ma, mb = json.load(open(pa / "metricas.json")), json.load(open(pb / "metricas.json"))
em_b = {(i["prompt"], i["semente"]): i for i in mb["imagens"]}
rng = random.Random(a.semente)
pares = []
for i in ma["imagens"]:
    j = em_b.get((i["prompt"], i["semente"]))
    if j is None:
        continue
    img = lambda p, x: "data:image/png;base64," + base64.b64encode((p / x["arquivo"]).read_bytes()).decode()
    troca = rng.random() < 0.5
    esq, dir_ = (img(pb, j), img(pa, i)) if troca else (img(pa, i), img(pb, j))
    pares.append({"c": i["categoria"], "p": i["prompt"], "s": i["semente"], "e": esq, "d": dir_, "t": troca})
rng.shuffle(pares)
if a.max:
    pares = pares[:a.max]

nomes = {"a": pa.name, "b": pb.name}
html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MiniT2I blind evaluation</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:24px;text-align:center}
.img{display:inline-block;margin:10px;cursor:pointer;border:3px solid transparent;border-radius:8px}
.img:hover{border-color:#7aa2ff} img{width:320px;height:320px;image-rendering:pixelated;border-radius:6px;display:block}
button{background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:8px 16px;margin:6px;cursor:pointer;font-size:15px}
#prompt{font-size:22px;margin:16px} .pouco{color:#999;font-size:14px} table{margin:16px auto;border-collapse:collapse}
td,th{padding:6px 14px;border-bottom:1px solid #333}
</style></head><body>
<h2>Which image matches the prompt better?</h2><div class="pouco">click an image or use the keys 1 (left), 2 (right) and 0 (tie)</div>
<div id="prompt"></div><div id="par"></div>
<div><button onclick="votar(0)">tie</button><button onclick="voltar()">undo</button></div>
<div class="pouco" id="prog"></div><div id="fim"></div>
<script>
const P = __PARES__, N = __NOMES__;
let i = 0, votos = [];
function mostrar(){
  if(i >= P.length) return final();
  const p = P[i];
  document.getElementById('prompt').textContent = '"' + p.p + '"';
  document.getElementById('par').innerHTML = `<div class="img" onclick="votar(1)"><img src="${p.e}"></div><div class="img" onclick="votar(2)"><img src="${p.d}"></div>`;
  document.getElementById('prog').textContent = (i + 1) + ' of ' + P.length;
}
function votar(v){
  if(i >= P.length) return;
  const p = P[i];
  // v 1 esquerda, 2 direita, 0 empate. Traduz para qual experimento venceu.
  let venc = 'tie';
  if(v) { const esqA = !p.t; venc = (v === 1) === esqA ? 'a' : 'b'; }
  votos.push({categoria: p.c, prompt: p.p, semente: p.s, vencedor: venc});
  i++; mostrar();
}
function voltar(){ if(i > 0){ i--; votos.pop(); document.getElementById('fim').innerHTML=''; mostrar(); } }
function final(){
  document.getElementById('prompt').textContent = 'Done';
  document.getElementById('par').innerHTML = '';
  const cats = {};
  for(const v of votos){ cats[v.categoria] ??= {a:0,b:0,tie:0}; cats[v.categoria][v.vencedor]++; }
  const tot = {a:0,b:0,tie:0}; for(const c in cats) for(const k in tot) tot[k] += cats[c][k];
  let h = `<table><tr><th>category</th><th>${N.a}</th><th>${N.b}</th><th>tie</th></tr>`;
  for(const c in cats) h += `<tr><td>${c}</td><td>${cats[c].a}</td><td>${cats[c].b}</td><td>${cats[c].tie}</td></tr>`;
  h += `<tr><th>total</th><th>${tot.a}</th><th>${tot.b}</th><th>${tot.tie}</th></tr></table>`;
  const n = tot.a + tot.b; if(n) h += `<div>${N.a} wins ${(100*tot.a/n).toFixed(1)}% of the decided pairs</div>`;
  h += '<button onclick="baixar()">download votes</button>';
  document.getElementById('fim').innerHTML = h;
}
function baixar(){
  const b = new Blob([JSON.stringify({a: N.a, b: N.b, votos}, null, 1)], {type:'application/json'});
  const l = document.createElement('a'); l.href = URL.createObjectURL(b); l.download = `votes_${N.a}_${N.b}.json`; l.click();
}
document.addEventListener('keydown', e => { if(e.key==='1') votar(1); if(e.key==='2') votar(2); if(e.key==='0') votar(0); });
mostrar();
</script></body></html>"""
html = html.replace("__PARES__", json.dumps(pares)).replace("__NOMES__", json.dumps(nomes))
pathlib.Path(a.saida).write_text(html, encoding="utf-8")
print(f"{len(pares)} pares em {a.saida}")
