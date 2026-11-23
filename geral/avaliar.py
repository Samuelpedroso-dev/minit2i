"""Avaliação fixa de um checkpoint, para comparar experimentos da 0.3 entre si.

Gera sempre os mesmos prompts (prompts_avaliacao.json, separados por categoria) com as mesmas
sementes, salva cada imagem e calcula
    CLIP score por categoria, com o CLIP ViT L/14, que é outro modelo e não o encoder usado no treino
    FID opcional contra fotos reais do conjunto de validação do CC3M, que o treino nunca viu
O cego.py usa as pastas geradas aqui para a avaliação humana às cegas.

    python avaliar.py --ckpt /content/ckpt/geral.pt --saida /content/drive/MyDrive/minit2i/avaliacao/e0
    python avaliar.py --ckpt /content/ckpt/geral.pt --saida ... --fid 5000

O FID só é comparável entre execuções com o mesmo --fid, a mesma resolução e as mesmas fotos de
referência. Com poucas amostras ele fica enviesado para cima, então 5000 é o mínimo razoável.
"""
import argparse, io, json, os, pathlib, tarfile, time
import numpy as np
import torch
from PIL import Image
from modelo import DiTGeral
from congelados import Texto, VAE, MAXLEN

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="arquivo .pt, o _b.pt ao lado também é considerado")
ap.add_argument("--saida", required=True)
ap.add_argument("--prompts", default=str(pathlib.Path(__file__).with_name("prompts_avaliacao.json")))
ap.add_argument("--sementes", type=int, default=4, help="imagens por prompt, sementes 0 até N menos 1")
ap.add_argument("--passos", type=int, default=25)
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--desloc", type=float, default=1.0, help="deslocamento dos tempos na geração")
ap.add_argument("--maxlen", type=int, default=None)
ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
ap.add_argument("--fid", type=int, default=0, help="número de imagens para o FID, 0 desliga")
ap.add_argument("--lote", type=int, default=64)
ap.add_argument("--falso", action="store_true")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saida = pathlib.Path(a.saida); saida.mkdir(parents=True, exist_ok=True)

melhor = None
for c in (a.ckpt, a.ckpt[:-3] + "_b.pt"):
    if os.path.exists(c):
        try:
            ck = torch.load(c, map_location="cpu")
        except Exception:
            continue
        if melhor is None or ck["passo"] > melhor["passo"]:
            melhor = ck
if melhor is None:
    raise SystemExit("nenhum checkpoint encontrado em " + a.ckpt)
model = DiTGeral(**melhor["cfg"]); model.load_state_dict({k: v.float() for k, v in melhor["ema"].items()})
model.to(dev).eval()
passo, opcoes = melhor["passo"], melhor.get("opcoes", {})
maxlen = a.maxlen or opcoes.get("maxlen") or MAXLEN
del melhor
texto, vae = Texto(dev, a.falso, maxlen), VAE(dev, a.falso)
vazio = texto.tokenizar([""]).to(dev)
print(f"checkpoint do passo {passo}, {maxlen} tokens de texto", flush=True)


def gerar(prompts, sementes):
    """Uma imagem por par (prompt, semente). Cada semente vira um lote próprio, assim a mesma
    semente dá o mesmo ruído inicial em qualquer checkpoint."""
    out = []
    for i in range(0, len(prompts), a.lote):
        ps, ss = prompts[i:i + a.lote], sementes[i:i + a.lote]
        cond = texto(texto.tokenizar(ps).to(dev)); inc = texto(vazio.expand(len(ps), -1))
        zs = []
        for j, s in enumerate(ss):  # ruído por imagem, independente do tamanho do lote
            g = torch.Generator().manual_seed(int(s))
            zs.append(torch.randn(model.ch, model.res, model.res, generator=g))
        z = amostrar_com_ruido(cond, inc, torch.stack(zs).to(dev))
        img = ((vae.decodificar(z) + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        out += list(img)
    return out


@torch.no_grad()
def amostrar_com_ruido(cond, inc, x):
    """O mesmo amostrador do modelo.py, mas partindo de um ruído dado."""
    txt, mask, pooled = cond; B = txt.shape[0]
    ts = torch.linspace(1, 0, a.passos + 1, device=dev)
    if a.desloc != 1.0:
        ts = a.desloc * ts / (1 + (a.desloc - 1) * ts)
    T2, M2, P2 = torch.cat([txt, inc[0]]), torch.cat([mask, inc[1]]), torch.cat([pooled, inc[2]])
    for i in range(a.passos):
        t = ts[i].expand(2 * B)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            v = model(torch.cat([x, x]), t, T2, M2, P2).float()
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + a.cfg * (vc - vu))
    return x


class Clip:
    def __init__(self):
        if a.falso:
            return
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained(a.clip).to(dev).eval()
        self.p = CLIPProcessor.from_pretrained(a.clip)

    @torch.no_grad()
    def __call__(self, imgs, textos):
        if a.falso:
            return [0.0] * len(imgs)
        r = []
        for i in range(0, len(imgs), 32):
            x = self.p(text=textos[i:i + 32], images=[Image.fromarray(m) for m in imgs[i:i + 32]],
                       return_tensors="pt", padding=True, truncation=True).to(dev)
            e_i = torch.nn.functional.normalize(self.m.get_image_features(pixel_values=x["pixel_values"]), dim=-1)
            e_t = torch.nn.functional.normalize(self.m.get_text_features(input_ids=x["input_ids"],
                                                                        attention_mask=x["attention_mask"]), dim=-1)
            r += (100 * (e_i * e_t).sum(-1)).clamp(min=0).tolist()
        return r


# 1. prompts fixos
t0 = time.time()
cats = json.load(open(a.prompts, encoding="utf-8"))
itens = [(c, k, p, s) for c, ps in cats.items() for k, p in enumerate(ps) for s in range(a.sementes)]
imgs = gerar([p for _, _, p, _ in itens], [s for *_, s in itens])
clip = Clip()
notas = clip(imgs, [p for _, _, p, _ in itens])
indice = []
for (c, k, p, s), im, n in zip(itens, imgs, notas):
    nome = f"{c}/{k:02d}_s{s}.png"
    (saida / c).mkdir(exist_ok=True)
    Image.fromarray(im).save(saida / nome)
    indice.append({"categoria": c, "prompt": p, "semente": s, "arquivo": nome, "clip": round(n, 3)})
por_cat = {c: round(float(np.mean([i["clip"] for i in indice if i["categoria"] == c])), 3) for c in cats}
print("CLIP score por categoria", por_cat, flush=True)

# grade por categoria, uma linha por prompt
S = imgs[0].shape[0]
for c, ps in cats.items():
    tela = Image.new("RGB", (a.sementes * S, len(ps) * S))
    for i in indice:
        if i["categoria"] == c:
            k = int(i["arquivo"].split("/")[1][:2])
            tela.paste(Image.open(saida / i["arquivo"]), (i["semente"] * S, k * S))
    tela.save(saida / f"grade_{c}.png")

# 2. FID opcional
fid = None
if a.fid:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from huggingface_hub import hf_hub_download
    m = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    res = model.res * 8
    reais, caps = [], []
    k = 0
    while len(reais) < a.fid:
        tar = hf_hub_download("pixparse/cc3m-wds", f"cc3m-validation-{k:04d}.tar", repo_type="dataset", local_dir="/tmp/cc3m_val")
        grupos = {}
        with tarfile.open(tar) as tf:
            for mb in tf:
                if mb.isfile():
                    ch, _, ext = mb.name.rpartition(".")
                    grupos.setdefault(ch, {})[ext] = tf.extractfile(mb).read()
        for g in grupos.values():
            if len(reais) >= a.fid or "jpg" not in g or "txt" not in g:
                continue
            try:
                im = Image.open(io.BytesIO(g["jpg"])).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            if min(w, h) < res * 0.75 or max(w, h) / min(w, h) > 2.5:
                continue
            s = res / min(w, h)
            im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
            w, h = im.size; l, t = (w - res) // 2, (h - res) // 2
            reais.append(np.asarray(im.crop((l, t, l + res, t + res)))); caps.append(g["txt"].decode("utf-8", "ignore").strip())
        k += 1
    para_t = lambda xs: torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev)
    for i in range(0, len(reais), 128):
        m.update(para_t(reais[i:i + 128]), real=True)
    geradas = gerar(caps, list(range(10_000, 10_000 + len(caps))))
    for i in range(0, len(geradas), 128):
        m.update(para_t(geradas[i:i + 128]), real=False)
    fid = round(float(m.compute()), 2)
    print(f"FID {fid} com {len(caps)} imagens", flush=True)

json.dump({"passo": passo, "opcoes": opcoes, "amostrador": {"passos": a.passos, "cfg": a.cfg, "desloc": a.desloc},
           "clip_modelo": a.clip, "clip_por_categoria": por_cat,
           "clip_medio": round(float(np.mean(notas)), 3), "fid": fid, "fid_n": a.fid or None,
           "imagens": indice}, open(saida / "metricas.json", "w"), ensure_ascii=False, indent=1)
print(f"pronto em {time.time()-t0:.0f}s, resultados em {saida}", flush=True)
