"""Mede o quanto as imagens geradas obedecem ao prompt. Um classificador geometrico le
cor, tamanho, posicao e forma de cada imagem. Antes de medir o modelo, o classificador e
conferido nas imagens reais do treino, onde a resposta certa e conhecida.
Uso, dentro de treino/
    python3 medir.py ckpt.pt
"""
import itertools, json, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
import formas
from dit import MiniT2I, Tokenizer, sample

PAL = {k: np.array(v, float) for k, v in formas.CORES.items()}
ANC = {k: np.array(v) * 32 for k, v in formas.POSICOES.items()}


def mascara_ideal(forma, cx, cy, r):
    """Silhueta exata da forma, com antisserrilhado, em valores de 0 a 1."""
    S = 8
    im = Image.new("L", (32 * S, 32 * S), 0); d = ImageDraw.Draw(im)
    cx, cy, r = cx * S, cy * S, r * S
    if forma == "círculo": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif forma == "quadrado": d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif forma == "triângulo": d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=255)
    else: d.polygon([(cx, cy - r * 1.2), (cx + r, cy), (cx, cy + r * 1.2), (cx - r, cy)], fill=255)
    return np.asarray(im.resize((32, 32), Image.BOX)).astype(float) / 255


AREA = {"círculo": np.pi, "quadrado": 4, "triângulo": 2, "losango": 2.4}


def ler_forma(a, fundo, dist, m):
    """Estima a opacidade de cada pixel e compara com a silhueta ideal de cada forma,
    com a mesma area e o mesmo centro de massa, em pequenos ajustes de escala e posicao."""
    fg = a[dist >= np.percentile(dist[m], 70)].mean(0)
    alpha = np.clip(((a - fundo) @ (fg - fundo)) / max(1e-6, (fg - fundo) @ (fg - fundo)), 0, 1)
    w = alpha.sum(); ys, xs = np.mgrid[0:32, 0:32]
    cx0, cy0 = (alpha * (xs + .5)).sum() / w, (alpha * (ys + .5)).sum() / w
    melhor = (1e9, None)
    for f in formas.FORMAS:
        r0 = np.sqrt(w / AREA[f]); desl = r0 / 3 if f == "triângulo" else 0
        for dr in (0.92, 0.97, 1.0, 1.03, 1.08):
            for dx in (-.5, 0, .5):
                for dy in (-.5, 0, .5):
                    e = ((mascara_ideal(f, cx0 + dx, cy0 - desl * dr + dy, r0 * dr) - alpha) ** 2).sum()
                    if e < melhor[0]: melhor = (e, f)
    return melhor[1]


def ler(img):
    """img uint8 (32,32,3). Devolve cor, tamanho, posicao, forma, ou None se nao houver objeto."""
    a = img.astype(float)
    fundo = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)
    m = np.abs(a - fundo).max(-1) > 60
    if m.sum() < 3:
        return None
    ys, xs = np.nonzero(m)
    cor = min(PAL, key=lambda k: np.linalg.norm(a[m].mean(0) - PAL[k]))
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    pos = min(ANC, key=lambda k: np.linalg.norm(np.array([cx, cy]) - ANC[k]))
    r = (xs.max() - xs.min() + 1) / 2
    tam = "pequeno" if r < (formas.TAMANHOS["pequeno"] + formas.TAMANHOS["grande"]) * 32 / 4 else "grande"
    return {"cor": cor, "tamanho": tam, "posicao": pos, "forma": ler_forma(a, fundo, np.abs(a - fundo).max(-1), m)}


def placar(lidos, alvos):
    ac = {k: 0 for k in ("cor", "tamanho", "posicao", "forma", "tudo")}
    for l, a in zip(lidos, alvos):
        ok = {k: l is not None and l[k] == a[k] for k in ("cor", "tamanho", "posicao", "forma")}
        for k, v in ok.items(): ac[k] += v
        ac["tudo"] += all(ok.values())
    return {k: round(v / len(alvos), 4) for k, v in ac.items()} | {"n": len(alvos)}


if __name__ == "__main__":
    # 1. conferencia do classificador nas imagens reais
    imgs, caps, alvos = [], [], []
    import random
    rng = random.Random(123)
    for _ in range(2000):
        f, c, t, p = rng.choice(formas.FORMAS), rng.choice(list(formas.CORES)), rng.choice(list(formas.TAMANHOS)), rng.choice(list(formas.POSICOES))
        imgs.append(np.asarray(formas.desenhar(f, c, t, p, rng=rng))); alvos.append({"cor": c, "tamanho": t, "posicao": p, "forma": f})
    conf = placar([ler(i) for i in imgs], alvos)
    print("classificador nas imagens reais", conf)

    # 2. o modelo, todas as 280 combinacoes, 2 sementes cada
    ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "ckpt.pt")
    tokz = Tokenizer([]); tokz.itos = ck["vocab"]; tokz.stoi = {w: i for i, w in enumerate(tokz.itos)}
    model = MiniT2I(len(tokz.itos)); model.load_state_dict(ck["ema"]); model.eval()
    combos = list(itertools.product(formas.FORMAS, formas.CORES, formas.TAMANHOS, formas.POSICOES))
    res = {"vistas": ([], []), "nunca vistas": ([], [])}
    for semente in (11, 22):
        for i in range(0, len(combos), 70):
            lote = combos[i:i + 70]
            out = sample(model, tokz.encode([formas.legenda(*c) for c in lote]), steps=30, cfg=4.0, seed=semente)
            out = ((out + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
            for (f, c, t, p), im in zip(lote, out):
                grupo = "nunca vistas" if (f, c) in formas.FORA_DO_TREINO else "vistas"
                res[grupo][0].append(ler(im)); res[grupo][1].append({"cor": c, "tamanho": t, "posicao": p, "forma": f})
    final = {"passo": ck["step"], "classificador": conf, "passos": 30, "guidance": 4.0}
    for g, (l, a) in res.items():
        final[g] = placar(l, a); print(g, final[g])
    json.dump(final, open("../resultados/medicao.json", "w"), ensure_ascii=False, indent=1)
