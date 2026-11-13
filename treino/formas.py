"""Gera um dataset sintetico de formas coloridas com legenda em portugues."""
import random
import numpy as np
from PIL import Image, ImageDraw

FORMAS = ["círculo", "quadrado", "triângulo", "losango"]
CORES = {"vermelho": (230, 40, 40), "verde": (40, 200, 60), "azul": (50, 90, 240), "amarelo": (240, 220, 40),
         "roxo": (150, 60, 200), "branco": (245, 245, 245), "laranja": (250, 140, 20)}
TAMANHOS = {"pequeno": 0.18, "grande": 0.34}
POSICOES = {"à esquerda": (0.28, 0.5), "à direita": (0.72, 0.5), "no topo": (0.5, 0.28),
            "embaixo": (0.5, 0.72), "no centro": (0.5, 0.5)}
# combinacoes que NUNCA aparecem no treino, para testar se o modelo generaliza
FORA_DO_TREINO = {("triângulo", "azul"), ("losango", "amarelo")}


def vocab():
    words = {"um"}
    for s in list(FORMAS) + list(CORES) + list(TAMANHOS) + list(POSICOES):
        words.update(s.split())
    return words


def desenhar(forma, cor, tam, pos, S=32, rng=random):
    big = S * 4
    bg = tuple(int(v) for v in np.array([25, 25, 35]) + rng.randint(-8, 8))
    im = Image.new("RGB", (big, big), bg)
    d = ImageDraw.Draw(im)
    cx, cy = POSICOES[pos]
    cx, cy = (cx + rng.uniform(-0.05, 0.05)) * big, (cy + rng.uniform(-0.05, 0.05)) * big
    r = TAMANHOS[tam] * big * rng.uniform(0.9, 1.1) / 2
    col = tuple(int(np.clip(c + rng.randint(-15, 15), 0, 255)) for c in CORES[cor])
    if forma == "círculo":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif forma == "quadrado":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif forma == "triângulo":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    else:
        d.polygon([(cx, cy - r * 1.2), (cx + r, cy), (cx, cy + r * 1.2), (cx - r, cy)], fill=col)
    return im.resize((S, S), Image.LANCZOS)


def legenda(forma, cor, tam, pos):
    return f"um {forma} {cor} {tam} {pos}"


def gerar(n, seed=0):
    rng = random.Random(seed)
    imgs, caps = np.zeros((n, 32, 32, 3), np.uint8), []
    i = 0
    while i < n:
        f, c = rng.choice(FORMAS), rng.choice(list(CORES))
        if (f, c) in FORA_DO_TREINO:
            continue
        t, p = rng.choice(list(TAMANHOS)), rng.choice(list(POSICOES))
        imgs[i] = np.asarray(desenhar(f, c, t, p, rng=rng))
        caps.append(legenda(f, c, t, p))
        i += 1
    return imgs, caps
