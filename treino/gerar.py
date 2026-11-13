"""Gera imagens a partir de prompts. Uso: python gerar.py "um círculo vermelho grande à esquerda" ...
Sem argumentos, gera uma grade de teste com legendas."""
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from dit import MiniT2I, Tokenizer, sample

ck = torch.load("ckpt.pt")
tokz = Tokenizer([]); tokz.itos = ck["vocab"]; tokz.stoi = {w: i for i, w in enumerate(tokz.itos)}
model = MiniT2I(len(tokz.itos)); model.load_state_dict(ck["ema"]); model.eval()

prompts = sys.argv[1:] or [
    "um círculo vermelho grande à esquerda", "um quadrado verde pequeno no topo",
    "um losango roxo grande no centro", "um círculo laranja pequeno embaixo",
    "um quadrado branco grande à direita", "um triângulo amarelo grande no centro",
    "um triângulo azul grande no centro", "um losango amarelo pequeno à direita",  # nunca vistos no treino
]
N = 4  # amostras por prompt
tok = tokz.encode([p for p in prompts for _ in range(N)])
imgs = sample(model, tok, steps=30, cfg=4.0, seed=1)
imgs = ((imgs + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()

S, LW = 96, 330
grid = Image.new("RGB", (LW + N * (S + 6), len(prompts) * (S + 6) + 6), (250, 250, 250))
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default(size=15)
for r, p in enumerate(prompts):
    y = 6 + r * (S + 6)
    novo = r >= 6 and not sys.argv[1:]
    draw.text((8, y + S // 2 - 18), p, fill=(20, 20, 20), font=font)
    if novo:
        draw.text((8, y + S // 2 + 2), "(combinação nunca vista no treino)", fill=(200, 60, 30), font=font)
    for c in range(N):
        im = Image.fromarray(imgs[r * N + c]).resize((S, S), Image.NEAREST)
        grid.paste(im, (LW + c * (S + 6), y))
grid.save("resultado.png")
print(f"passo {ck['step']}, salvo em resultado.png")
