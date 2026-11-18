"""Gera imagens com o checkpoint mais recente do MiniT2I geral. Roda em GPU ou só em CPU.
    python gerar.py --ckpt /content/drive/MyDrive/minit2i/geral.pt "um cachorro na grama" "a red car"
"""
import argparse, os, time
import torch
from PIL import Image, ImageDraw
from modelo import DiTGeral, amostrar
from congelados import Texto, VAE

ap = argparse.ArgumentParser()
ap.add_argument("prompts", nargs="+")
ap.add_argument("--ckpt", required=True, help="caminho do geral.pt, o geral_b.pt ao lado também é considerado")
ap.add_argument("--n", type=int, default=4, help="imagens por prompt")
ap.add_argument("--passos", type=int, default=25)
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--semente", type=int, default=None)
ap.add_argument("--saida", default="teste.png")
ap.add_argument("--falso", action="store_true")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()
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
model = DiTGeral(**melhor["cfg"])
model.load_state_dict({k: v.float() for k, v in melhor["ema"].items()})
model.to(dev).eval()
passo = melhor["passo"]; del melhor
texto, vae = Texto(dev, a.falso), VAE(dev, a.falso)
print(f"modelo do passo {passo} carregado em {time.time()-t0:.0f}s, rodando em {'GPU' if dev.type == 'cuda' else 'CPU'}", flush=True)

semente = a.semente if a.semente is not None else int(time.time()) % 100000
vazio = texto.tokenizar([""]).to(dev)
linhas = []
for p in a.prompts:
    t1 = time.time()
    cond = texto(texto.tokenizar([p] * a.n).to(dev))
    inc = texto(vazio.expand(a.n, -1))
    z = amostrar(model, cond, inc, passos=a.passos, cfg=a.cfg, semente=semente)
    img = ((vae.decodificar(z) + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    linhas.append((p, img))
    print(f"{p!r} pronto em {time.time()-t1:.0f}s", flush=True)

S = linhas[0][1].shape[1]; esc = 2
tela = Image.new("RGB", (a.n * S * esc, len(linhas) * (S * esc + 22)), (12, 12, 14))
d = ImageDraw.Draw(tela)
for r, (p, img) in enumerate(linhas):
    y = r * (S * esc + 22)
    d.text((6, y + 5), f"{p}   (passo {passo}, semente {semente})", fill=(230, 230, 230))
    for c in range(a.n):
        tela.paste(Image.fromarray(img[c]).resize((S * esc, S * esc), Image.LANCZOS), (c * S * esc, y + 22))
tela.save(a.saida)
print("salvo em", a.saida)
