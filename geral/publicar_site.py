"""Publica no site a versao atual do MiniT2I geral, a partir do checkpoint do Drive.
Roda em CPU. Precisa do segredo HF_TOKEN no Colab.
    python publicar_site.py --ckpt /content/drive/MyDrive/minit2i/geral.pt
"""
import argparse, os, time
import torch
from huggingface_hub import HfApi
from modelo import DiTGeral
from congelados import Texto, VAE
import web_export

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--hf_repo", default="minit2i")
a = ap.parse_args()

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit("falta o segredo HF_TOKEN no Colab (ícone de chave na barra da esquerda)")
api = HfApi(token=token)
usuario = api.whoami()["name"]
repo = f"{usuario}/{a.hf_repo}"
api.create_repo(repo, exist_ok=True, private=False)

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
    raise SystemExit("nenhum checkpoint em " + a.ckpt)
ema = DiTGeral(**melhor["cfg"])
ema.load_state_dict({k: v.float() for k, v in melhor["ema"].items()})
ema.eval()
passo, vistas = melhor["passo"], melhor.get("vistas", 0)
del melhor
dev = torch.device("cpu")
t0 = time.time()
texto, vae = Texto(dev), VAE(dev)
fixos = web_export.preparar_fixos(api, repo, texto, vae, dev, pasta="/content/web_export")
dt = web_export.publicar(api, repo, ema, dev, passo, vistas, fixos, pasta="/content/web_export")
print(f"site publicado com o passo {passo} em {time.time()-t0:.0f}s")
print(f"REPOSITORIO_DO_SITE={repo}")
