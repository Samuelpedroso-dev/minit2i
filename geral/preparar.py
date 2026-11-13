"""Prepara o dataset. Baixa os shards do CC3M um por vez, recorta cada foto em 128 por 128,
passa pelo VAE e salva os latentes e os tokens da legenda. Cada shard vira um arquivo .pt,
entao da para parar e retomar quando quiser, ele pula o que ja foi feito.
    python preparar.py --saida /content/drive/MyDrive/minit2i/dados --inicio 0 --fim 200
"""
import argparse, io, os, pathlib, tarfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from congelados import Texto, VAE

ap = argparse.ArgumentParser()
ap.add_argument("--saida", required=True)
ap.add_argument("--inicio", type=int, default=0)
ap.add_argument("--fim", type=int, default=576)
ap.add_argument("--res", type=int, default=128)
ap.add_argument("--minutos", type=float, default=1e9)
ap.add_argument("--lote", type=int, default=32)
ap.add_argument("--tmp", default="/tmp/cc3m")
ap.add_argument("--tar_local", default=None, help="usa um tar local em vez de baixar, para teste")
ap.add_argument("--falso", action="store_true")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saida = pathlib.Path(a.saida); saida.mkdir(parents=True, exist_ok=True)
texto, vae = Texto(dev, a.falso), VAE(dev, a.falso)


def abrir(par):
    jpg, cap = par
    try:
        im = Image.open(io.BytesIO(jpg)).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if min(w, h) < a.res * 0.75 or max(w, h) / min(w, h) > 2.5:
        return None
    s = a.res / min(w, h)
    im = im.resize((max(a.res, round(w * s)), max(a.res, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
    w, h = im.size; l, t = (w - a.res) // 2, (h - a.res) // 2
    return np.asarray(im.crop((l, t, l + a.res, t + a.res))), cap


def ler_tar(caminho):
    grupos = {}
    with tarfile.open(caminho) as tf:
        for m in tf:
            if not m.isfile():
                continue
            chave, _, ext = m.name.rpartition(".")
            grupos.setdefault(chave, {})[ext] = tf.extractfile(m).read()
    for g in grupos.values():
        jpg = g.get("jpg") or g.get("jpeg") or g.get("png") or g.get("webp")
        cap = g.get("txt")
        if jpg and cap:
            yield jpg, cap.decode("utf-8", "ignore").strip()


def processar(caminho, destino):
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        itens = [r for r in ex.map(abrir, ler_tar(caminho), chunksize=32) if r is not None]
    lats = []
    for i in range(0, len(itens), a.lote):
        x = torch.from_numpy(np.stack([im for im, _ in itens[i:i + a.lote]])).permute(0, 3, 1, 2).float() / 127.5 - 1
        lats.append(vae.codificar(x).cpu().half())
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    caps = [c for _, c in itens]
    dados = {"lat": torch.cat(lats), "ids": texto.tokenizar(caps), "cap": caps, "res": a.res}
    tmp = destino.with_suffix(".tmp")
    torch.save(dados, tmp); os.replace(tmp, destino)
    return len(caps), time.time() - t0


inicio = time.time()
nomes = [a.tar_local] if a.tar_local else [f"cc3m-train-{i:04d}.tar" for i in range(a.inicio, min(a.fim, 576))]
total = 0
for nome in nomes:
    destino = saida / (pathlib.Path(nome).stem + ".pt")
    if destino.exists():
        continue
    if time.time() - inicio > a.minutos * 60:
        print("tempo esgotado, rode de novo para continuar"); break
    if a.tar_local:
        caminho = a.tar_local
    else:
        from huggingface_hub import hf_hub_download
        caminho = hf_hub_download("pixparse/cc3m-wds", nome, repo_type="dataset", local_dir=a.tmp)
    n, dt = processar(caminho, destino)
    total += n
    if not a.tar_local:
        os.remove(caminho)
    feitos = len(list(saida.glob("*.pt")))
    print(f"{nome} {n} imagens em {dt:.0f}s, {feitos} shards prontos, {total} imagens nesta sessão", flush=True)
