"""Treina o MiniT2I geral numa GPU. Carrega os latentes prontos inteiros na GPU, retoma do
checkpoint, salva a cada 15 minutos e gera uma grade de exemplos a cada 30. Shards novos que
o preparar.py terminar durante o treino entram automaticamente.
    python treinar.py --dados /content/drive/MyDrive/minit2i/dados --ckpt /content/drive/MyDrive/minit2i/geral.pt
"""
import argparse, copy, os, pathlib, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from modelo import DiTGeral, perda_rf, amostrar
from congelados import Texto, VAE

ap = argparse.ArgumentParser()
ap.add_argument("--dados", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--amostras", default=None)
ap.add_argument("--minutos", type=float, default=170)
ap.add_argument("--lote", type=int, default=256)
ap.add_argument("--acum", type=int, default=2, help="divide cada lote em pedaços para caber na memória")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--dim", type=int, default=768)
ap.add_argument("--prof", type=int, default=12)
ap.add_argument("--cabecas", type=int, default=12)
ap.add_argument("--min_shards", type=int, default=10)
ap.add_argument("--hf_repo", default="minit2i", help="repositório público do Hugging Face com os pesos do site")
ap.add_argument("--hf_treino", default="minit2i-treino", help="repositório privado com os checkpoints")
ap.add_argument("--local", default="/content/ckpt", help="pasta local dos checkpoints quando há Hugging Face")
ap.add_argument("--falso", action="store_true")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = dev.type == "cuda"
if cuda:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dados_dir = pathlib.Path(a.dados)
amostras_dir = pathlib.Path(a.amostras or pathlib.Path(a.ckpt).parent / "amostras"); amostras_dir.mkdir(parents=True, exist_ok=True)

PROMPTS = ["um cachorro na grama", "a red car on a city street", "uma praia com coqueiros ao pôr do sol",
           "a plate of pasta", "um gato dormindo no sofá", "mountains covered in snow",
           "uma casa de madeira na floresta", "a portrait of a smiling woman"]


class Dados:
    """Latentes e tokens na GPU. Recarrega quando aparecem shards novos."""
    def __init__(self):
        self.vistos, self.lat, self.ids = set(), None, None

    def atualizar(self):
        novos = sorted(p for p in dados_dir.glob("*.pt") if p.name not in self.vistos)
        if not novos:
            return 0
        L, I = [], []
        for p in novos:
            try:
                d = torch.load(p, map_location="cpu")
            except Exception:
                continue
            L.append(d["lat"]); I.append(d["ids"]); self.vistos.add(p.name)
        if not L:
            return 0
        L, I = torch.cat(L).to(dev), torch.cat(I).to(dev)
        self.lat = L if self.lat is None else torch.cat([self.lat, L])
        self.ids = I if self.ids is None else torch.cat([self.ids, I])
        return len(L)

    def __len__(self):
        return 0 if self.lat is None else len(self.lat)


dados = Dados()
while True:
    dados.atualizar()
    if len(dados.vistos) >= a.min_shards or a.falso:
        break
    print(f"esperando shards, {len(dados.vistos)} de {a.min_shards} prontos", flush=True); time.sleep(60)
print(f"{len(dados)} imagens em {len(dados.vistos)} shards", flush=True)

texto, vae = Texto(dev, a.falso), VAE(dev, a.falso)
vazio = texto.tokenizar([""]).to(dev)
res_lat = dados.lat.shape[-1]

model = DiTGeral(lat_res=res_lat, dim=a.dim, depth=a.prof, heads=a.cabecas).to(dev)
ema = copy.deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
scaler = torch.amp.GradScaler(enabled=cuda)
passo, vistas = 0, 0
# com HF_TOKEN, os checkpoints ficam no disco local e sobem para o Hugging Face, que é rápido.
# sem ele, ficam no Drive como antes.
api = None
token = os.environ.get("HF_TOKEN", "").strip()
if token and not a.falso:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    usuario = api.whoami()["name"]
    REPO_WEB, REPO_TREINO = f"{usuario}/{a.hf_repo}", f"{usuario}/{a.hf_treino}"
    api.create_repo(REPO_WEB, exist_ok=True, private=False)
    api.create_repo(REPO_TREINO, exist_ok=True, private=True)
    print(f"Hugging Face conectado como {usuario}, site em {REPO_WEB}, checkpoints em {REPO_TREINO}", flush=True)
    os.makedirs(a.local, exist_ok=True)
    CKPTS = [os.path.join(a.local, "geral.pt"), os.path.join(a.local, "geral_b.pt")]
    if not any(os.path.exists(c) for c in CKPTS) and "geral.pt" in api.list_repo_files(REPO_TREINO):
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO_TREINO, "geral.pt", local_dir=a.local, token=token)
        print("checkpoint baixado do Hugging Face", flush=True)
    CKPTS_LEITURA = CKPTS + [a.ckpt, a.ckpt[:-3] + "_b.pt"]
else:
    print("sem HF_TOKEN, checkpoints vão para o Drive", flush=True)
    CKPTS = [a.ckpt, a.ckpt[:-3] + "_b.pt" if a.ckpt.endswith(".pt") else a.ckpt + "_b"]
    CKPTS_LEITURA = CKPTS


def melhor_ckpt():
    """Lê os dois arquivos alternados e fica com o de passo maior que abrir sem erro."""
    melhor, idx = None, -1
    for i, c in enumerate(CKPTS_LEITURA):
        if not os.path.exists(c):
            continue
        try:
            ck = torch.load(c, map_location="cpu")
        except Exception as e:
            print(f"checkpoint {c} ilegível, ignorado ({e.__class__.__name__})", flush=True); continue
        if melhor is None or ck["passo"] > melhor["passo"]:
            melhor, idx = ck, (i if i < len(CKPTS) else -1)
    return melhor, idx


ck, idx_lido = melhor_ckpt()
if ck is not None:
    model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
    if "scaler" in ck: scaler.load_state_dict(ck["scaler"])
    passo, vistas = ck["passo"], ck.get("vistas", 0)
    print(f"retomando do passo {passo}, {vistas/1e6:.2f} M imagens vistas", flush=True)
    del ck
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M parametros treináveis", flush=True)


import threading
fixos, envio = None, None
if api is not None:
    import web_export
    try:
        fixos = web_export.preparar_fixos(api, REPO_WEB, texto, vae, dev)
        print("encoder de texto e VAE do site prontos no Hugging Face", flush=True)
    except Exception as e:
        print(f"não consegui preparar o site no Hugging Face ({e}), o treino segue", flush=True)


def enviar(caminho, passo_, vistas_, ema_estado):
    """Roda em segundo plano. Sobe o checkpoint e publica o transformer do site."""
    try:
        t0 = time.time()
        api.upload_file(path_or_fileobj=caminho, path_in_repo="geral.pt", repo_id=REPO_TREINO,
                        commit_message=f"checkpoint passo {passo_}")
        print(f"checkpoint do passo {passo_} no Hugging Face em {time.time()-t0:.0f}s", flush=True)
        try:
            api.super_squash_history(REPO_TREINO)  # apaga versões antigas, só o último checkpoint ocupa espaço
        except Exception:
            pass
        if fixos is not None:
            m = DiTGeral(**model.cfg); m.load_state_dict(ema_estado); m.eval()
            dt = web_export.publicar(api, REPO_WEB, m, dev, passo_, vistas_, fixos, grade=str(amostras_dir / "ultima.png"))
            print(f"site atualizado com o passo {passo_} em {dt:.0f}s", flush=True)
            try:
                api.super_squash_history(REPO_WEB)
            except Exception:
                pass
            del m
    except Exception as e:
        print(f"falha ao enviar para o Hugging Face ({e}), tento de novo no próximo salvamento", flush=True)


def compacto(obj):
    """Tensores de ponto flutuante em bfloat16, o arquivo cai pela metade e sobe mais rápido para o Drive."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", torch.bfloat16) if obj.is_floating_point() else obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: compacto(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(compacto(v) for v in obj)
    return obj


n_salvos = idx_lido + 1  # o primeiro salvamento vai para o arquivo que NÃO foi lido


def salvar():
    """Alterna entre dois arquivos, assim um salvamento que não terminou de subir nunca apaga o anterior."""
    global n_salvos
    destino = CKPTS[n_salvos % 2]; n_salvos += 1
    tmp = destino + ".tmp"
    torch.save({"model": compacto(model.state_dict()), "ema": compacto(ema.state_dict()),
                "opt": compacto(opt.state_dict()), "scaler": scaler.state_dict(),
                "passo": passo, "vistas": vistas, "cfg": model.cfg}, tmp)
    os.replace(tmp, destino)
    print(f"checkpoint salvo em {os.path.basename(destino)} no passo {passo}", flush=True)
    global envio
    if api is not None:
        if envio is not None and envio.is_alive():
            print("envio anterior ainda em andamento, este fica só no disco local", flush=True)
        else:
            estado_ema = {k: v.detach().to("cpu", copy=True).float() for k, v in ema.state_dict().items()}
            envio = threading.Thread(target=enviar, args=(destino, passo, vistas, estado_ema), daemon=False)
            envio.start()


def grade():
    ema.eval()
    cond = texto(texto.tokenizar(PROMPTS).to(dev))
    inc = texto(vazio.expand(len(PROMPTS), -1))
    z = amostrar(ema, cond, inc, passos=25, cfg=4.5, semente=0)
    img = ((vae.decodificar(z) + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    from PIL import Image, ImageDraw
    n, S = len(PROMPTS), img.shape[1]
    tela = Image.new("RGB", (4 * S, 2 * (S + 18)), (0, 0, 0)); d = ImageDraw.Draw(tela)
    for i in range(n):
        x, y = (i % 4) * S, (i // 4) * (S + 18)
        tela.paste(Image.fromarray(img[i]), (x, y)); d.text((x + 3, y + S + 3), PROMPTS[i][:22], fill=(220, 220, 220))
    caminho = amostras_dir / f"passo_{passo:07d}.png"
    tela.save(caminho); tela.save(amostras_dir / "ultima.png")
    return caminho


t0 = ult_salvo = ult_grade = ult_dados = time.time()
perdas, t_log, v_log = [], time.time(), vistas
while time.time() - t0 < a.minutos * 60:
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1.0, (passo + 1) / 2000)
    opt.zero_grad(set_to_none=True)
    mb = a.lote // a.acum
    soma = 0.0
    for _ in range(a.acum):
        idx = torch.randint(0, len(dados), (mb,), device=dev)
        x0 = dados.lat[idx].float()
        x0 = torch.where(torch.rand(mb, 1, 1, 1, device=dev) < 0.5, x0.flip(3), x0)
        ids = dados.ids[idx]
        drop = torch.rand(mb, device=dev) < 0.1
        ids = torch.where(drop[:, None], vazio.expand(mb, -1), ids)
        txt, tmask, pooled = texto(ids)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
            loss = perda_rf(model, x0, txt, tmask, pooled) / a.acum
        scaler.scale(loss).backward()
        soma += loss.item()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    with torch.no_grad():
        dec = min(0.9999, (1 + passo) / (10 + passo))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - dec)
    passo += 1; vistas += a.lote; perdas.append(soma)
    if passo % 100 == 0:
        dt = time.time() - t_log
        print(f"passo {passo} perda {np.mean(perdas[-100:]):.4f} {(vistas - v_log)/dt:.0f} img/s "
              f"{vistas/1e6:.2f} M vistas {len(dados)/1e3:.0f} mil imagens", flush=True)
        t_log, v_log = time.time(), vistas
    agora = time.time()
    if agora - ult_salvo > 1800:
        salvar(); ult_salvo = agora
    if agora - ult_grade > 1800:
        print("grade salva em", grade(), flush=True); ult_grade = agora
    if agora - ult_dados > 1200:
        n = dados.atualizar(); ult_dados = agora
        if n: print(f"+{n} imagens novas, total {len(dados)}", flush=True)
salvar()
print("grade salva em", grade(), flush=True)
if envio is not None:
    envio.join()
print(f"salvo no passo {passo}", flush=True)
