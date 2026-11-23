"""Treina o MiniT2I geral numa GPU. Carrega os latentes prontos inteiros na GPU, retoma do
checkpoint, salva a cada 15 minutos e gera uma grade de exemplos a cada 30. Shards novos que
o preparar.py terminar durante o treino entram automaticamente.
    python treinar.py --dados /content/drive/MyDrive/minit2i/dados --ckpt /content/drive/MyDrive/minit2i/geral.pt
"""
import argparse, copy, json, os, pathlib, time
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
# opções da 0.3. Sem elas o treino é exatamente o da 0.2.
ap.add_argument("--legendas", default=None, help="pasta com os .jsonl do relegendar.py")
ap.add_argument("--p_nova", type=float, default=0.8, help="chance de usar a legenda nova quando ela existe")
ap.add_argument("--filtro", default=None, help="ex. marca_dagua<0.3,texto<0.4,nitidez>40")
ap.add_argument("--restringir", default=None, help="usa só shards que têm .jsonl nesta pasta, para todos os experimentos verem as mesmas imagens")
ap.add_argument("--maxlen", type=int, default=None, help="tokens de texto, as legendas novas são mais longas")
ap.add_argument("--tempos", default="logit", help="logit, uniforme, desloc:3 ou misto:0.2")
ap.add_argument("--decair_inicio", type=int, default=None, help="passo em que o learning rate começa a cair")
ap.add_argument("--decair_fim", type=int, default=None, help="passo em que ele chega em --lr_min")
ap.add_argument("--lr_min", type=float, default=0.0)
ap.add_argument("--iniciar_de", default=None, help="checkpoint base quando este experimento ainda não tem o seu")
ap.add_argument("--passos_exp", type=int, default=None, help="para depois de N passos contados do início do experimento")
ap.add_argument("--decair_frac", type=float, default=None, help="com --passos_exp, decai o lr nessa fração final")
ap.add_argument("--sem_site", action="store_true", help="não publica no site nem no painel, para experimentos")
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


def ler_filtro(txt):
    regras = []
    for r in (txt or "").split(","):
        r = r.strip()
        if not r:
            continue
        op = "<" if "<" in r else ">"
        nome, v = r.split(op)
        regras.append((nome.strip(), op, float(v)))
    return regras


REGRAS = ler_filtro(a.filtro)


def passa(linha):
    for nome, op, v in REGRAS:
        x = linha["nitidez"] if nome == "nitidez" else linha["clip"][nome]
        if (x >= v) if op == "<" else (x <= v):
            return False
    return True


class Dados:
    """Latentes e tokens na GPU. Recarrega quando aparecem shards novos. Com --legendas, guarda
    também os tokens da legenda nova e quais linhas têm uma. Com --filtro, sorteia só as linhas
    que passam."""
    def __init__(self):
        self.vistos, self.lat, self.ids, self.cap = set(), None, None, []
        self.ids_nova, self.tem_nova, self.validos = None, None, None
        self.tok = None  # o tokenizador entra depois, só quando precisa retokenizar
        self.desalinhados, self.truncadas, self.n_novas = 0, 0, 0

    def _legendas(self, p, n):
        if not a.legendas:
            return None
        j = pathlib.Path(a.legendas) / (p.stem + ".jsonl")
        if not j.exists():
            return None
        linhas = [json.loads(l) for l in open(j, encoding="utf-8")]
        if len(linhas) != n:
            self.desalinhados += 1
            return None
        return linhas

    def atualizar(self):
        novos = sorted(p for p in dados_dir.glob("*.pt") if p.name not in self.vistos)
        if not novos:
            return 0
        L, I, IN, TN, V = [], [], [], [], []
        for p in novos:
            if a.restringir and not (pathlib.Path(a.restringir) / (p.stem + ".jsonl")).exists():
                continue
            try:
                d = torch.load(p, map_location="cpu")
            except Exception:
                continue
            n = len(d["lat"]); caps = list(d.get("cap", [""] * n))
            linhas = self._legendas(p, n)
            if linhas is not None and any(l["orig"] != c for l, c in zip(linhas, caps)):
                self.desalinhados += 1; linhas = None
            ids = d["ids"]
            if self.tok is not None and ids.shape[1] != self.tok.maxlen:
                ids = self.tok.tokenizar(caps)
            L.append(d["lat"]); I.append(ids); self.cap += caps; self.vistos.add(p.name)
            if self.tok is not None:
                if linhas is not None:
                    nov = [l["nova"] for l in linhas]
                    IN.append(self.tok.tokenizar(nov)); TN.append(torch.ones(n, dtype=torch.bool))
                    self.n_novas += n
                    if not self.tok.falso:
                        comp = self.tok.tok(nov, truncation=False).input_ids
                        self.truncadas += sum(len(c) > self.tok.maxlen for c in comp)
                else:
                    IN.append(ids); TN.append(torch.zeros(n, dtype=torch.bool))
            # sem .jsonl para o shard não há notas, então ele entra inteiro
            V.append(torch.tensor([passa(l) for l in linhas], dtype=torch.bool) if (REGRAS and linhas is not None)
                     else torch.ones(n, dtype=torch.bool))
        if not L:
            return 0
        base = len(self)
        L, I, V = torch.cat(L).to(dev), torch.cat(I).to(dev), torch.cat(V)
        self.lat = L if self.lat is None else torch.cat([self.lat, L])
        self.ids = I if self.ids is None else torch.cat([self.ids, I])
        if IN:
            IN, TN = torch.cat(IN).to(dev), torch.cat(TN).to(dev)
            self.ids_nova = IN if self.ids_nova is None else torch.cat([self.ids_nova, IN])
            self.tem_nova = TN if self.tem_nova is None else torch.cat([self.tem_nova, TN])
        v = (torch.nonzero(V).flatten() + base).to(dev)
        self.validos = v if self.validos is None else torch.cat([self.validos, v])
        return len(L)

    def sortear(self, mb):
        return self.validos[torch.randint(0, len(self.validos), (mb,), device=dev)]

    def __len__(self):
        return 0 if self.lat is None else len(self.lat)


dados = Dados()
if a.legendas or a.maxlen:
    from congelados import MAXLEN
    dados.tok = Texto(torch.device("cpu"), a.falso, a.maxlen or MAXLEN)
    if not a.falso:
        dados.tok.m = None  # só o tokenizador é usado aqui
while True:
    dados.atualizar()
    if len(dados.vistos) >= a.min_shards or a.falso:
        break
    print(f"esperando shards, {len(dados.vistos)} de {a.min_shards} prontos", flush=True); time.sleep(60)
print(f"{len(dados)} imagens em {len(dados.vistos)} shards", flush=True)
if a.legendas:
    print(f"legendas novas em {dados.n_novas} imagens, {dados.truncadas} cortadas pelo limite de tokens, "
          f"{dados.desalinhados} shards desalinhados ignorados", flush=True)
if REGRAS:
    print(f"filtro {a.filtro} deixa {len(dados.validos)} de {len(dados)} imagens", flush=True)

from congelados import MAXLEN
texto, vae = Texto(dev, a.falso, a.maxlen or MAXLEN), VAE(dev, a.falso)
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
exp_inicio = None
if ck is not None:
    exp_inicio = ck.get("exp_inicio")
elif a.iniciar_de:
    base = a.iniciar_de
    if base == "hf":  # o checkpoint principal da 0.2, no repositório privado de sempre
        from huggingface_hub import hf_hub_download
        base = hf_hub_download(f"{usuario}/minit2i-treino", "geral.pt", local_dir="/content/base", token=token)
    ck = torch.load(base, map_location="cpu")
    exp_inicio = ck["passo"]
    print(f"experimento novo, começando do checkpoint base no passo {exp_inicio}", flush=True)
if ck is not None:
    model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
    for p_, st in opt.state.items():  # o checkpoint compacto guarda o otimizador em bfloat16, volta ao tipo dos pesos
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.float() if k == "step" else v.to(p_.device, p_.dtype)
    if "scaler" in ck: scaler.load_state_dict(ck["scaler"])
    passo, vistas = ck["passo"], ck.get("vistas", 0)
    print(f"retomando do passo {passo}, {vistas/1e6:.2f} M imagens vistas", flush=True)
    del ck
if exp_inicio is None:
    exp_inicio = passo
if a.passos_exp:
    print(f"experimento do passo {exp_inicio} ao {exp_inicio + a.passos_exp}, tempos {a.tempos}", flush=True)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M parametros treináveis", flush=True)


import threading
fixos, envio = None, None
if api is not None and not a.sem_site:
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
        return {k: (v if k == "step" else compacto(v)) for k, v in obj.items()}
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
                "passo": passo, "vistas": vistas, "cfg": model.cfg, "exp_inicio": exp_inicio,
                "opcoes": {k: v for k, v in vars(a).items() if k not in ("dados", "ckpt", "local")}}, tmp)
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
    cond = texto(texto.tokenizar(PROMPTS).to(dev))  # vazio já tem o mesmo comprimento, veio do mesmo texto
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


painel = None
if api is not None and not a.sem_site:
    from ao_vivo import AoVivo
    painel = AoVivo(api, REPO_WEB)
    print("painel ao vivo ligado", flush=True)
img_s_atual = 0.0
t0 = ult_salvo = ult_grade = ult_dados = time.time()
perdas, t_log, v_log = [], time.time(), vistas
def taxa(p):
    """Aquecimento de 2000 passos, depois constante. Com decaimento, cai em linha reta até lr_min."""
    lr = a.lr * min(1.0, (p + 1) / 2000)
    ini, fim = a.decair_inicio, a.decair_fim
    if a.passos_exp and a.decair_frac:
        fim = exp_inicio + a.passos_exp
        ini = fim - int(a.passos_exp * a.decair_frac)
    if ini is not None and fim is not None and p >= ini:
        f = min(1.0, (p - ini) / max(1, fim - ini))
        lr = lr + (a.lr_min - lr) * f
    return lr


def acabou():
    return a.passos_exp is not None and passo - exp_inicio >= a.passos_exp


while time.time() - t0 < a.minutos * 60 and not acabou():
    for g in opt.param_groups:
        g["lr"] = taxa(passo)
    opt.zero_grad(set_to_none=True)
    mb = a.lote // a.acum
    soma = 0.0
    for _ in range(a.acum):
        idx = dados.sortear(mb)
        x0 = dados.lat[idx].float()
        x0 = torch.where(torch.rand(mb, 1, 1, 1, device=dev) < 0.5, x0.flip(3), x0)
        ids = dados.ids[idx]
        if dados.ids_nova is not None:
            usar = dados.tem_nova[idx] & (torch.rand(mb, device=dev) < a.p_nova)
            ids = torch.where(usar[:, None], dados.ids_nova[idx], ids)
        drop = torch.rand(mb, device=dev) < 0.1
        ids = torch.where(drop[:, None], vazio.expand(mb, -1), ids)
        txt, tmask, pooled = texto(ids)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
            loss = perda_rf(model, x0, txt, tmask, pooled, a.tempos) / a.acum
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
        img_s_atual = (vistas - v_log) / dt
        if painel is not None: painel.anotar(passo, np.mean(perdas[-100:]))
        print(f"passo {passo} perda {np.mean(perdas[-100:]):.4f} lr {taxa(passo):.2e} {(vistas - v_log)/dt:.0f} img/s "
              f"{vistas/1e6:.2f} M vistas {len(dados)/1e3:.0f} mil imagens", flush=True)
        t_log, v_log = time.time(), vistas
    if painel is not None and passo % 25 == 0:
        painel.capturar(ema, dados, texto, vae, passo, vistas, img_s_atual, np.mean(perdas[-100:]), dev)
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
