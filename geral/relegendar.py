"""Legendas novas e notas de qualidade para o CC3M, para a versão 0.3.

Para cada shard do CC3M ele baixa o tar de novo, aplica o mesmo filtro de tamanho do preparar.py
(assim a linha i daqui é a linha i do shard .pt) e grava um .jsonl separado com
    i, legenda original, legenda nova, largura, altura, nitidez e notas do CLIP
As legendas originais nunca são alteradas. O treino escolhe qual usar com --legendas.

    python relegendar.py --saida /content/drive/MyDrive/minit2i/legendas --inicio 0 --fim 200 \
        --dados /content/drive/MyDrive/minit2i/dados

Com HF_TOKEN no ambiente e --hf_repo, cada shard pronto também sobe para um dataset privado,
o que permite rodar isto numa conta do Colab e treinar em outra.
"""
import argparse, io, json, os, pathlib, tarfile, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--saida", required=True)
ap.add_argument("--inicio", type=int, default=0)
ap.add_argument("--fim", type=int, default=200)
ap.add_argument("--dados", default=None, help="pasta dos shards .pt, usada para conferir o alinhamento das linhas")
ap.add_argument("--res", type=int, default=128, help="a mesma resolução usada no preparar.py")
ap.add_argument("--modelo", default="microsoft/Florence-2-base")
ap.add_argument("--tarefa", default="<DETAILED_CAPTION>")
ap.add_argument("--feixes", type=int, default=1, help="beam search, 3 é um pouco melhor e bem mais lento")
ap.add_argument("--lote", type=int, default=32)
ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
ap.add_argument("--minutos", type=float, default=1e9)
ap.add_argument("--hf_repo", default=None, help="ex. minit2i-legendas, dataset privado no Hugging Face")
ap.add_argument("--tmp", default="/tmp/cc3m")
ap.add_argument("--tar_local", default=None)
ap.add_argument("--falso", action="store_true", help="sem modelos, só para testar o fluxo")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if dev.type == "cuda" else torch.float32
saida = pathlib.Path(a.saida); saida.mkdir(parents=True, exist_ok=True)

# rótulos para a triagem por zero shot. Cada imagem recebe a probabilidade de cada um.
ROTULOS = {
    "limpa": "a clean high quality photo",
    "marca_dagua": "a photo with a watermark or logo stamped over it",
    "texto": "an image with large text, a poster, a screenshot or a document",
    "borrada": "a blurry low quality photo",
    "ilustracao": "a drawing, clipart or illustration",
}


def ok_tamanho(w, h):
    """Mesmo critério do preparar.py. Se mudar lá, tem que mudar aqui."""
    return not (min(w, h) < a.res * 0.75 or max(w, h) / min(w, h) > 2.5)


def ler_tar(caminho):
    grupos = {}
    with tarfile.open(caminho) as tf:
        for m in tf:
            if not m.isfile():
                continue
            chave, _, ext = m.name.rpartition(".")
            grupos.setdefault(chave, {})[ext] = tf.extractfile(m).read()
    for chave, g in grupos.items():
        jpg = g.get("jpg") or g.get("jpeg") or g.get("png") or g.get("webp")
        cap = g.get("txt")
        if jpg and cap:
            yield chave, jpg, cap.decode("utf-8", "ignore").strip()


def itens_do_shard(caminho):
    """Lista na mesma ordem e com o mesmo filtro do preparar.py."""
    out = []
    for chave, jpg, cap in ler_tar(caminho):
        try:
            im = Image.open(io.BytesIO(jpg)).convert("RGB")
        except Exception:
            continue
        if not ok_tamanho(*im.size):
            continue
        tam = im.size
        s = 384 / min(im.size)  # o Florence usa 768, 384 já basta e economiza memória
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        out.append((chave, im, cap, tam))
    return out


def nitidez(im):
    """Variância do laplaciano na escala de cinza. Valores baixos indicam foto borrada."""
    g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


class Legendador:
    def __init__(self):
        if a.falso:
            return
        from unittest.mock import patch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        def sem_flash(arquivo):  # o código remoto do Florence pede flash_attn, que não existe na T4
            return [i for i in get_imports(arquivo) if i != "flash_attn"]
        with patch("transformers.dynamic_module_utils.get_imports", sem_flash):
            self.m = AutoModelForCausalLM.from_pretrained(a.modelo, trust_remote_code=True, torch_dtype=dtype,
                                                          attn_implementation="eager").to(dev).eval()
            self.proc = AutoProcessor.from_pretrained(a.modelo, trust_remote_code=True)

    @torch.no_grad()
    def __call__(self, imgs):
        if a.falso:
            return [f"a test caption for an image of {im.size[0]} by {im.size[1]} pixels" for im in imgs]
        x = self.proc(text=[a.tarefa] * len(imgs), images=imgs, return_tensors="pt")
        ger = self.m.generate(input_ids=x["input_ids"].to(dev), pixel_values=x["pixel_values"].to(dev, dtype),
                              max_new_tokens=96, num_beams=a.feixes, do_sample=False)
        txt = self.proc.batch_decode(ger, skip_special_tokens=True)
        return [" ".join(t.replace("<pad>", "").split()) for t in txt]


class Juiz:
    """CLIP para notas de qualidade por zero shot."""
    def __init__(self):
        self.nomes = list(ROTULOS)
        if a.falso:
            return
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained(a.clip, torch_dtype=dtype).to(dev).eval()
        self.proc = CLIPProcessor.from_pretrained(a.clip)
        t = self.proc(text=[ROTULOS[n] for n in self.nomes], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            e = self.m.get_text_features(**t)
        self.txt = torch.nn.functional.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def __call__(self, imgs):
        if a.falso:
            return [{n: round(1 / len(self.nomes), 4) for n in self.nomes} for _ in imgs]
        x = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, dtype)
        e = torch.nn.functional.normalize(self.m.get_image_features(pixel_values=x).float(), dim=-1)
        p = (100 * e @ self.txt.T).softmax(-1).cpu().numpy()
        return [{n: round(float(v), 4) for n, v in zip(self.nomes, linha)} for linha in p]


def conferir(destino_pt, linhas):
    """Confere se a linha i daqui tem a mesma legenda original da linha i do shard .pt."""
    if not destino_pt or not destino_pt.exists():
        return None
    try:
        caps = torch.load(destino_pt, map_location="cpu").get("cap")
    except Exception:
        return None
    if caps is None:
        return None
    return len(caps) == len(linhas) and all(c == l["orig"] for c, l in zip(caps, linhas))


leg, juiz = Legendador(), Juiz()
api = None
if a.hf_repo and os.environ.get("HF_TOKEN") and not a.falso:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"].strip())
    REPO = f"{api.whoami()['name']}/{a.hf_repo}"
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    print("enviando para o dataset", REPO, flush=True)

inicio = time.time()
nomes = [a.tar_local] if a.tar_local else [f"cc3m-train-{i:04d}.tar" for i in range(a.inicio, min(a.fim, 576))]
for nome in nomes:
    stem = pathlib.Path(nome).stem
    destino = saida / f"{stem}.jsonl"
    if destino.exists():
        continue
    if time.time() - inicio > a.minutos * 60:
        print("tempo esgotado, rode de novo para continuar"); break
    t0 = time.time()
    if a.tar_local:
        caminho = a.tar_local
    else:
        from huggingface_hub import hf_hub_download
        caminho = hf_hub_download("pixparse/cc3m-wds", nome, repo_type="dataset", local_dir=a.tmp)
    itens = itens_do_shard(caminho)
    linhas = []
    for j in range(0, len(itens), a.lote):
        bloco = itens[j:j + a.lote]
        imgs = [it[1] for it in bloco]
        novas, notas = leg(imgs), juiz(imgs)
        for k, ((chave, im, cap, tam), nova, nota) in enumerate(zip(bloco, novas, notas)):
            linhas.append({"i": j + k, "chave": chave, "orig": cap, "nova": nova,
                           "w": tam[0], "h": tam[1], "nitidez": round(nitidez(im), 1), "clip": nota})
    ok = conferir(pathlib.Path(a.dados) / f"{stem}.pt" if a.dados else None, linhas)
    if ok is False:
        print(f"AVISO {stem} não bate com o shard .pt, as linhas estão desalinhadas", flush=True)
    tmp = destino.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    os.replace(tmp, destino)
    if not a.tar_local:
        os.remove(caminho)
    if api is not None:
        try:
            api.upload_file(path_or_fileobj=str(destino), path_in_repo=destino.name, repo_id=REPO, repo_type="dataset")
        except Exception as e:
            print(f"falha ao enviar {destino.name} ({e}), fica só no disco", flush=True)
    dt = time.time() - t0
    feitos = len(list(saida.glob("*.jsonl")))
    print(f"{stem} {len(linhas)} legendas em {dt:.0f}s ({len(linhas)/max(dt,1e-6):.1f}/s), "
          f"{feitos} shards prontos, alinhado {ok}", flush=True)
