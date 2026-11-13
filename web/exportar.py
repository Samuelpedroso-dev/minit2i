"""Exporta o checkpoint para o navegador. Pesos em float16 codificados em base64,
configuracao e vocabulario em JSON, mais um caso de referencia calculado pelo PyTorch
para o JavaScript conferir. Uso, a partir da raiz do repositorio
    python3 web/exportar.py treino/ckpt.pt
"""
import base64, json, pathlib, sys
import numpy as np
import torch

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "treino"))
from dit import MiniT2I, Tokenizer  # noqa
import formas  # noqa

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else RAIZ / "treino" / "ckpt.pt")
cfg = dict(img=32, ch=3, patch=4, dim=128, depth=6, heads=4, txt_depth=2, maxlen=12)
model = MiniT2I(len(ck["vocab"])); model.load_state_dict(ck["ema"]); model.eval()

tensores, partes, ofs = [], [], 0
for nome, t in model.state_dict().items():
    a = t.detach().float().numpy().astype(np.float16).ravel()
    tensores.append({"nome": nome, "forma": list(t.shape), "ofs": ofs, "n": a.size})
    partes.append(a); ofs += a.size
pesos = np.concatenate(partes)
(RAIZ / "pesos.b64.txt").write_text(base64.b64encode(pesos.tobytes()).decode())

# caso de referencia, entrada fixa, saida do PyTorch em float32
tokz = Tokenizer([]); tokz.itos = ck["vocab"]; tokz.stoi = {w: i for i, w in enumerate(tokz.itos)}
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 3, 32, 32, generator=g)
prompt = "um triângulo azul grande no centro"
tok = tokz.encode([prompt])
with torch.no_grad():
    v = model(x, torch.tensor([0.5]), tok)
torch.save({"ema": model.state_dict(), "vocab": ck["vocab"], "passo": ck["step"]}, RAIZ / "modelo" / "minit2i.pt")
json.dump({
    "arquitetura": "minit2i", **cfg, "vocab": ck["vocab"], "passo": ck["step"],
    "parametros": int(pesos.size), "tensores": tensores,
    "grupos": {"formas": formas.FORMAS, "cores": list(formas.CORES), "tamanhos": list(formas.TAMANHOS), "posicoes": list(formas.POSICOES)},
    "referencia": {"prompt": prompt, "tok": tok[0].tolist(), "t": 0.5,
                   "x": [round(float(f), 6) for f in x.ravel()], "v": [round(float(f), 6) for f in v.ravel()]},
}, open(RAIZ / "modelo.json", "w"), ensure_ascii=False)
print(f"{pesos.size} parametros, passo {ck['step']}")
