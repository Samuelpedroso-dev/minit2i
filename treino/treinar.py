"""Treina o MiniT2I. Retoma do checkpoint se existir, para rodar em sessoes curtas (Colab, Kaggle)."""
import argparse, copy, os, time
import numpy as np
import torch
from dit import MiniT2I, Tokenizer, rf_loss, VAZIO
import formas

ap = argparse.ArgumentParser()
ap.add_argument("--minutos", type=float, default=9)
ap.add_argument("--ckpt", default="ckpt.pt")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--n", type=int, default=20000)
a = ap.parse_args()
torch.set_num_threads(os.cpu_count())
torch.manual_seed(int(time.time()))

if not os.path.exists("dados.npz"):
    imgs, caps = formas.gerar(a.n)
    np.savez_compressed("dados.npz", imgs=imgs, caps=np.array(caps))
d = np.load("dados.npz")
X = torch.from_numpy(d["imgs"]).permute(0, 3, 1, 2).float() / 127.5 - 1
tokz = Tokenizer(formas.vocab())
T = tokz.encode(list(d["caps"]))

model = MiniT2I(len(tokz.itos))
ema = copy.deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
step = 0
if os.path.exists(a.ckpt):
    ck = torch.load(a.ckpt)
    model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    print(f"retomando do passo {step}")
print(f"parametros {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

t0, losses = time.time(), []
while time.time() - t0 < a.minutos * 60:
    idx = torch.randint(0, len(X), (a.batch,))
    x, tok = X[idx], T[idx].clone()
    drop = torch.rand(a.batch) < 0.1  # 10% sem prompt, para o classifier free guidance
    tok[drop] = 0; tok[drop, 0] = VAZIO
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1, (step + 1) / 500)
    loss = rf_loss(model, x, tok)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        decay = min(0.999, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - decay)
    losses.append(loss.item()); step += 1
    if step % 200 == 0:
        print(f"passo {step} perda {np.mean(losses[-200:]):.4f} {time.time()-t0:.0f}s", flush=True)
torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
            "vocab": tokz.itos}, a.ckpt)
print(f"salvo no passo {step}")
