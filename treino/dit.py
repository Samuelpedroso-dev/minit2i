"""Modelo texto para imagem minimo: encoder de texto proprio + Diffusion Transformer
treinado com rectified flow. Tudo do zero, sem pesos pre treinados."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, VAZIO = 0, 1  # token de padding e token de "prompt vazio" (usado no guidance)


class Tokenizer:
    def __init__(self, vocab):
        self.itos = ["<pad>", "<vazio>"] + sorted(vocab)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, texts, maxlen=12):
        out = torch.zeros(len(texts), maxlen, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [self.stoi[w] for w in t.lower().split() if w in self.stoi][:maxlen] or [VAZIO]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim, depth=2, heads=4, maxlen=12):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, maxlen, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tok):
        mask = tok == PAD
        x = self.emb(tok) + self.pos[:, : tok.shape[1]]
        x = self.norm(self.enc(x, src_key_padding_mask=mask))
        keep = (~mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1)
        return x, mask, pooled


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * 1000 * freqs[None]
    return torch.cat([args.cos(), args.sin()], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, txt, tmask):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + sc1) + sh1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross(h, txt, txt, key_padding_mask=tmask, need_weights=False)[0]
        h = self.n3(x) * (1 + sc2) + sh2
        return x + g2 * self.mlp(h)


class MiniT2I(nn.Module):
    def __init__(self, vocab_size, img=32, ch=3, patch=4, dim=128, depth=6, heads=4, txt_depth=2):
        super().__init__()
        self.img, self.ch, self.patch = img, ch, patch
        self.text = TextEncoder(vocab_size, dim, txt_depth, heads)
        self.embed = nn.Conv2d(ch, dim, patch, patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.fn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.fada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * ch)
        for m in (self.fada[1], self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, tok):
        B = x.shape[0]
        txt, tmask, pooled = self.text(tok)
        c = self.t_mlp(timestep_embedding(t, txt.shape[-1])) + pooled
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            h = blk(h, c, txt, tmask)
        sh, sc = self.fada(c).unsqueeze(1).chunk(2, -1)
        h = self.out(self.fn(h) * (1 + sc) + sh)
        g, p = self.img // self.patch, self.patch
        h = h.view(B, g, g, p, p, self.ch).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(B, self.ch, self.img, self.img)


def rf_loss(model, x0, tok):
    """Rectified flow. x_t = (1 t) x0 + t ruido, a rede preve v = ruido menos x0."""
    B = x0.shape[0]
    t = torch.sigmoid(torch.randn(B, device=x0.device))  # logit normal, como no SD3
    noise = torch.randn_like(x0)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
    return F.mse_loss(model(xt, t, tok), noise - x0)


@torch.no_grad()
def sample(model, tok, steps=30, cfg=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = tok.shape[0]
    x = torch.randn(B, model.ch, model.img, model.img, generator=g).to(tok.device)
    empty = torch.zeros_like(tok); empty[:, 0] = VAZIO
    ts = torch.linspace(1, 0, steps + 1, device=tok.device)
    for i in range(steps):
        t = ts[i].expand(B)
        v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([tok, empty]))
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
    return x.clamp(-1, 1)
