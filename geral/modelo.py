"""MiniT2I geral. Diffusion Transformer no espaco latente do VAE do Stable Diffusion,
condicionado por um encoder de texto CLIP multilingue congelado. O transformer e treinado
do zero. Rectified flow, classifier free guidance, adaLN compartilhado como no PixArt,
normalizacao de consulta e chave para estabilidade em float16."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


class Atencao(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.qn, self.kn = RMSNorm(d // h), RMSNorm(d // h)

    def forward(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape; M = ctx.shape[1]; hd = D // self.h
        q = self.q(x).view(B, N, self.h, hd).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        # mascara aditiva no mesmo tipo da consulta, assim o grafo exportado para ONNX fica todo em float16
        am = None if mask is None else torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(mask, torch.finfo(q.dtype).min)[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(o.transpose(1, 2).reshape(B, N, D))


class Bloco(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Atencao(d, h)
        self.n2 = nn.LayerNorm(d, eps=1e-6)
        self.cross = Atencao(d, h)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.tabela = nn.Parameter(torch.randn(6, d) / d ** 0.5)
        for m in (self.attn.o, self.cross.o, self.mlp[2]):  # cada bloco comeca como identidade
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, mod, txt, tmask):
        sh1, sc1, g1, sh2, sc2, g2 = (mod + self.tabela[None]).unbind(1)
        h = self.n1(x) * (1 + sc1[:, None]) + sh1[:, None]
        x = x + g1[:, None] * self.attn(h)
        x = x + self.cross(self.n2(x), txt, tmask)
        h = self.n3(x) * (1 + sc2[:, None]) + sh2[:, None]
        return x + g2[:, None] * self.mlp(h)


def sincos_2d(d, g):
    y, x = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing="ij")
    om = 1.0 / 10000 ** (torch.arange(d // 4, dtype=torch.float32) / (d // 4))
    oy, ox = y.flatten()[:, None] * om[None], x.flatten()[:, None] * om[None]
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)[None]


def emb_tempo(t, d):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000 * f[None]
    return torch.cat([a.cos(), a.sin()], -1)


class DiTGeral(nn.Module):
    def __init__(self, lat_ch=4, lat_res=16, patch=2, dim=768, depth=12, heads=12, txt_dim=768):
        super().__init__()
        self.cfg = dict(lat_ch=lat_ch, lat_res=lat_res, patch=patch, dim=dim, depth=depth, heads=heads, txt_dim=txt_dim)
        self.ch, self.res, self.p = lat_ch, lat_res, patch
        self.embed = nn.Conv2d(lat_ch, dim, patch, patch)
        self.register_buffer("pos", sincos_2d(dim, lat_res // patch), persistent=False)
        self.txt_proj = nn.Sequential(nn.LayerNorm(txt_dim), nn.Linear(txt_dim, dim))
        self.pool_proj = nn.Sequential(nn.LayerNorm(txt_dim), nn.Linear(txt_dim, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.t_bloco = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.blocos = nn.ModuleList([Bloco(dim, heads) for _ in range(depth)])
        self.fn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.fada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * lat_ch)
        for m in (self.fada[1], self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, txt, tmask, pooled):
        B, D = x.shape[0], self.embed.out_channels
        c = self.t_mlp(emb_tempo(t, D).to(x.dtype)) + self.pool_proj(pooled)
        mod = self.t_bloco(c).view(B, 6, D)
        txt = self.txt_proj(txt)
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos.to(x.dtype)
        for b in self.blocos:
            h = b(h, mod, txt, tmask)
        sh, sc = self.fada(c).unsqueeze(1).chunk(2, -1)
        h = self.out(self.fn(h) * (1 + sc) + sh)
        g, p = self.res // self.p, self.p
        h = h.view(B, g, g, p, p, self.ch).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(B, self.ch, self.res, self.res)


def perda_rf(model, x0, txt, tmask, pooled):
    B = x0.shape[0]
    t = torch.sigmoid(torch.randn(B, device=x0.device))
    z = torch.randn_like(x0)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * z
    return F.mse_loss(model(xt, t, txt, tmask, pooled).float(), (z - x0).float())


@torch.no_grad()
def amostrar(model, cond, incond, passos=25, cfg=4.5, semente=0, shape=None):
    """cond e incond sao tuplas (txt, mask, pooled) com o mesmo tamanho de lote."""
    txt, mask, pooled = cond
    B = txt.shape[0]; dev = txt.device
    g = torch.Generator(device="cpu").manual_seed(semente)
    x = torch.randn(B, model.ch, model.res, model.res, generator=g).to(dev)
    ts = torch.linspace(1, 0, passos + 1, device=dev)
    T2 = torch.cat([txt, incond[0]]); M2 = torch.cat([mask, incond[1]]); P2 = torch.cat([pooled, incond[2]])
    for i in range(passos):
        t = ts[i].expand(2 * B)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            v = model(torch.cat([x, x]), t, T2, M2, P2).float()
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
    return x
