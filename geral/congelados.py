"""Os dois modelos congelados que o MiniT2I usa e nao treina. O encoder de texto CLIP
multilingue, para entender prompts em portugues e ingles, e o VAE do Stable Diffusion,
que comprime 128 por 128 pixels em 16 por 16 por 4. Com falso=True, versoes aleatorias
minusculas para testar o pipeline sem internet."""
import torch

TEXTO_REPO = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
VAE_REPO = "stabilityai/sd-vae-ft-ema"
ESCALA_VAE = 0.18215
MAXLEN = 48


class Texto:
    def __init__(self, dev, falso=False, maxlen=MAXLEN):
        self.dev, self.falso, self.maxlen = dev, falso, maxlen
        if falso:
            g = torch.Generator().manual_seed(0)
            self.tabela = torch.randn(30000, 768, generator=g).to(dev)
            self.pad = 0
            return
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(TEXTO_REPO)
        try:
            m = AutoModel.from_pretrained(TEXTO_REPO)
        except Exception:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(TEXTO_REPO, device="cpu")[0].auto_model
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        self.m = m.to(dev, dtype).eval().requires_grad_(False)
        self.pad = self.tok.pad_token_id

    def tokenizar(self, textos):
        if self.falso:
            ids = torch.zeros(len(textos), self.maxlen, dtype=torch.int32)
            for i, t in enumerate(textos):
                w = [101] + [hash(p) % 29000 + 1000 for p in t.lower().split()][: self.maxlen - 2] + [102]
                ids[i, : len(w)] = torch.tensor(w)
            return ids
        return self.tok(list(textos), padding="max_length", truncation=True, max_length=self.maxlen,
                        return_tensors="pt").input_ids.int()

    @torch.no_grad()
    def __call__(self, ids):
        ids = ids.to(self.dev).long()
        am = ids != self.pad
        if self.falso:
            h = self.tabela[ids]
        else:
            h = self.m(input_ids=ids, attention_mask=am.long()).last_hidden_state.float()
        amf = am.float()[..., None]
        pooled = (h * amf).sum(1) / amf.sum(1).clamp(min=1)
        return h, ~am, pooled


class VAE:
    def __init__(self, dev, falso=False):
        self.dev, self.falso = dev, falso
        if falso:
            return
        from diffusers import AutoencoderKL
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        self.m = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(dev).eval().requires_grad_(False)
        self.dtype = dtype

    @torch.no_grad()
    def codificar(self, x):  # x em [-1, 1], (B, 3, H, W)
        if self.falso:
            return torch.nn.functional.avg_pool2d(x, 8).repeat(1, 2, 1, 1)[:, :4] * 0.5
        return self.m.encode(x.to(self.dev, self.dtype)).latent_dist.mean.float() * ESCALA_VAE

    @torch.no_grad()
    def decodificar(self, z):
        if self.falso:
            return torch.nn.functional.interpolate(z[:, :3] * 2, scale_factor=8).clamp(-1, 1)
        return self.m.decode((z / ESCALA_VAE).to(self.dev, self.dtype)).sample.float().clamp(-1, 1)
