"""Exporta o MiniT2I geral para o navegador, em ONNX, e envia ao Hugging Face.

Tres arquivos rodam no site com o ONNX Runtime Web
    dit.onnx     o transformer treinado, em float16, roda na placa de video por WebGPU
    texto.onnx   o encoder de texto CLIP multilingue, quantizado em int8
    vae.onnx     o decodificador do VAE, em float16
mais vocab.txt para o tokenizador e meta.json com a versao.
Entradas e saidas dos tres sao sempre float32, as conversoes ficam dentro do grafo.
"""
import json, os, pathlib, time
import torch
import torch.nn as nn

from congelados import MAXLEN, ESCALA_VAE


class DiTWeb(nn.Module):
    def __init__(self, dit, dtype):
        super().__init__()
        self.m, self.dtype = dit, dtype

    def forward(self, x, t, txt, tmask, pooled):
        d = self.dtype
        return self.m(x.to(d), t.to(d), txt.to(d), tmask, pooled.to(d)).float()


class TextoWeb(nn.Module):
    """DistilBERT reescrito em torch puro com os pesos do original. O exportador do ONNX
    se confunde com as mascaras dinamicas da biblioteca transformers, esta versao exporta limpa."""
    def __init__(self, bert):
        super().__init__()
        self.m = bert
        c = bert.config
        self.h, self.d = c.n_heads, c.dim
        self.ln_eps = 1e-12

    def forward(self, ids, am):
        m, B, L = self.m, ids.shape[0], ids.shape[1]
        e = m.embeddings
        x = e.word_embeddings(ids) + e.position_embeddings.weight[:L].unsqueeze(0)
        x = e.LayerNorm(x)
        hd = self.d // self.h
        neg = (1.0 - am.float())[:, None, None, :] * -1e4
        for camada in m.transformer.layer:
            a = camada.attention
            q = a.q_lin(x).view(B, L, self.h, hd).transpose(1, 2)
            k = a.k_lin(x).view(B, L, self.h, hd).transpose(1, 2)
            v = a.v_lin(x).view(B, L, self.h, hd).transpose(1, 2)
            w = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5 + neg, -1)
            o = a.out_lin((w @ v).transpose(1, 2).reshape(B, L, self.d))
            x = camada.sa_layer_norm(o + x)
            f = camada.ffn
            y = f.lin2(torch.nn.functional.gelu(f.lin1(x)))
            x = camada.output_layer_norm(y + x)
        h = x.float()
        amf = am.float().unsqueeze(-1)
        return h, (h * amf).sum(1) / amf.sum(1).clamp(min=1)


class VAEWeb(nn.Module):
    def __init__(self, vae, dtype):
        super().__init__()
        self.m, self.dtype = vae, dtype

    def forward(self, z):
        img = self.m.decode((z / ESCALA_VAE).to(self.dtype)).sample.float().clamp(-1, 1)
        return img


def _exportar(mod, args, caminho, entradas, saidas, dinamicos):
    torch.onnx.export(mod, args, str(caminho), input_names=entradas, output_names=saidas,
                      dynamic_axes=dinamicos, opset_version=17, do_constant_folding=True, dynamo=False)


def exportar_dit(ema, saida, dev):
    dtype = torch.float16  # float16 tambem exporta na CPU, o site sempre roda por WebGPU
    import copy
    m = DiTWeb(copy.deepcopy(ema).to(dev, dtype).eval(), dtype)
    c = ema.cfg; B = 2
    args = (torch.randn(B, c["lat_ch"], c["lat_res"], c["lat_res"], device=dev), torch.rand(B, device=dev),
            torch.randn(B, MAXLEN, c["txt_dim"], device=dev), torch.zeros(B, MAXLEN, dtype=torch.bool, device=dev),
            torch.randn(B, c["txt_dim"], device=dev))
    args[3][:, 5:] = True
    b = {0: "lote"}
    with torch.no_grad():
        _exportar(m, args, saida / "dit.onnx", ["x", "t", "txt", "tmask", "pooled"], ["v"],
                  {"x": b, "t": b, "txt": b, "tmask": b, "pooled": b, "v": b})
    del m


def meta_tokenizador(tok):
    return {"pad": tok.pad_token_id, "cls": tok.cls_token_id, "sep": tok.sep_token_id, "unk": tok.unk_token_id,
            "minusculas": bool(getattr(tok, "do_lower_case", False))}


def exportar_texto(texto, saida):
    """texto e o congelados.Texto. Exporta em float32 na CPU e quantiza para int8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    dev_orig = next(texto.m.parameters()).device
    bert = texto.m.float().cpu().eval()
    m = TextoWeb(bert)
    ids = torch.ones(1, MAXLEN, dtype=torch.long); am = torch.ones(1, MAXLEN, dtype=torch.long); am[:, 6:] = 0
    tmp = saida / "texto_fp32.onnx"
    with torch.no_grad():
        _exportar(m, (ids, am), tmp, ["ids", "am"], ["h", "pooled"], {"ids": {0: "lote"}, "am": {0: "lote"}})
    quantize_dynamic(str(tmp), str(saida / "texto.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"])
    for f in saida.glob("texto_fp32.onnx*"):
        f.unlink()
    if dev_orig.type == "cuda":
        texto.m.to(dev_orig, torch.float16)
    tok = texto.tok
    vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    (saida / "vocab.txt").write_text("\n".join(w for w, _ in vocab), encoding="utf-8")


def exportar_vae(vae, saida, dev):
    dtype = torch.float16
    import copy
    m = VAEWeb(copy.deepcopy(vae.m).to(dev, dtype).eval(), dtype)
    z = torch.randn(1, 4, 16, 16, device=dev)
    with torch.no_grad():
        _exportar(m, (z,), saida / "vae.onnx", ["z"], ["img"], {"z": {0: "lote", 2: "h", 3: "w"}, "img": {0: "lote", 2: "H", 3: "W"}})


def preparar_fixos(api, repo, texto, vae, dev, pasta="/content/web_export"):
    """Texto e VAE nao mudam com o treino. Exporta e envia uma vez so, antes do treino comecar."""
    saida = pathlib.Path(pasta); saida.mkdir(parents=True, exist_ok=True)
    existentes = set(api.list_repo_files(repo))
    novos = []
    if "web/texto.onnx" not in existentes or "web/vocab.txt" not in existentes:
        exportar_texto(texto, saida); novos += ["texto.onnx", "vocab.txt"]
    if "web/vae.onnx" not in existentes:
        exportar_vae(vae, saida, dev); novos.append("vae.onnx")
    if novos:
        from huggingface_hub import CommitOperationAdd
        api.create_commit(repo, [CommitOperationAdd(path_in_repo=f"web/{f}", path_or_fileobj=str(saida / f)) for f in novos],
                          commit_message="encoder de texto e VAE do site")
    tamanhos = {i.path.split("/")[-1]: i.size for i in api.get_paths_info(repo, ["web/texto.onnx", "web/vae.onnx"])}
    return {"tokenizador": meta_tokenizador(texto.tok), "tamanhos": tamanhos}


def publicar(api, repo, ema, dev, passo, vistas, fixos, pasta="/content/web_export", grade=None):
    """Exporta o transformer do passo atual e envia para repo/web junto com meta.json."""
    saida = pathlib.Path(pasta); saida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    exportar_dit(ema, saida, dev)
    tamanhos = dict(fixos["tamanhos"]); tamanhos["dit.onnx"] = (saida / "dit.onnx").stat().st_size
    meta = {"passo": passo, "vistas": vistas, "cfg": ema.cfg, "maxlen": MAXLEN, "data": time.strftime("%Y-%m-%d %H:%M"),
            "dtype": "fp16", "tokenizador": fixos["tokenizador"],
            "tokenizador_versao": "1", "versao": f"{passo}", "tamanhos": tamanhos}
    (saida / "meta.json").write_text(json.dumps(meta))
    arquivos = ["dit.onnx", "meta.json"]
    if grade is not None and os.path.exists(grade):
        import shutil; shutil.copy(grade, saida / "grade.png"); arquivos.append("grade.png")
    from huggingface_hub import CommitOperationAdd
    api.create_commit(repo, [CommitOperationAdd(path_in_repo=f"web/{f}", path_or_fileobj=str(saida / f)) for f in arquivos],
                      commit_message=f"site no passo {passo}")
    return time.time() - t0
