"""Painel ao vivo do treino. A cada poucos minutos pega fotos reais do lote, mistura ruído,
pede para a rede reconstruir e mede o que cada bloco do transformer está fazendo. Tudo vai
para a pasta ao_vivo do repositório público do Hugging Face, que o painel do site lê."""
import json, pathlib, threading, time
import torch

PASTA = pathlib.Path("/content/ao_vivo")
META_VISTAS = 15e6


class AoVivo:
    def __init__(self, api, repo, n=6, intervalo=120):
        self.api, self.repo, self.n, self.intervalo = api, repo, n, intervalo
        self.historico, self.envio, self.ult = [], None, 0.0
        self.inicio_sessao = time.time()
        PASTA.mkdir(parents=True, exist_ok=True)
        try:  # continua o gráfico de perda das sessões anteriores
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo, "ao_vivo/estado.json", token=api.token)
            self.historico = json.loads(pathlib.Path(p).read_text()).get("historico", [])
        except Exception:
            pass

    def anotar(self, passo, perda):
        if not self.historico or passo - self.historico[-1][0] >= 50:
            self.historico.append([int(passo), round(float(perda), 5)])
            self.historico = self.historico[-1500:]

    @torch.no_grad()
    def capturar(self, ema, dados, texto, vae, passo, vistas, img_s, perda, dev):
        if self.api is None or time.time() - self.ult < self.intervalo:
            return
        if self.envio is not None and self.envio.is_alive():
            return
        self.ult = time.time()
        try:
            self._capturar(ema, dados, texto, vae, passo, vistas, img_s, perda, dev)
        except Exception as e:
            print(f"painel ao vivo falhou ({e.__class__.__name__}: {e}), o treino segue", flush=True)

    def _capturar(self, ema, dados, texto, vae, passo, vistas, img_s, perda, dev):
        from PIL import Image
        n = self.n
        idx = torch.randint(0, len(dados), (n,), device=dev)
        x0 = dados.lat[idx].float()
        txt, tmask, pooled = texto(dados.ids[idx])
        t = torch.linspace(0.35, 0.9, n, device=dev)[torch.randperm(n, device=dev)]
        z = torch.randn_like(x0)
        tt = t[:, None, None, None]
        xt = (1 - tt) * x0 + tt * z
        camadas = []
        ganchos = [b.register_forward_hook(lambda m, i, o: camadas.append(o.float().norm(dim=-1).cpu()))
                   for b in ema.blocos]
        ema.eval()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            v = ema(xt, t, txt, tmask, pooled).float()
        for g in ganchos:
            g.remove()
        x0_prev = xt - tt * v
        perdas = ((v - (z - x0)) ** 2).mean(dim=(1, 2, 3)).cpu().tolist()
        lado = int(camadas[0].shape[1] ** 0.5)

        def jpg(lat, nome):
            im = ((vae.decodificar(lat) + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            for i in range(n):
                Image.fromarray(im[i]).save(PASTA / f"{nome}{i}.jpg", quality=82)

        jpg(x0, "real"); jpg(xt, "ruido"); jpg(x0_prev, "previsto")
        legendas = [dados.cap[i] if i < len(dados.cap) else "" for i in idx.cpu().tolist()]
        amostras = []
        for i in range(n):
            amostras.append({
                "legenda": legendas[i], "t": round(float(t[i]), 3), "perda": round(perdas[i], 4),
                "real": f"real{i}.jpg", "ruido": f"ruido{i}.jpg", "previsto": f"previsto{i}.jpg",
                "camadas": [[round(float(x), 2) for x in c[i].tolist()] for c in camadas],
            })
        estado = {
            "atualizado": time.time(), "passo": int(passo), "vistas": int(vistas), "meta": META_VISTAS,
            "img_s": round(float(img_s), 1), "perda": round(float(perda), 5), "imagens_no_dataset": len(dados),
            "sessao_inicio": self.inicio_sessao, "grade_lado": lado, "historico": self.historico, "amostras": amostras,
        }
        (PASTA / "estado.json").write_text(json.dumps(estado, ensure_ascii=False))
        self.envio = threading.Thread(target=self._enviar, args=(passo,), daemon=True)
        self.envio.start()

    def _enviar(self, passo):
        try:
            self.api.upload_folder(folder_path=str(PASTA), path_in_repo="ao_vivo", repo_id=self.repo,
                                   commit_message=f"painel ao vivo, passo {passo}")
        except Exception as e:
            print(f"painel ao vivo não subiu ({e.__class__.__name__}), tento de novo depois", flush=True)
