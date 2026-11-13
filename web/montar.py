"""Monta a pagina. Junta motor, trabalhador, interface e as tabelas com os numeros medidos
num unico index.html na raiz. Uso, a partir da raiz do repositorio
    python3 web/montar.py
"""
import json, pathlib, re

RAIZ = pathlib.Path(__file__).resolve().parent.parent
WEB = RAIZ / "web"
VERSAO = "0.1"


def limpa(js):
    js = re.sub(r'^\s*import .*?;\s*$', '', js, flags=re.M)
    return js.replace("export class", "class").replace("export function", "function").replace("export const", "const")


def pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def milhar(n):
    return f"{n:,}".replace(",", ".")


cfg = json.load(open(RAIZ / "modelo.json"))
med = json.load(open(RAIZ / "resultados" / "medicao.json"))
ref = cfg["referencia"]

# erro do caso de referencia, medido pelo teste em node e gravado em resultados/
conf = json.load(open(RAIZ / "resultados" / "referencia.json"))

linhas = [("Cor", "cor"), ("Tamanho", "tamanho"), ("Posição", "posicao"), ("Forma", "forma"), ("Tudo certo ao mesmo tempo", "tudo")]
tabela = ['<div class="rolagem"><table class="dados"><thead><tr><th>atributo</th>'
          f'<th>combinações vistas, {med["vistas"]["n"]} imagens</th><th>nunca vistas, {med["nunca vistas"]["n"]} imagens</th>'
          '</tr></thead><tbody>']
for nome, k in linhas:
    tabela.append(f"<tr><td>{nome}</td><td>{pct(med['vistas'][k])}</td><td>{pct(med['nunca vistas'][k])}</td></tr>")
tabela.append("</tbody></table></div>")

tamanho = (RAIZ / "pesos.b64.txt").stat().st_size / 1e6
html = (WEB / "corpo.html").read_text()
subs = {
    "TRABALHADOR": limpa((WEB / "motor.js").read_text()) + "\n" + limpa((WEB / "trabalhador.js").read_text()),
    "APP": (WEB / "app.js").read_text(),
    "TABELA": "".join(tabela),
    "VERSAO": VERSAO,
    "PARAMETROS": str(cfg["parametros"]),
    "PARAMETROS_PT": milhar(cfg["parametros"]),
    "PASSO_PT": milhar(cfg["passo"]),
    "TAMANHO": f"{tamanho:.1f} MB".replace(".", ","),
    "ERRO_REF": f"{conf['erro_max']:.1e}".replace(".", ","),
    "ESCALA_REF": f"{conf['escala']:.1f}".replace(".", ","),
    "PASSOS_MED": str(med["passos"]),
    "GUIDANCE_MED": f"{med['guidance']:g}".replace(".", ","),
    "CLASSIF_TUDO": pct(med["classificador"]["tudo"]),
}
for k, v in subs.items():
    html = html.replace("{{" + k + "}}", v)
assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
(RAIZ / "index.html").write_text(html)
print(f"index.html, {len(html) / 1e3:.0f} KB")
