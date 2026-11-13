// Confere o motor JavaScript contra o caso de referencia do PyTorch e grava o erro.
// Uso, a partir da raiz do repositorio
//     node web/conferir.mjs
import fs from "fs";
import { decodificarF16, MiniT2I } from "./motor.js";

const raiz = new URL("..", import.meta.url).pathname;
const cfg = JSON.parse(fs.readFileSync(raiz + "modelo.json"));
const m = new MiniT2I(cfg, decodificarF16(fs.readFileSync(raiz + "pesos.b64.txt", "utf8")));
const r = cfg.referencia;
const { tok } = m.tokenizar(r.prompt);
if (JSON.stringify(Array.from(tok)) !== JSON.stringify(r.tok)) throw new Error("tokenizacao diferente do Python");
const t0 = performance.now();
const v = m.forward(Float32Array.from(r.x), r.t, m.codificarTexto(tok));
const ms = performance.now() - t0;
let erro = 0, escala = 0;
for (let i = 0; i < v.length; i++) { erro = Math.max(erro, Math.abs(v[i] - r.v[i])); escala = Math.max(escala, Math.abs(r.v[i])); }
fs.writeFileSync(raiz + "resultados/referencia.json", JSON.stringify({ erro_max: erro, escala, ms_forward: Math.round(ms) }, null, 1));
console.log(`erro maximo ${erro.toExponential(2)}, escala ${escala.toFixed(2)}, ${ms.toFixed(0)} ms por passo`);
