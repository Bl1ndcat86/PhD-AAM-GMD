"""
GMD-AAM: Governance by Mechanism Design — con pensamiento visible
=================================================================
Basado en tests confirmados:
  - /api/generate + "\n<think>" funciona para ambos agentes
  - Auditor: genera </think> + JSON   (Caso A)
  - Governor: razona sin </think>, termina en "NIVEL: L#" (Caso B)
  - CSV: 1500 chars de think para trazabilidad completa
"""

import os
import re
import json
import time
import requests
import webbrowser
import subprocess
import pandas as pd
from collections import Counter

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO     = "deepseek-r1:14b"
TRIALS     = 3
LIMIT      = 3
DATA_PATH  = "data/PS_20174392719_1491204439457_log.csv"

OPT_AUDITOR  = {"temperature": 0.3, "num_predict": 2000, "num_ctx": 4096}
OPT_GOVERNOR = {"temperature": 0.4, "num_predict": 2000, "num_ctx": 4096}

# Lambda: coeficiente de aversion al riesgo (tesis)
# 1.0 = tolerancia normal | 50.0 = aversion extrema (banca)
LAMBDA_COEF = 5.0


# =====================================================================
# OLLAMA — /api/generate con \n<think>
# =====================================================================
def ollama_call(prompt: str, options: dict, role: str = "", prefill: str = "", mode: str = "governor") -> tuple[str, str]:
    raw = ""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   MODELO,
                "prompt":  prompt.rstrip() + "\n<think>\n" + prefill if prefill else prompt.rstrip() + "\n<think>",
                "stream":  True,
                "options": options,
            },
            timeout=300,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                raw  += chunk.get("response", "")
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
    except requests.exceptions.Timeout:
        print(f"  [{role}] Timeout")
    except Exception as e:
        print(f"  [{role}] Error: {e}")

    return _parse_think(raw, mode=mode)


def _parse_think(raw: str, mode: str = "governor") -> tuple[str, str]:
    raw = raw.strip()

    if "<think>" in raw and "</think>" in raw:
        m = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
        if m:
            return m.group(2).strip(), m.group(1).strip()

    if "</think>" in raw and "<think>" not in raw:
        parts = raw.split("</think>", 1)
        clean = parts[1].strip() if len(parts) > 1 else ""
        if clean:
            return clean, parts[0].strip()

    if mode == "auditor":
        for pat in [r"```json", r"```\s*\{", r'\{"riesgo"', r'\{"inconsistencias"']:
            m = re.search(pat, raw, re.DOTALL)
            if m and m.start() > 20:
                return raw[m.start():].strip(), raw[:m.start()].strip()
        if raw.strip().startswith(("{", "```")):
            return raw.strip(), ""
        return raw, raw

    else:  # governor
        for pat in [
            r"NIVEL\s*:\s*L[0-4]",
            r"\*\*\s*Conclusi[oó]n",
            r"Conclusi[oó]n\s+[Ff]inal",
            r"Por\s+(?:lo\s+)?tanto.*nivel",
            r"(?:asigno|determino|elijo)\s+(?:el\s+)?nivel",
        ]:
            matches = list(re.finditer(pat, raw, re.IGNORECASE | re.DOTALL))
            if matches:
                pos = matches[-1].start()
                if pos > 30:
                    return raw[pos:].strip(), raw[:pos].strip()

        nivel_m = list(re.finditer(r"\bL[0-4]\b", raw.upper()))
        if nivel_m:
            last  = nivel_m[-1]
            start = raw.rfind(chr(10), 0, last.start())
            if start > 30:
                return raw[start:].strip(), raw[:start].strip()

        return raw, raw



# =====================================================================
# VALIDACIÓN MATEMÁTICA
# =====================================================================
def precheck(amount: float, old_bal: float, new_bal: float) -> dict:
    issues = []
    expected = round(old_bal - amount, 2)
    actual   = round(new_bal, 2)
    if abs(expected - actual) > 0.01:
        issues.append(
            f"Balance esperado ${expected} != real ${actual} "
            f"(diff: ${abs(expected - actual):.2f})"
        )
    if amount > old_bal:
        issues.append(
            f"Monto ${amount} excede balance ${old_bal} "
            f"(exceso: ${amount - old_bal:.2f})"
        )
    if new_bal == 0.0 and abs(amount - old_bal) > 0.01:
        issues.append(
            f"Cuenta vaciada a $0 pero monto ${amount} != balance ${old_bal}"
        )
    return {"issues": issues, "clean": len(issues) == 0}


# =====================================================================
# AGENTE 1: AUDITOR
# =====================================================================
def agent_auditor(amount, old_bal, new_bal, issues) -> tuple[dict, str, str]:
    expected = round(old_bal - amount, 2)
    ratio    = round(amount / old_bal * 100, 1) if old_bal > 0 else 0

    issue_block = ""
    if issues:
        issue_block = (
            "\n\nALERTA MATEMATICA DETECTADA POR PRECHECK:\n"
            + "\n".join("- " + i for i in issues)
            + "\nEstas inconsistencias deben aparecer en tu analisis."
        )

    prompt = (
        "Eres el Agente Auditor del sistema GMD-AAM (Governance as Mechanism Design).\n"
        "IMPORTANTE: Razona y responde SIEMPRE en español.\n\n"
        "Tu mision es el Sensing Effort: analizar la transaccion paso a paso\n"
        "para estimar Pe(t,p) — Probabilidad de Error del mecanismo AAM.\n\n"
        "TRANSACCION A AUDITAR:\n"
        "  Monto transaccionado:  $" + str(amount) + "\n"
        "  Balance inicial V(t):  $" + str(old_bal) + "\n"
        "  Balance final real:    $" + str(new_bal) + "\n"
        "  Balance esperado:      $" + str(expected) + "\n"
        "  Ratio monto/balance:   " + str(ratio) + "%"
        + issue_block + "\n\n"
        "ANALIZA estos tres criterios uno por uno:\n"
        "  1. CONSISTENCIA MATEMATICA:\n"
        "     balance_esperado = $" + str(expected) + "\n"
        "     balance_final    = $" + str(new_bal) + "\n"
        "     Diferencia: $" + str(round(abs(expected - new_bal), 2)) + "\n"
        "     Conclusion: " + ("HAY INCONSISTENCIA — balances no cuadran." if abs(expected - new_bal) > 0.01 else "OK — balances matematicamente correctos.") + "\n\n"
        "  2. VIABILIDAD FINANCIERA:\n"
        "     Monto $" + str(amount) + " vs Balance disponible $" + str(old_bal) + "\n"
        "     Conclusion: " + ("HAY PROBLEMA — monto supera el balance disponible." if amount > old_bal else "OK — monto dentro del balance disponible.") + "\n\n"
        "  3. VACIADO ANOMALO:\n"
        "     Balance final = $" + str(new_bal) + ("  <- CERO" if new_bal == 0.0 else "") + "\n"
        "     Conclusion: " + ("HAY VACIADO ANOMALO — balance cero pero monto no coincide con balance inicial." if new_bal == 0.0 and abs(amount - old_bal) > 0.01 else "OK — sin vaciado anomalo.") + "\n\n"
        "Explica que encontraste en cada criterio y por que es o no una inconsistencia.\n"
        "Al final entrega el JSON separando tus notas generales de los riesgos operativos reales. Si todo esta correcto, deja Pe y Ce VACIOS []:\n"
        '{"riesgo": {"notas": ["tu razonamiento general y notas cualitativas aqui"], "Pe": ["solo si hay error matematico"], "Ce": ["solo si hay un riesgo operacional real"]}}'
    )

    clean, think = ollama_call(prompt, OPT_AUDITOR, role="Auditor",
                               prefill="Bien, inicio el Sensing Effort en español. Analizo criterio por criterio:\n\n1. CONSISTENCIA MATEMATICA:",
                               mode="auditor")

    result = {"riesgo": {"notas": [], "Pe": issues[:] if issues else [], "Ce": []}}
    for pattern in [
        r'```json\s*(.*?)\s*```',
        r'(\{[^{}]*"riesgo"[^{}]*\})',
        r'(\{.*\})',
    ]:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                riesgo = parsed.get("riesgo", {})
                
                # Extraemos las tres variables de la respuesta de Gemma
                notas = riesgo.get("notas", [])
                pe = riesgo.get("inconsistencias") or riesgo.get("Pe") or []
                ce = riesgo.get("observaciones")   or riesgo.get("Ce") or []
                
                ruido = ["verificar", "documentacion", "autorizar", "politicas",
                         "practicas", "documentada", "autorizada", "asegurar"]
                ce = [c for c in ce if not any(r in c.lower() for r in ruido)]
                
                result = {"riesgo": {"notas": notas, "Pe": pe, "Ce": ce}}
                break
            except json.JSONDecodeError:
                continue

    return result, prompt, think


# =====================================================================
# AGENTE 2: GOVERNOR
# =====================================================================
def agent_governor(audit: dict, amount, old_bal) -> tuple[str, str, str]:
    audit_str  = json.dumps(audit, ensure_ascii=False)
    pe_list    = audit.get("riesgo", {}).get("Pe", [])
    ce_list    = audit.get("riesgo", {}).get("Ce", [])
    pe_count   = len(pe_list)
    ce_count   = len(ce_list)
    
    pe_detail  = "\n".join("  - " + p for p in pe_list) if pe_list else "  (ninguna — balances matematicamente correctos)"
    ce_detail  = "\n".join("  - " + c for c in ce_list) if ce_list else "  (ninguna)"
    ratio      = round(amount / old_bal * 100, 1) if old_bal > 0 else 0
    
    V_task     = max(round(amount * 0.005, 2), 1.0)
    Ce_est     = amount  
    
    if pe_count == 0:
        Pe_base = 0.0
    elif pe_count == 1:
        Pe_base = 0.35
    else:
        Pe_base = min(0.35 * pe_count, 0.95)
        
    pareto_matrix = {
        "L0": round(V_task - 100.0 - (Pe_base * 0.00 * Ce_est * LAMBDA_COEF), 2),
        "L1": round(V_task -  50.0 - (Pe_base * 0.05 * Ce_est * LAMBDA_COEF), 2),
        "L2": round(V_task -  15.0 - (Pe_base * 0.30 * Ce_est * LAMBDA_COEF), 2),
        "L3": round(V_task -   5.0 - (Pe_base * 0.80 * Ce_est * LAMBDA_COEF), 2),
        "L4": round(V_task -   0.0 - (Pe_base * 1.00 * Ce_est * LAMBDA_COEF), 2),
    }
    matrix_str = json.dumps(pareto_matrix, indent=2)
    V = V_task  

    prompt = (
        "Eres el Risk Governor del sistema GMD-AAM (Governance as Mechanism Design).\n"
        "IMPORTANTE: Responde SIEMPRE en español. Todo tu razonamiento debe ser en español.\n\n"
        "Tu funcion es implementar el Autonomy Allocation Mechanism (AAM):\n"
        "Aplicar la funcion de utilidad U(t,p) = V(t) - Cc(p) - (Pe(t,p) x Ce(t) x lambda)\n"
        "para determinar que nivel de soberania L0-L4 maximiza la utilidad neta.\n\n"
        "REPORTE DEL AGENTE AUDITOR:\n" + audit_str + "\n\n"
        "PARAMETROS ECONOMICOS DE LA TRANSACCION:\n"
        "MATRIZ DE PAGOS U(t,p) CALCULADA (elige el valor MAS ALTO):\n"
        + matrix_str + "\n\n"
        "PARAMETROS ECONOMICOS:\n"
        "  V(t) Beneficio de procesar la transaccion: $" + str(V) + "\n"
        "  Ce(t) Monto en riesgo si falla:            $" + str(amount) + "\n"
        "  Pe(t,p) Probabilidad de error:             " + str(Pe_base) + "\n"
        "  Lambda aversion al riesgo:                 " + str(LAMBDA_COEF) + "\n"
        "  Ratio monto/balance:                       " + str(ratio) + "%\n\n"
        "INCONSISTENCIAS MATEMATICAS DETECTADAS (Pe=" + str(pe_count) + "):\n"
        + pe_detail + "\n\n"
        "OBSERVACIONES OPERACIONALES (" + str(ce_count) + "):\n"
        + ce_detail + "\n\n"
        "ESCALA DE AUTONOMIA GMD-AAM — mide AUTONOMIA DEL AGENTE, no nivel de riesgo:\n"
        "  L0 — Shadow Mode (Aumentacion Consultiva):\n"
        "       Agente: solo lectura, cero privilegios de ejecucion.\n"
        "       Humano: Ejecutor Unico, valida y acciona todo manualmente.\n"
        "       Economia: Cc(p) MAXIMO. U(t,p) negativa por costo de control total.\n"
        "       USAR: cuando Pe(t,p) es tan alto que cualquier accion destruye valor.\n\n"
        "  L1 — Human-in-the-Loop (Autonomia Tutelada):\n"
        "       Agente: planifica y orquesta, pero NO ejecuta sin firma humana.\n"
        "       Humano: Validador Atomico, provee token de aprobacion por accion.\n"
        "       Economia: Cc(p) = $50 por transaccion. Alta latencia operacional.\n"
        "       USAR: Pe=0 pero hay observaciones que elevan Ce(t). Riesgo moderado.\n\n"
        "  L2 — Sandbox Execution (Autonomia Confinada):\n"
        "       Agente: ejecuta acciones reversibles en entorno controlado.\n"
        "       Humano: Supervisor Tecnico, configura los limites del sandbox.\n"
        "       Economia: Riesgo mitigado. Cc(p) reducido. U(t,p) positiva.\n"
        "       USAR: Pe=0, observaciones=0, ratio < 50%. Transaccion estandar.\n\n"
        "  L3 — Human-on-the-Loop (Gestion por Excepcion):\n"
        "       Agente: opera en real-time, ejecuta hasta hallar incertidumbre.\n"
        "       Humano: Interventor Reactivo, actua solo si la IA reporta duda.\n"
        "       Economia: Alta eficiencia. Cc(p) casi cero. U(t,p) maxima en casos limpios.\n"
        "       USAR: Pe=0, observaciones=0, historial limpio, alta confianza contextual.\n\n"
        "  L4 — High Agency (Autonomia Auditada):\n"
        "       Agente: privilegios irrestrictos, operacion continua y autonoma.\n"
        "       Humano: Auditor Post-Mortem. Revisa resultados a posteriori.\n"
        "       Economia: Velocidad maxima. Unit Economics optimizado. Sin costo de control.\n"
        "       USAR: Pe=0, transaccion limpia, alta confianza. El agente opera sin supervision.\n"
        "       NUNCA usar si hay inconsistencias matematicas — el riesgo destruye el valor.\n\n"
        "REGLAS DE ASIGNACION BASADAS EN U(t,p):\n"
        "  Pe=0, obs=0, ratio < 20%  -> L3 o L4 (U positiva, delegar al agente)\n"
        "  Pe=0, obs=0, ratio > 50%  -> L2 (monto alto, sandbox por precaucion)\n"
        "  Pe=0, obs > 0             -> L1 o L2 (observaciones elevan Ce(t))\n"
        "  Pe > 0 inconsistencia leve -> L1 (firma humana requerida)\n"
        "  Pe > 0 inconsistencia grave -> L0 (Pe x Ce destruye valor, humano toma control total)\n"
        "  NUNCA L4 si Pe > 0 — autonomia irrestricta con riesgo confirmado es inaceptable.\n\n"
        "RAZONA en español: evalua Pe(t,p), Ce(t), ratio y U(t,p).\n"
        "Explica por que el nivel elegido maximiza U(t,p) vs los otros niveles.\n"
        "Termina con:\n"
        "NIVEL: L#"
    )

    clean, think = ollama_call(prompt, OPT_GOVERNOR, role="Governor",
                               prefill="Voy a aplicar el Principio de Compatibilidad de Incentivos evaluando la Matriz de Pagos en español:")

    nivel = None
    for text in [clean, think]:
        matches = re.findall(r"\bL([0-4])\b", text.upper())
        if matches:
            nivel = "L" + matches[-1]
            break

    if nivel is None:
        if pe_count >= 2 or (pe_count >= 1 and amount > old_bal):
            nivel = "L0"
        elif pe_count == 1:
            nivel = "L1"
        elif ce_count > 0:
            nivel = "L2"
        else:
            nivel = "L4"

    return nivel, prompt, think


# =====================================================================
# TRIAL
# =====================================================================
def run_trial(amount, old_bal, new_bal, pc, t) -> dict:
    print(f"  [Trial {t+1}/{TRIALS}] Auditando...", end=" ", flush=True)
    audit, audit_prompt, audit_think = agent_auditor(amount, old_bal, new_bal, pc["issues"])
    pe = audit.get("riesgo", {}).get("Pe", [])
    ce = audit.get("riesgo", {}).get("Ce", [])
    print(f"Pe={len(pe)} Ce={len(ce)} think={len(audit_think)}c | Gobernando...", end=" ", flush=True)
    level, gov_prompt, gov_think = agent_governor(audit, amount, old_bal)
    print(f"-> {level} (gov_think={len(gov_think)}c)")
    return {
        "trial":        t + 1,
        "vote":         level,
        "audit_json":   audit,
        "audit_think":  audit_think,
        "audit_prompt": audit_prompt,
        "gov_think":    gov_think,
        "gov_prompt":   gov_prompt,
    }


# =====================================================================
# CONSENSO + DEFINICIONES DE NIVEL
# =====================================================================
NIVEL_INFO = {
    "L0": {
        "label": "L0 — Shadow Mode",
        "desc":  "Aumentacion Consultiva",
        "agent": "Nulos. Solo lectura. Actua como oraculo/copiloto.",
        "human": "Ejecutor Unico. Valida y acciona manualmente.",
        "econ":  "Cc(p) MAXIMO. Riesgo sistemico cero. U(t,p) negativa por costo de control total.",
        "when":  "Pe(t,p) tan alto que cualquier accion destruye valor. Maxima incertidumbre epistemica.",
    },
    "L1": {
        "label": "L1 — Human-in-the-Loop",
        "desc":  "Autonomia Tutelada",
        "agent": "Orquestacion. Planifica pero no ejecuta sin firma.",
        "human": "Validador Atomico. Provee el token de aprobacion.",
        "econ":  "Cc(p) = $50/tx. Alta Latencia. Inversion ineludible en seguridad para impactos severos.",
        "when":  "Pe=0 pero observaciones elevan Ce(t). Riesgo operacional moderado.",
    },
    "L2": {
        "label": "L2 — Sandbox Execution",
        "desc":  "Autonomia Confinada",
        "agent": "Libertad Restringida. Acciones reversibles o en sandbox.",
        "human": "Supervisor Tecnico. Configura los limites de la caja.",
        "econ":  "Riesgo Mitigado. Cc(p) reducido. U(t,p) positiva. Menos costo laboral.",
        "when":  "Pe=0, obs=0, ratio<50%. Transaccion estandar con confianza suficiente.",
    },
    "L3": {
        "label": "L3 — Human-on-the-Loop",
        "desc":  "Gestion por Excepcion",
        "agent": "Operacion Real-Time. Ejecuta hasta hallar incertidumbre.",
        "human": "Interventor Reactivo. Solo actua si la IA reporta duda.",
        "econ":  "Alta Eficiencia. Cc(p) casi cero. U(t,p) maxima. Optimiza tiempo del experto.",
        "when":  "Pe=0, obs=0, ratio bajo. Alta confianza contextual del agente.",
    },
    "L4": {
        "label": "L4 — High Agency",
        "desc":  "Autonomia Auditada",
        "agent": "Irrestricta. Operacion continua y autonoma.",
        "human": "Auditor Post-Mortem. Revisa resultados a posteriori.",
        "econ":  "Velocidad Maxima. Unit Economics optimizado. Error estadisticamente despreciable.",
        "when":  "Pe=0, transaccion limpia, alta confianza contextual. Agente opera sin supervision previa.",
    },
}


def consensus(votes: list) -> str:
    count      = Counter(votes)
    max_v      = max(count.values())
    candidates = [v for v, c in count.items() if c == max_v]
    return max(candidates, key=lambda x: int(x[1]))


# =====================================================================
# HTML
# =====================================================================
def build_html(all_results: list) -> str:
    data_js  = json.dumps(all_results,  ensure_ascii=True).replace("</", "<\\/")
    nivel_js = json.dumps(NIVEL_INFO,   ensure_ascii=True).replace("</", "<\\/")

    css = """*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#fff;font-family:'Courier New',monospace}
.hdr{background:#111;border-bottom:2px solid #f1c40f;padding:12px 20px;display:flex;align-items:center;gap:12px}
.hdr h1{color:#f1c40f;font-size:13px;letter-spacing:2px}
.badge{background:#f1c40f;color:#000;padding:2px 8px;font-size:10px;font-weight:bold}
.main{display:grid;grid-template-columns:1fr 360px;height:calc(100vh - 48px)}
.left{padding:14px;overflow-y:auto;border-right:1px solid #1e1e1e}
.right{background:#090909;overflow-y:auto;padding:14px}
.pw{height:3px;background:#1e1e1e;margin-bottom:12px}
.pf{height:100%;background:#f1c40f;transition:width .4s}
.dots{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px}
.dot{width:8px;height:8px;border-radius:50%;background:#1e1e1e;transition:background .3s}
.dot.done{background:#f1c40f}.dot.cur{background:#fff;box-shadow:0 0 4px #fff}
.card{background:#141414;border:1px solid #252525;border-radius:5px;margin-bottom:10px;overflow:hidden}
.chdr{background:#1a1a1a;padding:10px 14px;display:flex;align-items:center;gap:8px;cursor:pointer}
.chdr:hover{background:#202020}
.cbdy{display:none;padding:12px 14px;border-top:1px solid #1e1e1e}
.cbdy.open{display:block}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:bold}
.L0{background:#27ae60;color:#fff}.L1{background:#2ecc71;color:#000}
.L2{background:#f39c12;color:#000}.L3{background:#e67e22;color:#fff}.L4{background:#e74c3c;color:#fff}
.nbox{border-left:3px solid #f1c40f;background:#0c0c0c;border-radius:0 5px 5px 0;padding:10px 12px;margin:8px 0 12px}
.ntitle{color:#f1c40f;font-size:11px;font-weight:bold;margin-bottom:7px}
.nrow{display:flex;gap:8px;font-size:10px;margin-bottom:4px}
.nk{color:#555;min-width:65px;font-size:9px;text-transform:uppercase;padding-top:1px}
.nv{color:#ccc;line-height:1.5}.nvc{color:#f1c40f}
.tabs{display:flex;gap:5px;margin-bottom:10px;flex-wrap:wrap}
.tab{padding:3px 10px;background:#1a1a1a;border:1px solid #252525;border-radius:3px;cursor:pointer;font-size:10px;color:#888;font-family:inherit}
.tab.active{background:#f1c40f;color:#000;border-color:#f1c40f}
.panel{display:none}.panel.active{display:block}
.sl{font-size:9px;color:#666;text-transform:uppercase;letter-spacing:1px;margin:9px 0 3px}
.tbtn{background:none;border:1px solid #2a2a2a;color:#888;padding:4px 10px;font-size:10px;cursor:pointer;font-family:inherit;border-radius:3px;margin:3px 0 5px;display:block}
.tbtn:hover{border-color:#f1c40f;color:#f1c40f}
.tbox{background:#040404;border-left:3px solid #f1c40f;padding:10px;font-size:10px;color:#bbb;line-height:1.7;max-height:300px;overflow-y:auto;display:none;white-space:pre-wrap;word-break:break-word;margin-bottom:6px}
.tbox.open{display:block}
.jbox{background:#040f04;border:1px solid #152015;border-radius:3px;padding:8px;font-size:10px;color:#6ec86e;white-space:pre-wrap;margin-top:3px}
.gdec{background:#08080f;border-left:3px solid #3a7bd5;padding:8px 10px;font-size:11px;color:#90b0e0;border-radius:0 3px 3px 0;margin-top:4px;line-height:1.6}
.ok{color:#2ecc71;font-size:10px}.warn{color:#e74c3c;font-size:10px}
.sb{background:#141414;border:1px solid #1e1e1e;border-radius:5px;padding:10px;margin-bottom:8px}
.slbl{font-size:8px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}
.sval{font-size:16px;color:#f1c40f;font-weight:bold}
.ssub{font-size:9px;color:#444;margin-top:2px}
.dr{display:flex;align-items:center;gap:6px;margin-bottom:5px;font-size:10px}
.db{height:9px;border-radius:2px;transition:width .4s;min-width:2px}
.dc{color:#888;min-width:18px}
.nav{position:sticky;bottom:0;background:#090909;border-top:1px solid #1a1a1a;padding:8px 14px;display:flex;gap:8px;align-items:center}
.btn{background:#f1c40f;color:#000;border:none;padding:7px 16px;font-family:inherit;font-size:10px;font-weight:bold;cursor:pointer;border-radius:3px}
.btn:hover{background:#f39c12}.btn:disabled{background:#252525;color:#444;cursor:default}
.btn2{background:#1a1a1a;color:#666;border:1px solid #252525;padding:7px 14px;font-family:inherit;font-size:10px;cursor:pointer;border-radius:3px}
.btn2:hover{background:#202020;color:#ccc}
.ni{color:#555;font-size:9px;margin-left:auto}"""

    js = r"""
const D=__DATA__;
const N=__NIVEL__;
let r=0,st={t:0,p:0,s:0,a:0,d:{L0:0,L1:0,L2:0,L3:0,L4:0},u:0};
const f=n=>'$'+parseFloat(n).toLocaleString('es-CR',{minimumFractionDigits:2,maximumFractionDigits:2});
const e=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

function init(){
  const c=document.getElementById('stp');
  D.forEach((_,i)=>{const d=document.createElement('div');d.className='dot';d.id='dt'+i;c.appendChild(d);});
  document.getElementById('hs').textContent='0/'+D.length;
  document.getElementById('sc').textContent='0/'+D.length;
}

function buildCard(tx,idx){
  const votes=tx.trials.map(t=>t.vote);
  const unani=votes.every(v=>v===tx.decision);
  const nv=N[tx.decision]||{};
  let panels='';
  tx.trials.forEach((tr,ti)=>{
    const pe=(tr.audit_json&&tr.audit_json.riesgo&&tr.audit_json.riesgo.Pe)||[];
    const ceArr=(tr.audit_json&&tr.audit_json.riesgo&&tr.audit_json.riesgo.Ce)||[];
    const tA=tr.audit_think||'';
    const tG=tr.gov_think||'';
    const dA=tA.length>10?e(tA):'<em style="color:#333">Sin razonamiento capturado.</em>';
    const dG=tG.length>10?e(tG):'<em style="color:#333">Sin razonamiento capturado.</em>';
    const gAll=tG.split('\n').filter(l=>l.trim());
    const gLvl=gAll.filter(l=>/L[0-4]/i.test(l));
    const gConc=gLvl.length?e(gLvl[gLvl.length-1].trim()):e((N[tr.vote]||{}).desc||tr.vote);
    panels+=`<div class="panel ${ti===0?'active':''}" id="pn${idx}_${ti}">
<div class="sl">Voto: <span class="pill ${tr.vote}">${tr.vote}</span>
  ${pe.length?'<span class="warn">&#9888; '+pe.length+' inconsistencia(s)</span>':'<span class="ok">&#10003; Sin inconsistencias</span>'}
</div>
<div class="sl">Razonamiento Auditor (${tA.length} chars)</div>
<button class="tbtn" onclick="tog('tA${idx}_${ti}',this)">&#9654; mostrar pensamiento del auditor</button>
<div class="tbox" id="tA${idx}_${ti}">${dA}</div>
<div class="sl">Reporte del Auditor — variables Pe(t,p) y Ce(t)</div>
<div style="font-size:9px;color:#555;margin-bottom:3px">Pe(t,p) = Probabilidad de Error (inconsistencias) | Ce(t) = Observaciones operacionales</div>
<div class="jbox">${(()=>{
  const r=(tr.audit_json&&tr.audit_json.riesgo)||{};
  const inc=r.Pe||[];
  const obs=r.Ce||[];
  const nts=r.notas||[];
  return e(JSON.stringify({
    "Notas_Generales_Auditor": nts.length ? nts : ["(ninguna)"],
    "Pe(t,p) — Inconsistencias_matematicas": inc.length ? inc : ["ninguna — balances correctos"],
    "Ce(t) — Observaciones_operacionales": obs.length ? obs : ["ninguna"]
  },null,2));
})()}</div>
<div class="sl">Razonamiento Governor (${tG.length} chars)</div>
<button class="tbtn" onclick="tog('tG${idx}_${ti}',this)">&#9654; mostrar pensamiento del governor</button>
<div class="tbox" id="tG${idx}_${ti}">${dG}</div>
<div class="sl">Decision del Governor</div>
<div class="gdec"><strong>${e(tr.vote)}</strong> &mdash; ${e((N[tr.vote]||{}).desc||'')} &bull; ${gConc}</div>
</div>`;
  });
  const tabs=tx.trials.map((tr,ti)=>`<button class="tab ${ti===0?'active':''}" onclick="swTab(${idx},${ti})">Trial ${ti+1} <span class="pill ${tr.vote}" style="font-size:9px;padding:0 4px">${tr.vote}</span></button>`).join('');
  const pre=tx.precheck_issues&&tx.precheck_issues.length
    ?`<div class="warn" style="margin-bottom:6px">&#9888; ${e(tx.precheck_issues.join(' | '))}</div>`
    :'<div class="ok" style="margin-bottom:6px">&#10003; Precheck matematico: consistente</div>';
  const vh=votes.map(v=>`<span class="pill ${v}">${v}</span>`).join(' ');
  return `<div class="card" id="cd${idx}">
<div class="chdr" onclick="togCard(${idx})">
  <span style="color:#666;font-size:10px">Tx #${tx.tx_num}</span>
  <span style="color:#f1c40f;font-size:13px;font-weight:bold">${f(tx.amount)}</span>
  <span style="color:#888;font-size:10px;margin-left:auto">${f(tx.old_bal)} &rarr; ${f(tx.new_bal)}</span>
  <span class="pill ${tx.decision}" style="margin-left:8px">${tx.decision}</span>
  <span style="font-size:10px;color:#888;margin-left:5px">${e(nv.desc||'')}</span>
  <span style="font-size:9px;color:#444;margin-left:auto">${unani?'unanime':'mayoria'} ${tx.stability}/${tx.trials.length} ${tx.duration}s</span>
</div>
<div class="cbdy" id="bd${idx}">
  ${pre}
  <div class="sl">Votos: ${vh} &rarr; Consenso: <span class="pill ${tx.decision}">${tx.decision}</span></div>
  <div class="nbox">
    <div class="ntitle">${e(nv.label||tx.decision)}</div>
    <div class="nrow"><span class="nk">Agente</span><span class="nv">${e(nv.agent||'')}</span></div>
    <div class="nrow"><span class="nk">Humano</span><span class="nv">${e(nv.human||'')}</span></div>
    <div class="nrow"><span class="nk">Economia</span><span class="nv">${e(nv.econ||'')}</span></div>
    <div class="nrow"><span class="nk">Criterio</span><span class="nv nvc">${e(nv.when||'')}</span></div>
  </div>
  <div class="sl">Por que este consenso</div>
  <div style="background:#0a0a14;border-left:3px solid #3a7bd5;padding:8px 10px;font-size:10px;color:#90b0e0;border-radius:0 3px 3px 0;margin-bottom:10px;line-height:1.6">${e(tx.consensus_reason||"")}</div>
  <div class="sl">Detalle por trial</div>
  <div class="tabs">${tabs}</div>
  ${panels}
</div>
</div>`;
}

function togCard(i){document.getElementById('bd'+i).classList.toggle('open');}
function swTab(idx,ti){
  document.querySelectorAll('[id^="pn'+idx+'_"]').forEach(e=>e.classList.remove('active'));
  document.getElementById('pn'+idx+'_'+ti).classList.add('active');
  document.querySelectorAll('#cd'+idx+' .tab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('#cd'+idx+' .tab')[ti].classList.add('active');
}
function tog(id,btn){
  const el=document.getElementById(id);
  el.classList.toggle('open');
  btn.textContent=el.classList.contains('open')
    ?btn.textContent.replace('mostrar','ocultar').replace('\u25ba','\u25bc')
    :btn.textContent.replace('ocultar','mostrar').replace('\u25bc','\u25ba');
}
function updStats(tx){
  const l=tx.decision;
  st.d[l]=(st.d[l]||0)+1;st.t++;
  if(['L2','L3','L4'].includes(l)){st.p+=parseFloat(tx.amount);st.a++;}
  if(['L0','L1'].includes(l)) st.s+=50;
  if(tx.trials.map(t=>t.vote).every(v=>v===l)) st.u++;
  document.getElementById('sc').textContent=st.t+'/'+D.length;
  document.getElementById('hs').textContent=st.t+'/'+D.length;
  document.getElementById('sa').textContent=st.a+' anomalias';
  document.getElementById('sp').textContent=f(st.p);
  document.getElementById('ss').textContent=f(st.s);
  document.getElementById('su').textContent=st.u+'/'+st.t;
  const mx=Math.max(...Object.values(st.d),1);
  ['L0','L1','L2','L3','L4'].forEach((lv,i)=>{
    const v=st.d[lv]||0;
    document.getElementById('d'+i).style.width=(v/mx*120)+'px';
    document.getElementById('dc'+i).textContent=v;
  });
  document.getElementById('pf').style.width=(st.t/D.length*100)+'%';
}
function next(){
  if(r>=D.length){const b=document.getElementById('bn');b.textContent='\u2713 Completado';b.disabled=true;return;}
  const tx=D[r];
  document.getElementById('cards').insertAdjacentHTML('beforeend',buildCard(tx,r));
  updStats(tx);
  if(r>0)document.getElementById('dt'+(r-1)).className='dot done';
  document.getElementById('dt'+r).className='dot cur';
  document.getElementById('ni').textContent=(r+1)+' de '+D.length+' reveladas';
  r++;
  if(r>=D.length){
    document.getElementById('dt'+(r-1)).className='dot done';
    const b=document.getElementById('bn');b.textContent='\u2713 Completado';b.disabled=true;
  }
  setTimeout(()=>document.getElementById('cd'+(r-1)).scrollIntoView({behavior:'smooth',block:'start'}),100);
}
function showAll(){while(r<D.length)next();}
init();
"""

    js = js.replace("__DATA__",  data_js)
    js = js.replace("__NIVEL__", nivel_js)

    return (
        "<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
        "<title>GMD-AAM</title><style>" + css + "</style></head><body>"
        "<div class='hdr'><h1>GMD-AAM // AUTONOMY ALLOCATION MECHANISM</h1>"
        "<span class='badge' id='hm'>" + MODELO + "</span>"
        "<span class='badge' id='hs'>0/0</span></div>"
        "<div class='main'>"
        "<div class='left'>"
        "<div class='pw'><div class='pf' id='pf' style='width:0'></div></div>"
        "<div class='dots' id='stp'></div>"
        "<div id='cards'></div>"
        "<div class='nav'>"
        "<button class='btn' id='bn' onclick='next()'>&#9658; SIGUIENTE TX</button>"
        "<button class='btn2' onclick='showAll()'>&#9193; MOSTRAR TODO</button>"
        "<span class='ni' id='ni'>Click para revelar una a una</span>"
        "</div></div>"
        "<div class='right'>"
        "<div class='sb'><div class='slbl'>Transacciones</div>"
        "<div class='sval' id='sc'>0/0</div><div class='ssub' id='sa'>anomalias detectadas</div></div>"
        "<div class='sb'><div class='slbl'>Capital en alerta L2-L4</div>"
        "<div class='sval' id='sp'>$0.00</div><div class='ssub'>requiere revision humana</div></div>"
        "<div class='sb'><div class='slbl'>OpEx savings L0-L1</div>"
        "<div class='sval' id='ss' style='color:#2ecc71'>$0.00</div>"
        "<div class='ssub'>$50 por transaccion automatizada</div></div>"
        "<div class='sb'><div class='slbl'>Distribucion soberania</div>"
        "<div style='margin-top:6px'>"
        "<div class='dr'><span class='pill L0' style='width:26px;text-align:center'>L0</span><div class='db' id='d0' style='background:#27ae60'></div><span class='dc' id='dc0'>0</span></div>"
        "<div class='dr'><span class='pill L1' style='width:26px;text-align:center'>L1</span><div class='db' id='d1' style='background:#2ecc71'></div><span class='dc' id='dc1'>0</span></div>"
        "<div class='dr'><span class='pill L2' style='width:26px;text-align:center'>L2</span><div class='db' id='d2' style='background:#f39c12'></div><span class='dc' id='dc2'>0</span></div>"
        "<div class='dr'><span class='pill L3' style='width:26px;text-align:center'>L3</span><div class='db' id='d3' style='background:#e67e22'></div><span class='dc' id='dc3'>0</span></div>"
        "<div class='dr'><span class='pill L4' style='width:26px;text-align:center'>L4</span><div class='db' id='d4' style='background:#e74c3c'></div><span class='dc' id='dc4'>0</span></div>"
        "</div></div>"
        "<div class='sb'><div class='slbl'>Consenso unanime</div>"
        "<div class='sval' id='su'>-</div>"
        "<div class='ssub'>trials identicos / total tx</div></div>"
        "</div></div>"
        "<script>" + js + "</script>"
        "</body></html>"
    )


# =====================================================================
# MOTOR PRINCIPAL
# =====================================================================
def run(limit: int = LIMIT, trials: int = TRIALS):
    print(f"\n{'='*60}")
    print(f"  GMD-AAM | {MODELO} | Trials={trials} | Txs={limit}")
    print(f"{'='*60}\n")

    df = pd.read_csv(DATA_PATH).head(limit)
    all_results = []
    os.makedirs("results", exist_ok=True)

    for i, row in df.iterrows():
        tx_num  = i + 1
        amount  = float(row["amount"])
        old_bal = float(row["oldbalanceOrg"])
        new_bal = float(row["newbalanceOrig"])

        print(f"\n{'─'*50}")
        print(f"  Tx #{tx_num}  |  ${amount:,.2f}  |  ${old_bal:,.2f} -> ${new_bal:,.2f}")

        pc = precheck(amount, old_bal, new_bal)
        if pc["issues"]:
            print(f"  ALERTA: {pc['issues']}")

        trial_data = []
        t0 = time.time()
        for t in range(trials):
            trial_data.append(run_trial(amount, old_bal, new_bal, pc, t))

        votes     = [td["vote"] for td in trial_data]
        decision  = consensus(votes)
        duration  = round(time.time() - t0, 1)
        stability = votes.count(decision)

        print(f"  CONSENSO: {decision}  votos={votes}  {duration}s")

        vote_count = Counter(votes)
        if stability == len(votes):
            consensus_reason = (
                f"Unanimidad: los {len(votes)} trials votaron {decision}. "
                f"Alta confianza en la decision."
            )
        else:
            parts_r = []
            for lv, cnt in sorted(vote_count.items()):
                pct = round(cnt / len(votes) * 100)
                parts_r.append(f"{lv}: {cnt}/{len(votes)} trials ({pct}%)")
            winner_cnt = vote_count[decision]
            consensus_reason = (
                f"Mayoria simple: {decision} obtuvo {winner_cnt}/{len(votes)} votos. "
                f"Distribucion: {', '.join(parts_r)}. "
                f"En empate se elige el nivel mas alto (mas conservador)."
            )

        all_results.append({
            "tx_num":           tx_num,
            "amount":           amount,
            "old_bal":          old_bal,
            "new_bal":          new_bal,
            "precheck_clean":   pc["clean"],
            "precheck_issues":  pc["issues"],
            "trials":           trial_data,
            "votes":            votes,
            "decision":         decision,
            "stability":        stability,
            "duration":         duration,
            "consensus_reason": consensus_reason,
        })

        csv_rows = []
        for res in all_results:
            for tr in res["trials"]:
                csv_rows.append({
                    "tx_num":                      res["tx_num"],
                    "amount":                      res["amount"],
                    "old_bal":                     res["old_bal"],
                    "new_bal":                     res["new_bal"],
                    "anomaly":                     not res["precheck_clean"],
                    "precheck":                    " | ".join(res["precheck_issues"]) if res["precheck_issues"] else "ok",
                    "trial":                       tr["trial"],
                    "vote":                        tr["vote"],
                    "inconsistencias_matematicas": str(tr["audit_json"].get("riesgo", {}).get("Pe", [])),
                    "observaciones_operacionales": str(tr["audit_json"].get("riesgo", {}).get("Ce", [])),
                    "notas_generales":             str(tr["audit_json"].get("riesgo", {}).get("notas", [])),
                    "audit_think":                 (tr["audit_think"] or "")[:2000],
                    "gov_think":                   (tr["gov_think"]   or "")[:1500],
                    "consensus":                   res["decision"],
                    "stability":                   res["stability"],
                    "duration_s":                  res["duration"],
                })
        pd.DataFrame(csv_rows).to_csv(
            "results/GMD_LOG.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame([{
            "tx_num":    r["tx_num"],    "amount":    r["amount"],
            "decision":  r["decision"],  "votes":     str(r["votes"]),
            "stability": r["stability"], "duration":  r["duration"],
            "anomaly":   not r["precheck_clean"],
        } for r in all_results]).to_csv("results/GMD_SUMMARY.csv", index=False)

    decisions = [r["decision"] for r in all_results]
    dist      = Counter(decisions)
    anomalas  = sum(r["amount"] for r in all_results if r["decision"] in ["L2","L3","L4"])
    savings   = (dist.get("L0", 0) + dist.get("L1", 0)) * 50

    print(f"\n{'='*60}")
    print(f"  Capital en alerta: ${anomalas:,.2f}")
    print(f"  OpEx savings:      ${savings:,.2f}")
    for lv in ["L0","L1","L2","L3","L4"]:
        print(f"  {lv}: {'#'*dist.get(lv,0)} {dist.get(lv,0)}")
    print(f"{'='*60}")

    html    = build_html(all_results)
    out     = "results/gmd_think_visible.html"
    abs_out = os.path.abspath(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML: {abs_out}")
    print(f"CSV:  {os.path.abspath('results/GMD_LOG.csv')}")
    try:
        subprocess.Popen(["xdg-open", abs_out])
    except Exception:
        webbrowser.open(f"file://{abs_out}")


if __name__ == "__main__":
    run()