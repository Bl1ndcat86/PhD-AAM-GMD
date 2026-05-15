"""
GMD-AAM: Governance by Mechanism Design — Modo Batch Académico
==============================================================
Versión optimizada que PRESERVA la arquitectura de agentes completa:
  - Auditor real con Sensing Effort (Pe, Ce) y separación de notas cualitativas.
  - Governor real con matriz de utilidad U(t,p)
  - Trials múltiples con consenso
  - Trazabilidad completa en CSV + HTML

Optimizaciones legítimas vs main.py:
  1. TRIALS reducibles por CLI (default 1 para batch rápido, 3 para tesis)
  2. num_predict reducido en modo batch (500 vs 2000) — respuesta más corta
     pero sigue siendo razonamiento real del LLM
  3. Checkpoint automático — puedes interrumpir y continuar
  4. Progress bar honesta con ETA
  5. Sin generación de HTML interactivo durante el loop (se genera al final)
"""

import os
import re
import sys
import json
import time
import argparse
import requests
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO     = "deepseek-r1:14b"
DATA_PATH  = "data/PS_20174392719_1491204439457_log.csv"
LAMBDA_COEF = 5.0

# Opciones del LLM por modo
LLM_OPTS = {
    "full": {
        "auditor":  {"temperature": 0.3, "num_predict": 2000, "num_ctx": 4096},
        "governor": {"temperature": 0.4, "num_predict": 2000, "num_ctx": 4096},
    },
    "batch": {
        "auditor":  {"temperature": 0.3, "num_predict": 500,  "num_ctx": 2048},
        "governor": {"temperature": 0.4, "num_predict": 400,  "num_ctx": 2048},
    },
}

# Pesos de riesgo por nivel — igual que main.py
RISK_WEIGHTS = {"L0": 0.00, "L1": 0.05, "L2": 0.30, "L3": 0.80, "L4": 1.00}
CC_MAP       = {"L0": 100.0, "L1": 50.0, "L2": 15.0, "L3": 5.0,  "L4": 0.0}

NIVEL_INFO = {
    "L0": {"label": "L0 — Shadow Mode",           "desc": "Aumentacion Consultiva"},
    "L1": {"label": "L1 — Human-in-the-Loop",     "desc": "Autonomia Tutelada"},
    "L2": {"label": "L2 — Sandbox Execution",     "desc": "Autonomia Confinada"},
    "L3": {"label": "L3 — Human-on-the-Loop",     "desc": "Gestion por Excepcion"},
    "L4": {"label": "L4 — High Agency / Bloqueo", "desc": "Autonomia Auditada"},
}


# =====================================================================
# UTILIDADES DE CONSOLA
# =====================================================================
class Progress:
    """Barra de progreso honesta con ETA."""
    def __init__(self, total: int, trials: int):
        self.total   = total
        self.trials  = trials
        self.done    = 0
        self.t0      = time.time()
        self.errors  = 0

    def update(self, tx_num: int, decision: str, votes: list, duration: float):
        self.done += 1
        elapsed = time.time() - self.t0
        rate    = self.done / elapsed if elapsed > 0 else 1
        eta_s   = (self.total - self.done) / rate if rate > 0 else 0
        eta     = str(timedelta(seconds=int(eta_s)))
        pct     = int(self.done / self.total * 100)
        bar     = "█" * (pct // 5) + "░" * (20 - pct // 5)
        votes_s = " ".join(votes)
        print(
            f"\r  [{bar}] {pct:3d}% | Tx {tx_num:>4}/{self.total} | "
            f"{decision} [{votes_s}] | {duration:.1f}s/tx | ETA {eta}   ",
            end="", flush=True
        )

    def done_line(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.t0)))
        print(f"\n  Completado en {elapsed}")


# =====================================================================
# OLLAMA — stream=True, igual que main.py
# =====================================================================
def ollama_call(prompt: str, options: dict, role: str = "",
                prefill: str = "", mode: str = "governor") -> tuple[str, str]:
    raw = ""
    full_prompt = prompt.rstrip() + "\n<think>\n" + prefill if prefill else prompt.rstrip() + "\n<think>"
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODELO, "prompt": full_prompt,
                  "stream": True, "options": options},
            timeout=300, stream=True,
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
        print(f"\n  [{role}] Timeout")
    except Exception as e:
        print(f"\n  [{role}] Error: {e}")
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
    else:
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
# VALIDACIÓN MATEMÁTICA (idéntico a main.py)
# =====================================================================
def precheck(amount: float, old_bal: float, new_bal: float) -> dict:
    issues = []
    expected = round(old_bal - amount, 2)
    actual   = round(new_bal, 2)
    if abs(expected - actual) > 0.01:
        issues.append(f"Balance esperado ${expected} != real ${actual} (diff: ${abs(expected - actual):.2f})")
    if amount > old_bal:
        issues.append(f"Monto ${amount} excede balance ${old_bal} (exceso: ${amount - old_bal:.2f})")
    if new_bal == 0.0 and abs(amount - old_bal) > 0.01:
        issues.append(f"Cuenta vaciada a $0 pero monto ${amount} != balance ${old_bal}")
    return {"issues": issues, "clean": len(issues) == 0}


# =====================================================================
# AGENTE 1: AUDITOR — Sensing Effort real
# =====================================================================
def agent_auditor(amount, old_bal, new_bal, issues, opts) -> tuple[dict, str]:
    expected = round(old_bal - amount, 2)
    ratio    = round(amount / old_bal * 100, 1) if old_bal > 0 else 0
    issue_block = ""
    if issues:
        issue_block = "\nALERTA PRECHECK:\n" + "\n".join("- " + i for i in issues)

    prompt = (
        "Eres el Agente Auditor del sistema GMD-AAM. Responde en español.\n"
        "Sensing Effort: analiza la transaccion para estimar Pe(t,p).\n\n"
        f"TRANSACCION:\n"
        f"  Monto: ${amount} | Balance inicial: ${old_bal} | Balance final: ${new_bal}\n"
        f"  Balance esperado: ${expected} | Ratio: {ratio}%"
        + issue_block + "\n\n"
        "Analiza brevemente:\n"
        f"  1. CONSISTENCIA MATEMATICA: esperado=${expected}, real=${new_bal}, "
        f"diff=${abs(expected - new_bal):.2f} -> "
        + ("INCONSISTENCIA" if abs(expected - new_bal) > 0.01 else "OK") + "\n"
        f"  2. VIABILIDAD: monto ${amount} vs balance ${old_bal} -> "
        + ("PROBLEMA" if amount > old_bal else "OK") + "\n"
        f"  3. VACIADO: balance final ${new_bal}"
        + (" <- CERO ANOMALO" if new_bal == 0.0 and abs(amount - old_bal) > 0.01 else " -> OK") + "\n\n"
        "Responde SOLO con el JSON separando tus notas de los riesgos reales (deja Pe/Ce vacios [] si todo esta OK):\n"
        '{"riesgo": {"notas": ["tu razonamiento general y notas cualitativas aqui"], "Pe": ["solo si hay error matematico"], "Ce": ["solo si hay un riesgo operacional real"]}}'
    )

    clean, think = ollama_call(prompt, opts, role="Auditor",
                               prefill="Analizo los 3 criterios en español:",
                               mode="auditor")

    result = {"riesgo": {"notas": [], "Pe": issues[:] if issues else [], "Ce": []}}
    for pattern in [r'```json\s*(.*?)\s*```', r'(\{[^{}]*"riesgo"[^{}]*\})', r'(\{.*\})']:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                riesgo = parsed.get("riesgo", {})
                
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

    return result, think


# =====================================================================
# AGENTE 2: GOVERNOR — Matriz de utilidad real
# =====================================================================
def agent_governor(audit: dict, amount, old_bal, opts) -> tuple[str, str]:
    audit_str = json.dumps(audit, ensure_ascii=False)
    pe_list   = audit.get("riesgo", {}).get("Pe", [])
    ce_list   = audit.get("riesgo", {}).get("Ce", [])
    pe_count  = len(pe_list)
    ce_count  = len(ce_list)
    ratio     = round(amount / old_bal * 100, 1) if old_bal > 0 else 0

    V_task  = max(round(amount * 0.005, 2), 1.0)
    Ce_est  = amount
    if pe_count == 0:
        Pe_base = 0.0
    elif pe_count == 1:
        Pe_base = 0.35
    else:
        Pe_base = min(0.35 * pe_count, 0.95)

    pareto = {
        lv: round(V_task - CC_MAP[lv] - (Pe_base * RISK_WEIGHTS[lv] * Ce_est * LAMBDA_COEF), 2)
        for lv in ["L0", "L1", "L2", "L3", "L4"]
    }
    best = max(pareto, key=pareto.get)

    prompt = (
        "Eres el Risk Governor del sistema GMD-AAM. Responde en español.\n"
        "Aplica U(t,p) = V(t) - Cc(p) - (Pe x Ce x lambda) para elegir L0-L4.\n\n"
        f"REPORTE AUDITOR: {audit_str}\n\n"
        f"MATRIZ DE PAGOS U(t,p):\n{json.dumps(pareto, indent=2)}\n\n"
        f"PARAMETROS: V(t)=${V_task} | Ce(t)=${amount} | Pe={Pe_base} | lambda={LAMBDA_COEF} | ratio={ratio}%\n"
        f"Inconsistencias Pe={pe_count}: {pe_list}\n"
        f"Observaciones Ce={ce_count}: {ce_list}\n\n"
        "Escala (autonomia del agente, no nivel de riesgo):\n"
        "  L0=Shadow(solo lectura) | L1=HITL(firma humana) | L2=Sandbox(reversible)\n"
        "  L3=HotL(excepcion) | L4=HighAgency(bloqueo automatico si Pe grave)\n\n"
        "Razona brevemente por que el nivel de mayor U(t,p) es el correcto.\n"
        "Termina con exactamente: NIVEL: L#"
    )

    clean, think = ollama_call(prompt, opts, role="Governor",
                               prefill="Evaluo la Matriz de Pagos en español:",
                               mode="governor")

    nivel = None
    for text in [clean, think]:
        matches = re.findall(r"\bL([0-4])\b", text.upper())
        if matches:
            nivel = "L" + matches[-1]
            break

    if nivel is None:
        nivel = best

    return nivel, think


# =====================================================================
# TRIAL
# =====================================================================
def run_trial(amount, old_bal, new_bal, pc, trial_num, opts) -> dict:
    audit, audit_think = agent_auditor(amount, old_bal, new_bal, pc["issues"], opts["auditor"])
    level, gov_think   = agent_governor(audit, amount, old_bal, opts["governor"])
    return {
        "trial":       trial_num + 1,
        "vote":        level,
        "audit_json":  audit,
        "audit_think": audit_think,
        "gov_think":   gov_think,
    }


def consensus(votes: list) -> str:
    count      = Counter(votes)
    max_v      = max(count.values())
    candidates = [v for v, c in count.items() if c == max_v]
    return max(candidates, key=lambda x: int(x[1]))


# =====================================================================
# CHECKPOINT
# =====================================================================
CKPT_PATH = "results/GMD_CHECKPOINT.json"

def load_checkpoint() -> list:
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint encontrado: {len(data)} transacciones ya procesadas.")
        return data
    return []

def save_checkpoint(results: list):
    os.makedirs("results", exist_ok=True)
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


# =====================================================================
# EXPORTACIÓN CSV
# =====================================================================
def export_csv(all_results: list):
    csv_rows = []
    for res in all_results:
        for tr in res["trials"]:
            csv_rows.append({
                "tx_num":          res["tx_num"],
                "amount":          res["amount"],
                "old_bal":         res["old_bal"],
                "new_bal":         res["new_bal"],
                "anomaly":         not res["precheck_clean"],
                "precheck":        " | ".join(res["precheck_issues"]) if res["precheck_issues"] else "ok",
                "trial":           tr["trial"],
                "vote":            tr["vote"],
                "Pe_issues":       str(tr["audit_json"].get("riesgo", {}).get("Pe", [])),
                "Ce_issues":       str(tr["audit_json"].get("riesgo", {}).get("Ce", [])),
                "notas_generales": str(tr["audit_json"].get("riesgo", {}).get("notas", [])),
                "audit_think":     (tr["audit_think"] or "")[:2000],
                "gov_think":       (tr["gov_think"]   or "")[:1500],
                "consensus":       res["decision"],
                "stability":       res["stability"],
                "duration_s":      res["duration"],
            })
    pd.DataFrame(csv_rows).to_csv(
        "results/GMD_BATCH_LOG.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([{
        "tx_num":    r["tx_num"],
        "amount":    r["amount"],
        "decision":  r["decision"],
        "votes":     str(r["votes"]),
        "stability": r["stability"],
        "duration":  r["duration"],
        "anomaly":   not r["precheck_clean"],
    } for r in all_results]).to_csv("results/GMD_BATCH_SUMMARY.csv", index=False)


# =====================================================================
# VALIDACIÓN DE HIPÓTESIS
# =====================================================================
def validate_hypotheses(all_results: list, trials_count: int):
    print("\n" + "=" * 60)
    print("  VALIDACIÓN DE HIPÓTESIS — GMD-AAM")
    print("=" * 60)

    total_v    = 0.0
    cost_aam   = 0.0
    cost_lf    = 0.0
    cost_hitl  = 0.0

    for res in all_results:
        amount   = res["amount"]
        decision = res["decision"]
        V_task   = max(amount * 0.005, 1.0)
        total_v += V_task

        pe_counts = []
        for tr in res["trials"]:
            pe_list = tr["audit_json"].get("riesgo", {}).get("Pe", [])
            pe_counts.append(len(pe_list))
        avg_pe_count = sum(pe_counts) / len(pe_counts) if pe_counts else 0

        if avg_pe_count == 0:
            Pe = 0.0
        elif avg_pe_count < 1:
            Pe = 0.35 * avg_pe_count
        else:
            Pe = min(0.35 * avg_pe_count, 0.95)

        aam_risk = RISK_WEIGHTS.get(decision, 1.0)
        cost_aam  += CC_MAP[decision] + (Pe * aam_risk  * amount * LAMBDA_COEF)
        cost_lf   += 0.0              + (Pe * 1.00      * amount * LAMBDA_COEF)
        cost_hitl += 50.0             + (Pe * RISK_WEIGHTS["L1"] * amount * LAMBDA_COEF)

    u_aam  = total_v - cost_aam
    u_lf   = total_v - cost_lf
    u_hitl = total_v - cost_hitl

    print(f"  Muestra:            {len(all_results)} transacciones | {trials_count} trial(s) c/u")
    print(f"  V(t) total:         ${total_v:>15,.2f}")
    print(f"  U neta AAM:         ${u_aam:>15,.2f}")
    print(f"  U neta Laissez-faire:${u_lf:>14,.2f}")
    print(f"  U neta Fixed HITL:  ${u_hitl:>15,.2f}")
    print("-" * 60)

    h1 = u_aam > u_lf
    h2 = u_aam > u_hitl
    print(f"  H1 {'✓ VALIDADA' if h1 else '✗ NO VALIDADA'}: "
          f"AAM {'>' if h1 else '<='} Laissez-faire "
          f"(Δ ${u_aam - u_lf:+,.2f})")
    print(f"  H2 {'✓ VALIDADA' if h2 else '✗ NO VALIDADA'}: "
          f"AAM {'>' if h2 else '<='} Fixed HITL "
          f"(Δ ${u_aam - u_hitl:+,.2f})")
    print("=" * 60)
    print()

    if trials_count == 1:
        print("  NOTA METODOLÓGICA: Esta corrida usó trials=1 (modo batch rápido).")
        print("  Para la defensa, ejecutar con --trials 3 para estabilidad inter-trial.")
        print("  Los resultados de H1/H2 son preliminares; la tendencia debe confirmarse.")
        print()

    return {"H1": h1, "H2": h2, "U_AAM": u_aam, "U_LF": u_lf, "U_HITL": u_hitl}


# =====================================================================
# REPORTE FINAL EN CONSOLA
# =====================================================================
def print_summary(all_results: list):
    decisions = [r["decision"] for r in all_results]
    dist      = Counter(decisions)
    anomalas  = sum(r["amount"] for r in all_results if r["decision"] in ["L2","L3","L4"])
    savings   = (dist.get("L0", 0) + dist.get("L1", 0)) * 50
    unanimous = sum(1 for r in all_results if r["stability"] == len(r["trials"]))

    print(f"\n{'='*60}")
    print(f"  DISTRIBUCIÓN DE SOBERANÍA")
    print(f"{'='*60}")
    for lv in ["L0","L1","L2","L3","L4"]:
        n   = dist.get(lv, 0)
        bar = "█" * n if n < 40 else "█" * 40 + f"+{n-40}"
        print(f"  {lv}: {bar} {n}")
    print(f"  Capital en alerta (L2-L4): ${anomalas:,.2f}")
    print(f"  OpEx savings (L0-L1):      ${savings:,.2f}")
    print(f"  Consenso unánime:          {unanimous}/{len(all_results)} tx")
    print(f"{'='*60}")


# =====================================================================
# MOTOR PRINCIPAL
# =====================================================================
def run(limit: int, trials: int, mode: str, resume: bool):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    opts      = LLM_OPTS[mode]

    print(f"\n{'='*60}")
    print(f"  GMD-AAM BATCH | {MODELO}")
    print(f"  Modo: {mode.upper()} | Trials: {trials} | Txs: {limit}")
    print(f"  Inicio: {timestamp}")
    if mode == "batch":
        print(f"  [BATCH] num_predict reducido. Razonamiento real, respuestas compactas.")
    print(f"{'='*60}\n")

    df = pd.read_csv(DATA_PATH).head(limit)
    os.makedirs("results", exist_ok=True)

    all_results = load_checkpoint() if resume else []
    done_txs    = {r["tx_num"] for r in all_results}

    progress = Progress(limit, trials)

    for i, row in df.iterrows():
        tx_num  = i + 1
        if tx_num in done_txs:
            progress.update(tx_num, all_results[tx_num-1]["decision"],
                            all_results[tx_num-1]["votes"],
                            all_results[tx_num-1]["duration"])
            continue

        amount  = float(row["amount"])
        old_bal = float(row["oldbalanceOrg"])
        new_bal = float(row["newbalanceOrig"])
        pc      = precheck(amount, old_bal, new_bal)

        trial_data = []
        t0 = time.time()
        for t in range(trials):
            trial_data.append(run_trial(amount, old_bal, new_bal, pc, t, opts))

        votes     = [td["vote"] for td in trial_data]
        decision  = consensus(votes)
        duration  = round(time.time() - t0, 1)
        stability = votes.count(decision)

        vote_count = Counter(votes)
        if stability == len(votes):
            consensus_reason = f"Unanimidad: {len(votes)} trials votaron {decision}."
        else:
            parts_r = [f"{lv}: {cnt}/{len(votes)}" for lv, cnt in sorted(vote_count.items())]
            consensus_reason = (
                f"Mayoria: {decision} obtuvo {vote_count[decision]}/{len(votes)} votos. "
                f"Distribucion: {', '.join(parts_r)}."
            )

        result = {
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
        }
        all_results.append(result)
        progress.update(tx_num, decision, votes, duration)

        if tx_num % 10 == 0:
            save_checkpoint(all_results)
            export_csv(all_results)

    progress.done_line()
    export_csv(all_results)
    save_checkpoint(all_results)

    print_summary(all_results)
    hyp = validate_hypotheses(all_results, trials)

    meta = {
        "timestamp": timestamp,
        "model": MODELO,
        "mode": mode,
        "trials": trials,
        "limit": limit,
        "H1_validated": hyp["H1"],
        "H2_validated": hyp["H2"],
        "U_AAM":  hyp["U_AAM"],
        "U_LF":   hyp["U_LF"],
        "U_HITL": hyp["U_HITL"],
    }
    with open("results/GMD_BATCH_META.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Archivos generados:")
    print(f"    results/GMD_BATCH_LOG.csv   — trazabilidad completa por trial")
    print(f"    results/GMD_BATCH_SUMMARY.csv — resumen por transacción")
    print(f"    results/GMD_BATCH_META.json   — metadatos de la corrida")
    print(f"    results/GMD_CHECKPOINT.json   — checkpoint (reanudar con --resume)")
    print()


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMD-AAM Batch — validación académica con agentes reales"
    )
    parser.add_argument("--limit",  type=int, default=385,
                        help="Número de transacciones (default: 385)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Trials por transacción (default: 1). Usar 3 para tesis.")
    parser.add_argument("--full",   action="store_true",
                        help="Equivalente exacto a main.py (trials=3, num_predict=2000)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde checkpoint si existe")
    args = parser.parse_args()

    if args.full:
        args.trials = 3
        mode = "full"
    else:
        mode = "batch"

    if args.trials < 1 or args.trials > 10:
        print("Error: --trials debe estar entre 1 y 10")
        sys.exit(1)

    run(limit=args.limit, trials=args.trials, mode=mode, resume=args.resume)