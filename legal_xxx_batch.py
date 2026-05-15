"""
GMD-AAM: Dominio Legal — Versión XXX (Limpia)
==============================================
Versión final alineada con xxx.py.
  - Aislamiento de notas cualitativas (no inflan Pe/Ce).
  - Lambda = 50.0 (Aversión extrema).
  - Guardrail Determinista (Seguridad regulatoria).
  - Basado en el dataset legal_clean.json (CUAD).
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
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODELO      = "deepseek-r1:14b"
DATA_PATH   = "data/legal_clean.json"
LAMBDA_COEF = 50.0

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

RISK_WEIGHTS = {"L0": 0.00, "L1": 0.05, "L2": 0.30, "L3": 0.80, "L4": 1.00}
CC_MAP       = {"L0": 100.0, "L1": 50.0, "L2": 15.0, "L3": 5.0, "L4": 0.0}

# =====================================================================
# UTILIDADES DE CONSOLA
# =====================================================================
class Progress:
    def __init__(self, total, trials):
        self.total  = total
        self.trials = trials
        self.done   = 0
        self.t0     = time.time()

    def update(self, num, decision, votes, duration, guardrails=0):
        self.done += 1
        elapsed = time.time() - self.t0
        rate    = self.done / elapsed if elapsed > 0 else 1
        eta_s   = (self.total - self.done) / rate if rate > 0 else 0
        eta     = str(timedelta(seconds=int(eta_s)))
        pct     = int(self.done / self.total * 100)
        bar     = "█" * (pct // 5) + "░" * (20 - pct // 5)
        grd     = f" GRD={guardrails}" if guardrails > 0 else ""
        print(
            f"\r  [{bar}] {pct:3d}% | Contrato {num:>4}/{self.total} | "
            f"{decision} [{' '.join(votes)}]{grd} | {duration:.1f}s | ETA {eta}   ",
            end="", flush=True
        )

    def done_line(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.t0)))
        print(f"\n  Completado en {elapsed}")

# =====================================================================
# GUARDRAIL DETERMINISTA
# =====================================================================
def apply_guardrail(llm_vote, pe_count):
    if pe_count >= 2:
        if llm_vote != "L0":
            return "L0", True, f"Guardrail λ=50: Pe={pe_count} graves detectadas. Forzado L0."
    elif pe_count == 1:
        if llm_vote in ["L3", "L4"]:
            return "L1", True, f"Guardrail λ=50: Pe=1 riesgo detectado. Forzado L1 mínima supervisión."
    return llm_vote, False, ""

# =====================================================================
# OLLAMA & PARSER (Fix de Notas)
# =====================================================================
def ollama_call(prompt, options, role="", prefill="", mode="governor"):
    raw = ""
    full_prompt = prompt.rstrip() + "\n<think>\n" + prefill if prefill else prompt.rstrip() + "\n<think>"
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODELO, "prompt": full_prompt, "stream": True, "options": options},
            timeout=300, stream=True
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line: continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                raw += chunk.get("response", "")
                if chunk.get("done", False): break
            except: continue
    except Exception as e:
        print(f"\n  [{role}] Error: {e}")
    return _parse_think(raw, mode=mode)

def _parse_think(raw, mode="governor"):
    raw = raw.strip()
    if "<think>" in raw and "</think>" in raw:
        m = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
        if m: return m.group(2).strip(), m.group(1).strip()
    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        if len(parts) > 1 and parts[1].strip(): return parts[1].strip(), parts[0].strip()
    
    if mode == "auditor":
        for pat in [r"```json", r'\{"riesgo"']:
            m = re.search(pat, raw, re.DOTALL)
            if m and m.start() > 20: return raw[m.start():].strip(), raw[:m.start()].strip()
    else:
        for pat in [r"NIVEL\s*:\s*L[0-4]", r"Conclusi[oó]n\s+[Ff]inal"]:
            matches = list(re.finditer(pat, raw, re.IGNORECASE | re.DOTALL))
            if matches:
                pos = matches[-1].start()
                if pos > 30: return raw[pos:].strip(), raw[:pos].strip()
        nivel_m = list(re.finditer(r"L[0-4]", raw.upper()))
        if nivel_m:
            last = nivel_m[-1]
            start = raw.rfind(chr(10), 0, last.start())
            if start > 30: return raw[start:].strip(), raw[:start].strip()
    return raw, raw

# =====================================================================
# AGENTES
# =====================================================================
def agent_auditor(row, pc, opts):
    clauses = pc["clauses_summary"]
    issue_block = "\nALERTA PRECHECK:\n" + "\n".join("- " + i for i in pc["issues"]) if pc["issues"] else ""
    
    prompt = (
        "Eres el Agente Auditor GMD-AAM (Dominio Legal). Responde siempre en español.\n"
        "Sensing Effort: analiza el contrato para estimar Pe(t,p).\n\n"
        f"CONTRATO: {pc['doc_name']}\n"
        f"RESUMEN CLÁUSULAS: {json.dumps(clauses, indent=2)}\n"
        f"{issue_block}\n\n"
        "Analiza 3 criterios: Consistencia Interna, Riesgos (cláusulas expuestas) y Fechas/Integridad.\n"
        "Responde SOLO con el JSON separando tus notas de los riesgos reales (deja inconsistencias y observaciones vacias [] si todo esta OK):\n"
        '{"riesgo": {"notas": ["tu razonamiento cualitativo aqui"], "inconsistencias": ["solo si hay contradiccion legal o error"], "observaciones": ["solo exposicion contractual real"]}}'
    )
    
    clean, think = ollama_call(prompt, opts, role="Auditor", prefill="Analizo los criterios en español:", mode="auditor")
    
    result = {"riesgo": {"notas": [], "Pe": pc["issues"][:], "Ce": []}}
    for pattern in [r'```json\s*(.*?)\s*```', r'(\{[^{}]*"riesgo"[^{}]*\})', r'(\{.*\})']:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                riesgo = parsed.get("riesgo", {})
                
                notas = riesgo.get("notas", [])
                pe = riesgo.get("inconsistencias") or riesgo.get("Pe") or []
                ce = riesgo.get("observaciones") or riesgo.get("Ce") or []
                
                ruido = ["ninguna", "ninguno", "todo en orden", "ok", "correcto", "no hay", "consistente", "cumple", "vacío"]
                pe = [p for p in pe if not any(r in str(p).lower() for r in ruido)]
                ce = [c for c in ce if not any(r in str(c).lower() for r in ruido)]
                
                result = {"riesgo": {"notas": notas, "Pe": pe + pc["issues"], "Ce": ce}}
                break
            except: continue

    return result, think

def agent_governor(audit, pc, opts):
    pe_list, ce_list = audit["riesgo"]["Pe"], audit["riesgo"]["Ce"]
    pe_count, ce_count = len(pe_list), len(ce_list)
    
    V_task = max(10.0 - (pc["high_risk_count"] * 2.0), 1.0)
    Ce_est = 50.0 + (pc["high_risk_count"] * 100.0)
    Pe_base = 0.0 if pe_count == 0 else (0.35 if pe_count == 1 else min(0.35 * pe_count, 0.95))
    
    pareto = {lv: round(V_task - CC_MAP[lv] - (Pe_base * RISK_WEIGHTS[lv] * Ce_est * LAMBDA_COEF), 2) for lv in CC_MAP}
    best = max(pareto, key=pareto.get)
    
    prompt = (
        "Eres el Risk Governor GMD-AAM (Dominio Legal).\n"
        f"Aplica U(t,p) = V(t) - Cc(p) - (Pe x Ce x {LAMBDA_COEF}).\n"
        f"Audit: {json.dumps(audit, ensure_ascii=False)}\n"
        f"Matriz de Pagos: {json.dumps(pareto, indent=2)}\n\n"
        f"Regla: Si Pe>0, L3/L4 tienen U extrema negativa. Sigue la matriz.\n"
        "Termina con: NIVEL: L#"
    )
    
    clean, think = ollama_call(prompt, opts, role="Governor", prefill="Evalúo la matriz con aversión extrema:", mode="governor")
    
    llm_vote = best
    for text in [clean, think]:
        matches = re.findall(r"\bL([0-4])\b", text.upper())
        if matches:
            llm_vote = "L" + matches[-1]
            break
            
    final_vote, grd_applied, grd_reason = apply_guardrail(llm_vote, pe_count)
    return final_vote, llm_vote, grd_applied, grd_reason, think, V_task, Ce_est, Pe_base, pareto

# =====================================================================
# PROCESO BATCH
# =====================================================================
def run_trial(row, pc, t, opts):
    audit, a_think = agent_auditor(row, pc, opts["auditor"])
    g_res = agent_governor(audit, pc, opts["governor"])
    return {
        "trial": t+1, "vote": g_res[0], "llm_vote": g_res[1], 
        "guardrail_applied": g_res[2], "guardrail_reason": g_res[3],
        "audit_json": audit, "audit_think": a_think, "gov_think": g_res[4],
        "v_task": g_res[5], "ce_est": g_res[6], "pe_base": g_res[7]
    }

def precheck(row):
    issues = []
    if row.get("uncapped_liability") and row.get("cap_on_liability"): issues.append("Contradicción: Uncapped Liability y Cap On Liability.")
    if not row.get("governing_law") or row.get("governing_law") == "Not specified": issues.append("Jurisdicción no definida.")
    return {
        "issues": issues, "clean": len(issues)==0,
        "doc_name": row.get("doc_name", "Unknown"), 
        "high_risk_count": len(row.get("high_risk_clauses", [])), 
        "clauses_summary": row
    }

def consensus(votes):
    count = Counter(votes)
    max_v = max(count.values())
    candidates = [v for v, c in count.items() if c == max_v]
    return max(candidates, key=lambda x: int(x[1]))

def export_csv(all_results):
    os.makedirs("results", exist_ok=True)
    rows = []
    for r in all_results:
        for tr in r["trials"]:
            rows.append({
                "rec_num": r["rec_num"], "doc_name": r["doc_name"][:80],
                "trial": tr["trial"], "llm_vote": tr["llm_vote"], "vote": tr["vote"],
                "guardrail_applied": tr["guardrail_applied"], "guardrail_reason": tr["guardrail_reason"],
                "Pe_issues": str(tr["audit_json"].get("riesgo", {}).get("Pe", [])),
                "Ce_issues": str(tr["audit_json"].get("riesgo", {}).get("Ce", [])),
                "notas_generales": str(tr["audit_json"].get("riesgo", {}).get("notas", [])),
                "v_task": tr["v_task"], "ce_est": tr["ce_est"], "pe_base": tr["pe_base"]
            })
    pd.DataFrame(rows).to_csv("results/LEGAL_XXX_BATCH_LOG.csv", index=False)

def run(limit, trials, mode):
    if not os.path.exists(DATA_PATH): 
        print("ERROR: data/legal_clean.json no existe. Corre preprocess_legal.py")
        return
    
    with open(DATA_PATH, "r", encoding="utf-8") as f: records = json.load(f)[:limit]
    all_results = []
    progress = Progress(limit, trials)
    
    for idx, row in enumerate(records):
        pc = precheck(row)
        t0 = time.time()
        trials_data = [run_trial(row, pc, t, LLM_OPTS[mode]) for t in range(trials)]
        
        votes = [td["vote"] for td in trials_data]
        decision = consensus(votes)
        dur = time.time() - t0
        grd_count = sum(1 for td in trials_data if td["guardrail_applied"])
        
        all_results.append({
            "rec_num": idx+1, "doc_name": pc["doc_name"], "decision": decision, 
            "votes": votes, "trials": trials_data, "stability": votes.count(decision),
            "precheck_clean": pc["clean"]
        })
        
        progress.update(idx+1, decision, votes, dur, grd_count)
        if (idx+1) % 10 == 0: export_csv(all_results)
        
    progress.done_line()
    export_csv(all_results)
    print("\nBatch completado: results/LEGAL_XXX_BATCH_LOG.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    
    run(args.limit, 3 if args.full else args.trials, "full" if args.full else "batch")
