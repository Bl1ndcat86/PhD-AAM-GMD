#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMD-AAM: Dominio Legal — Adaptación al Legal-Clause-Dataset (Kaggle)
====================================================================
Versión refactorizada que lee del JSON preprocesado por preprocess_legal_kaggle.py
manteniendo arquitectura idéntica al motor CUAD original (legal_xxx_batch.py).

Pipeline:
    data/legal-clauses-csvs/*.csv
        │
        ▼  (preprocess_legal_kaggle.py — heurísticas semánticas robustas)
    data/legal_kaggle_clean.json
        │
        ▼  (este script — Auditor + Governor + apply_guardrail)
    results/LEGAL_KAGGLE_BATCH_LOG.csv
        │
        ▼  (gmd_evaluation_legal.py --log results/LEGAL_KAGGLE_BATCH_LOG.csv)
    results/legal_evaluation_report.{csv,txt}

Diferencias respecto a legal_xxx_batch.py (CUAD):
  - Cambio de ruta de entrada: data/legal_kaggle_clean.json en vez de
    data/legal_clean.json.
  - Cambio de ruta de salida: results/LEGAL_KAGGLE_BATCH_LOG.csv para
    distinguir del log de CUAD.
  - El precheck() trabaja con los flags ya producidos por el preprocess
    (uncapped_liability, cap_on_liability, governing_law, etc.), por lo
    que no se requiere shim estructural en runtime.

LAMBDA = 50.0 (alta severidad jurídica, calibración Sección 3.1.1 de la tesis).

Uso:
    python Legal-Clause-Dataset_xxx_batch.py --limit 100 --full
    python Legal-Clause-Dataset_xxx_batch.py --limit 50 --trials 3
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
DATA_PATH   = "data/legal_kaggle_clean.json"
LAMBDA_COEF = 50.0  # dominio legal — alta severidad

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
            f"\r  [{bar}] {pct:3d}% | Cláusula {num:>4}/{self.total} | "
            f"{decision} [{' '.join(votes)}]{grd} | {duration:.1f}s | ETA {eta}   ",
            end="", flush=True
        )

    def done_line(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.t0)))
        print(f"\n  Completado en {elapsed}")


# =====================================================================
# GUARDRAIL DETERMINISTA POSTDECISIONAL
# =====================================================================
def apply_guardrail(llm_vote: str, pe_count: int) -> tuple[str, bool, str]:
    """
    Capa determinística regulatoria que opera sobre el voto del Governor.
    Documentada en Sección 5.1 del Capítulo 2 de la tesis.
    """
    if pe_count >= 2:
        if llm_vote != "L0":
            return "L0", True, f"Guardrail λ=50: Pe={pe_count} graves detectadas. Forzado L0."
    elif pe_count == 1:
        if llm_vote in ["L3", "L4"]:
            return "L1", True, f"Guardrail λ=50: Pe=1 riesgo detectado. Forzado L1 mínima supervisión."
    return llm_vote, False, ""


# =====================================================================
# OLLAMA
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
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                raw += chunk.get("response", "")
                if chunk.get("done", False):
                    break
            except:
                continue
    except Exception as e:
        print(f"\n  [{role}] Error: {e}")
    return _parse_think(raw, mode=mode)


def _parse_think(raw, mode="governor"):
    raw = raw.strip()
    if "<think>" in raw and "</think>" in raw:
        m = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
        if m:
            return m.group(2).strip(), m.group(1).strip()
    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip(), parts[0].strip()
    if mode == "auditor":
        for pat in [r"```json", r'\{"riesgo"']:
            m = re.search(pat, raw, re.DOTALL)
            if m and m.start() > 20:
                return raw[m.start():].strip(), raw[:m.start()].strip()
    else:
        for pat in [r"NIVEL\s*:\s*L[0-4]", r"Conclusi[oó]n\s+[Ff]inal"]:
            matches = list(re.finditer(pat, raw, re.IGNORECASE | re.DOTALL))
            if matches:
                pos = matches[-1].start()
                if pos > 30:
                    return raw[pos:].strip(), raw[:pos].strip()
        nivel_m = list(re.finditer(r"L[0-4]", raw.upper()))
        if nivel_m:
            last = nivel_m[-1]
            start = raw.rfind(chr(10), 0, last.start())
            if start > 30:
                return raw[start:].strip(), raw[:start].strip()
    return raw, raw


# =====================================================================
# PRECHECK (dominio legal — usa flags producidos por el preprocess)
# =====================================================================
def precheck(record: dict) -> dict:
    issues = []
    if record.get("uncapped_liability") and record.get("cap_on_liability"):
        issues.append("Contradicción interna: la cláusula declara simultáneamente Uncapped Liability y Cap On Liability.")
    if not record.get("governing_law"):
        # Solo se reporta si el preprocess no detectó governing law en el texto.
        # No es un trigger sintético: refleja una ausencia real verificada.
        issues.append("Jurisdicción no detectada en el texto de la cláusula.")
    return {
        "issues":           issues,
        "clean":            len(issues) == 0,
        "doc_name":         record.get("doc_name", "Unknown"),
        "high_risk_count":  len(record.get("high_risk_clauses", [])),
        "clauses_summary":  record,
    }


# =====================================================================
# AGENTE 1: AUDITOR
# =====================================================================
def agent_auditor(record: dict, pc: dict, opts: dict) -> tuple[dict, str]:
    issue_block = "\nALERTA PRECHECK:\n" + "\n".join("- " + i for i in pc["issues"]) if pc["issues"] else ""

    flags = {
        "uncapped_liability": record.get("uncapped_liability", False),
        "cap_on_liability":   record.get("cap_on_liability", False),
        "indemnification":    record.get("indemnification", False),
        "governing_law":      record.get("governing_law", "") or "(no detectada)",
        "high_risk_clauses":  record.get("high_risk_clauses", []),
        "clause_category":    record.get("clause_category", ""),
    }

    prompt = (
        "Eres el Agente Auditor GMD-AAM (Dominio Legal). Responde siempre en español.\n"
        "Sensing Effort: analiza la cláusula para estimar Pe(t,p).\n\n"
        f"DOCUMENTO: {pc['doc_name']}\n"
        f"CATEGORÍA: {flags['clause_category']}\n"
        f"FLAGS DETECTADOS: {json.dumps({k: v for k, v in flags.items() if k != 'clause_category'}, ensure_ascii=False)}\n"
        f"TEXTO DE LA CLÁUSULA:\n{record.get('clause_text', '')[:1500]}\n"
        f"{issue_block}\n\n"
        "Analiza tres criterios:\n"
        "  1. RIESGO DE EXPOSICIÓN: ¿hay uncapped liability, indemnización amplia o cláusulas catastróficas?\n"
        "  2. AMBIGÜEDAD: ¿hay términos vagos, contradicciones internas o definiciones faltantes?\n"
        "  3. PROTECCIONES FALTANTES: ¿faltan límites, jurisdicción, plazos, notificaciones?\n\n"
        "Responde SOLO con el JSON, separando notas de riesgos reales (deja Pe/Ce vacios [] si todo esta OK):\n"
        '{"riesgo": {"notas": ["tu razonamiento general"], "inconsistencias": ["riesgo legal severo o contradiccion"], "observaciones": ["exposicion contractual moderada"]}}'
    )

    clean, think = ollama_call(prompt, opts, role="Auditor",
                               prefill="Analizo los criterios legales en español:",
                               mode="auditor")

    result = {"riesgo": {"notas": [], "Pe": pc["issues"][:], "Ce": []}}
    for pattern in [r'```json\s*(.*?)\s*```', r'(\{[^{}]*"riesgo"[^{}]*\})', r'(\{.*\})']:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                riesgo = parsed.get("riesgo", {})
                notas = riesgo.get("notas", [])
                pe = riesgo.get("inconsistencias") or riesgo.get("Pe") or []
                ce = riesgo.get("observaciones")   or riesgo.get("Ce") or []
                ruido = ["ninguna", "ninguno", "todo en orden", "ok", "correcto",
                         "no hay", "consistente", "cumple", "vacío", "n/a"]
                pe = [p for p in pe if not any(r in str(p).lower() for r in ruido)]
                ce = [c for c in ce if not any(r in str(c).lower() for r in ruido)]
                result = {"riesgo": {"notas": notas, "Pe": pe + pc["issues"], "Ce": ce}}
                break
            except json.JSONDecodeError:
                continue
    return result, think


# =====================================================================
# AGENTE 2: GOVERNOR
# =====================================================================
def agent_governor(audit: dict, pc: dict, opts: dict):
    pe_count = len(audit["riesgo"]["Pe"])
    ce_count = len(audit["riesgo"]["Ce"])

    V_task  = max(10.0 - (pc["high_risk_count"] * 2.0), 1.0)
    Ce_est  = 50.0 + (pc["high_risk_count"] * 100.0)
    Pe_base = 0.0 if pe_count == 0 else (0.35 if pe_count == 1 else min(0.35 * pe_count, 0.95))

    pareto = {
        lv: round(V_task - CC_MAP[lv] - (Pe_base * RISK_WEIGHTS[lv] * Ce_est * LAMBDA_COEF), 2)
        for lv in CC_MAP
    }
    best = max(pareto, key=pareto.get)

    prompt = (
        "Eres el Risk Governor GMD-AAM (Dominio Legal). Responde en español.\n"
        f"Aplica U(t,p) = V(t) - Cc(p) - (Pe x Ce x {LAMBDA_COEF}).\n\n"
        f"REPORTE AUDITOR: {json.dumps(audit, ensure_ascii=False)}\n\n"
        f"MATRIZ DE PAGOS U(t,p):\n{json.dumps(pareto, indent=2)}\n\n"
        f"PARAMETROS: V(t)={V_task} | Ce(t)={Ce_est} | Pe={Pe_base} | lambda={LAMBDA_COEF}\n"
        f"Inconsistencias Pe={pe_count} | Observaciones Ce={ce_count}\n\n"
        "Razona brevemente por qué el nivel de mayor U(t,p) es el correcto.\n"
        "Termina con exactamente: NIVEL: L#"
    )

    clean, think = ollama_call(prompt, opts, role="Governor",
                               prefill="Evalúo la matriz con aversión extrema:",
                               mode="governor")

    llm_vote = best
    for text in [clean, think]:
        matches = re.findall(r"\bL([0-4])\b", text.upper())
        if matches:
            llm_vote = "L" + matches[-1]
            break

    final_vote, grd_applied, grd_reason = apply_guardrail(llm_vote, pe_count)
    return final_vote, llm_vote, grd_applied, grd_reason, think, V_task, Ce_est, Pe_base, pareto


# =====================================================================
# TRIAL
# =====================================================================
def run_trial(record: dict, pc: dict, t: int, opts: dict) -> dict:
    audit, a_think = agent_auditor(record, pc, opts["auditor"])
    final_vote, llm_vote, grd_applied, grd_reason, g_think, V, Ce, Pe, pareto = agent_governor(audit, pc, opts["governor"])
    return {
        "trial":             t + 1,
        "vote":              final_vote,
        "llm_vote":          llm_vote,
        "guardrail_applied": grd_applied,
        "guardrail_reason":  grd_reason,
        "audit_json":        audit,
        "audit_think":       a_think,
        "gov_think":         g_think,
        "v_task":            V,
        "ce_est":            Ce,
        "pe_base":           Pe,
        "pareto":            pareto,
    }


def consensus(votes: list) -> str:
    count = Counter(votes)
    max_v = max(count.values())
    candidates = [v for v, c in count.items() if c == max_v]
    return max(candidates, key=lambda x: int(x[1]))


# =====================================================================
# CHECKPOINT + EXPORT
# =====================================================================
CKPT_PATH = "results/LEGAL_KAGGLE_CHECKPOINT.json"

def load_checkpoint() -> list:
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Checkpoint encontrado: {len(data)} cláusulas ya procesadas.")
        return data
    return []

def save_checkpoint(results: list):
    os.makedirs("results", exist_ok=True)
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def export_csv(all_results: list):
    os.makedirs("results", exist_ok=True)
    rows = []
    for r in all_results:
        for tr in r["trials"]:
            rows.append({
                "rec_num":           r["rec_num"],
                "doc_name":          r["doc_name"][:80],
                "clause_category":   r.get("clause_category", ""),
                "trial":             tr["trial"],
                "llm_vote":          tr["llm_vote"],
                "vote":              tr["vote"],
                "guardrail_applied": tr["guardrail_applied"],
                "guardrail_reason":  tr["guardrail_reason"],
                "Pe_issues":         str(tr["audit_json"].get("riesgo", {}).get("Pe", [])),
                "Ce_issues":         str(tr["audit_json"].get("riesgo", {}).get("Ce", [])),
                "notas_generales":   str(tr["audit_json"].get("riesgo", {}).get("notas", [])),
                "audit_think":       (tr["audit_think"] or "")[:2000],
                "gov_think":         (tr["gov_think"]   or "")[:1500],
                "v_task":            tr["v_task"],
                "ce_est":            tr["ce_est"],
                "pe_base":           tr["pe_base"],
                "consensus":         r["decision"],
                "stability":         r["stability"],
            })
    pd.DataFrame(rows).to_csv("results/LEGAL_KAGGLE_BATCH_LOG.csv", index=False, encoding="utf-8")
    pd.DataFrame([{
        "rec_num":   r["rec_num"],
        "doc_name":  r["doc_name"][:80],
        "decision":  r["decision"],
        "votes":     str(r["votes"]),
        "stability": r["stability"],
        "duration":  r.get("duration", 0),
    } for r in all_results]).to_csv("results/LEGAL_KAGGLE_BATCH_SUMMARY.csv", index=False)


def print_summary(all_results: list):
    decisions = [r["decision"] for r in all_results]
    dist      = Counter(decisions)
    unanimous = sum(1 for r in all_results if r["stability"] == len(r["trials"]))
    n_grd     = sum(1 for r in all_results
                    if any(t["guardrail_applied"] for t in r["trials"]))

    print(f"\n{'='*60}")
    print(f"  DISTRIBUCIÓN DE SOBERANÍA — Legal-Clause-Dataset (Kaggle)")
    print(f"{'='*60}")
    for lv in ["L0","L1","L2","L3","L4"]:
        n   = dist.get(lv, 0)
        bar = "█" * n if n < 40 else "█" * 40 + f"+{n-40}"
        print(f"  {lv}: {bar} {n}")
    print(f"  Consenso unánime:               {unanimous}/{len(all_results)} cláusulas")
    print(f"  Cláusulas con guardrail activo: {n_grd}/{len(all_results)}")
    print(f"{'='*60}")


# =====================================================================
# MOTOR PRINCIPAL
# =====================================================================
def run(limit: int, trials: int, mode: str, resume: bool):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    opts      = LLM_OPTS[mode]

    print(f"\n{'='*60}")
    print(f"  GMD-AAM BATCH — Legal-Clause-Dataset (Kaggle)")
    print(f"  Modelo: {MODELO} | Modo: {mode.upper()} | Trials: {trials} | Recs: {limit}")
    print(f"  Inicio: {timestamp} | Lambda: {LAMBDA_COEF} (legal)")
    print(f"{'='*60}\n")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} no existe.")
        print(f"  Ejecuta primero:")
        print(f"    python preprocess_legal_kaggle.py --csv-dir data/legal-clauses-csvs")
        sys.exit(2)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    records = records[:limit]

    all_results = load_checkpoint() if resume else []
    done_recs   = {r["rec_num"] for r in all_results}

    progress = Progress(limit, trials)

    for idx, record in enumerate(records):
        rec_num = idx + 1
        if rec_num in done_recs:
            r = all_results[rec_num - 1]
            progress.update(rec_num, r["decision"], r["votes"], r.get("duration", 0))
            continue

        pc = precheck(record)
        t0 = time.time()
        trials_data = [run_trial(record, pc, t, opts) for t in range(trials)]

        votes     = [td["vote"] for td in trials_data]
        decision  = consensus(votes)
        duration  = round(time.time() - t0, 1)
        stability = votes.count(decision)
        grd_count = sum(1 for td in trials_data if td["guardrail_applied"])

        all_results.append({
            "rec_num":         rec_num,
            "doc_name":        pc["doc_name"],
            "clause_category": record.get("clause_category", ""),
            "decision":        decision,
            "votes":           votes,
            "trials":          trials_data,
            "stability":       stability,
            "duration":        duration,
            "precheck_clean":  pc["clean"],
        })

        progress.update(rec_num, decision, votes, duration, grd_count)

        if rec_num % 10 == 0:
            save_checkpoint(all_results)
            export_csv(all_results)

    progress.done_line()
    export_csv(all_results)
    save_checkpoint(all_results)
    print_summary(all_results)

    meta = {
        "timestamp": timestamp,
        "model":     MODELO,
        "dataset":   "Legal-Clause-Dataset (Kaggle)",
        "mode":      mode,
        "trials":    trials,
        "limit":     limit,
        "lambda":    LAMBDA_COEF,
        "domain":    "legal",
    }
    with open("results/LEGAL_KAGGLE_BATCH_META.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Archivos generados:")
    print(f"    results/LEGAL_KAGGLE_BATCH_LOG.csv")
    print(f"    results/LEGAL_KAGGLE_BATCH_SUMMARY.csv")
    print(f"    results/LEGAL_KAGGLE_BATCH_META.json")
    print(f"    results/LEGAL_KAGGLE_CHECKPOINT.json (reanudar con --resume)")
    print(f"\n  Para evaluar estadísticamente:")
    print(f"    python gmd_evaluation_legal.py --log results/LEGAL_KAGGLE_BATCH_LOG.csv")
    print()


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMD-AAM Batch — Legal-Clause-Dataset (Kaggle)"
    )
    parser.add_argument("--limit",  type=int, default=100,
                        help="Número de cláusulas (default 100)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials por cláusula (default 3, recomendado para tesis)")
    parser.add_argument("--full",   action="store_true",
                        help="Modo full (num_predict=2000) en lugar de batch (500)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde checkpoint si existe")
    args = parser.parse_args()

    if args.full:
        mode = "full"
        if args.trials < 3:
            args.trials = 3
    else:
        mode = "batch"

    if args.trials < 1 or args.trials > 10:
        print("Error: --trials debe estar entre 1 y 10")
        sys.exit(1)

    run(limit=args.limit, trials=args.trials, mode=mode, resume=args.resume)
