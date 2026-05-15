#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMD-AAM: Governance by Mechanism Design — Modo Batch Académico (Ethereum)
=======================================================================
Validación empírica utilizando el dataset Ethereum Fraud Detection.
Al ser un libro mayor público, los saldos (Balances) son reales y auditables.
El AAM evalúa el riesgo agregado (wallet) basado en invariantes matemáticos
y patrones de comportamiento (ej. drain patterns).
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
DATA_PATH   = "data/transaction_dataset.csv" # Nombre estándar del CSV de Kaggle
LAMBDA_COEF = 5.0 # Mantenemos aversión financiera estándar

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
CC_MAP       = {"L0": 100.0, "L1": 50.0, "L2": 15.0, "L3": 5.0,  "L4": 0.0}

class Progress:
    def __init__(self, total: int, trials: int):
        self.total   = total
        self.trials  = trials
        self.done    = 0
        self.t0      = time.time()

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
            f"\r  [{bar}] {pct:3d}% | Wallet {tx_num:>4}/{self.total} | "
            f"{decision} [{votes_s}] | {duration:.1f}s/tx | ETA {eta}   ",
            end="", flush=True
        )

    def done_line(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.t0)))
        print(f"\n  Completado en {elapsed}")

def ollama_call(prompt: str, options: dict, role: str = "", prefill: str = "", mode: str = "governor") -> tuple[str, str]:
    raw = ""
    full_prompt = prompt.rstrip() + "\n<think>\n" + prefill if prefill else prompt.rstrip() + "\n<think>"
    try:
        resp = requests.post(
            OLLAMA_URL, json={"model": MODELO, "prompt": full_prompt, "stream": True, "options": options},
            timeout=300, stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line: continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                raw  += chunk.get("response", "")
                if chunk.get("done", False): break
            except json.JSONDecodeError: continue
    except Exception as e:
        print(f"\n  [{role}] Error: {e}")
    return _parse_think(raw, mode=mode)

def _parse_think(raw: str, mode: str = "governor") -> tuple[str, str]:
    raw = raw.strip()
    if "<think>" in raw and "</think>" in raw:
        m = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
        if m: return m.group(2).strip(), m.group(1).strip()
    if "</think>" in raw and "<think>" not in raw:
        parts = raw.split("</think>", 1)
        clean = parts[1].strip() if len(parts) > 1 else ""
        if clean: return clean, parts[0].strip()
    
    if mode == "auditor":
        for pat in [r"```json", r"```\s*\{", r'\{"riesgo"', r'\{"inconsistencias"']:
            m = re.search(pat, raw, re.DOTALL)
            if m and m.start() > 20: return raw[m.start():].strip(), raw[:m.start()].strip()
            
        # WORKAROUND: Separamos las condiciones para evitar que la UI inyecte links basura
        clean_raw = raw.strip()
        if clean_raw.startswith("{") or clean_raw.startswith("```"): 
            return clean_raw, ""
            
        return raw, raw
    else:
        for pat in [r"NIVEL\s*:\s*L[0-4]", r"\*\*\s*Conclusi[oó]n", r"Conclusi[oó]n\s+[Ff]inal", r"(?:asigno|determino|elijo)\s+(?:el\s+)?nivel"]:
            matches = list(re.finditer(pat, raw, re.IGNORECASE | re.DOTALL))
            if matches:
                pos = matches[-1].start()
                if pos > 30: return raw[pos:].strip(), raw[:pos].strip()
        nivel_m = list(re.finditer(r"\bL[0-4]\b", raw.upper()))
        if nivel_m:
            last  = nivel_m[-1]
            start = raw.rfind(chr(10), 0, last.start())
            if start > 30: return raw[start:].strip(), raw[:start].strip()
        return raw, raw

# =====================================================================
# INVARIANTES BLOCKCHAIN
# =====================================================================
def precheck_ethereum(received: float, sent: float, balance: float, sent_contracts: float) -> dict:
    issues = []
    # Invariante 1: Consistencia Matemática (tolerancia para el gas fee)
    expected_balance = received - sent - sent_contracts
    if abs(expected_balance - balance) > 0.05: 
        issues.append(f"Inconsistencia matemática: Balance real ({balance:.4f} ETH) difiere del balance calculado ({expected_balance:.4f} ETH).")
    
    # Invariante 2: Sobre-giro ilógico en blockchain
    if sent > received and balance < 0:
        issues.append(f"Anomalía de flujo: Los fondos enviados ({sent:.4f} ETH) superan a los recibidos ({received:.4f} ETH) resultando en balance negativo.")
        
    # Invariante 3: Patrón de vaciado rápido (Drain pattern clásico de fraude)
    if balance <= 0.001 and received > 0 and abs(received - sent) < 0.05:
        issues.append("Patrón Drain: La cuenta fue vaciada casi a 0 ETH tras recibir los fondos.")

    return {"issues": issues, "clean": len(issues) == 0}

def agent_auditor(received, sent, balance, sent_contracts, issues, opts) -> tuple[dict, str]:
    total_volume = received + sent
    issue_block = "\nALERTA PRECHECK (INVARIANTES ROTOS):\n" + "\n".join("- " + i for i in issues) if issues else ""

    prompt = (
        "Eres el Agente Auditor del sistema GMD-AAM. Responde en español.\n"
        "Sensing Effort: analiza el historial agregado de este wallet de Ethereum para estimar Pe(t,p).\n\n"
        f"MÉTRICAS DEL WALLET:\n"
        f"  Total Recibido:       {received:.4f} ETH\n"
        f"  Total Enviado:        {sent:.4f} ETH\n"
        f"  Balance Actual:       {balance:.4f} ETH\n"
        f"  Enviado a Contratos:  {sent_contracts:.4f} ETH\n"
        f"  Volumen Transado:     {total_volume:.4f} ETH\n"
        + issue_block + "\n\n"
        "Analiza brevemente la consistencia de estos flujos contables y si hay patrones sospechosos (ej. drain o inconsistencias matemáticas).\n"
        "Responde SOLO con el JSON separando tus notas de los riesgos reales:\n"
        '{"riesgo": {"notas": ["razonamiento"], "Pe": ["inconsistencias o anomalías operativas severas"], "Ce": ["observaciones"]}}'
    )
    clean, think = ollama_call(prompt, opts, role="Auditor", prefill="Analizo los invariantes del wallet en español:", mode="auditor")

    result = {"riesgo": {"notas": [], "Pe": issues[:] if issues else [], "Ce": []}}
    for pattern in [r'```json\s*(.*?)\s*```', r'(\{[^{}]*"riesgo"[^{}]*\})', r'(\{.*\})']:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                riesgo = parsed.get("riesgo", {})
                
                ruido = ["ninguna", "ok", "correcto", "no hay", "cumple", "vacío", "normal"]
                pe = [p for p in (riesgo.get("inconsistencias") or riesgo.get("Pe") or []) if not any(r in str(p).lower() for r in ruido)]
                ce = [c for c in (riesgo.get("observaciones") or riesgo.get("Ce") or []) if not any(r in str(c).lower() for r in ruido)]
                
                result = {"riesgo": {"notas": riesgo.get("notas", []), "Pe": pe + issues, "Ce": ce}}
                break
            except json.JSONDecodeError: continue
    return result, think

def agent_governor(audit: dict, total_volume, balance, opts) -> tuple[str, str]:
    pe_count, ce_count = len(audit.get("riesgo", {}).get("Pe", [])), len(audit.get("riesgo", {}).get("Ce", []))
    
    # Adaptación Económica: V(t) es el valor de auditar este wallet (basado en volumen)
    # Ce_est es la exposición total en riesgo (el volumen que movió el wallet)
    V_task = max(round(total_volume * 0.005, 4), 1.0)
    Ce_est = max(total_volume, 1.0) 
    Pe_base = 0.0 if pe_count == 0 else (0.35 if pe_count == 1 else min(0.35 * pe_count, 0.95))

    pareto = {lv: round(V_task - CC_MAP[lv] - (Pe_base * RISK_WEIGHTS[lv] * Ce_est * LAMBDA_COEF), 2) for lv in ["L0", "L1", "L2", "L3", "L4"]}
    best = max(pareto, key=pareto.get)

    prompt = (
        "Eres el Risk Governor del sistema GMD-AAM. Responde en español.\n"
        "Aplica U(t,p) = V(t) - Cc(p) - (Pe x Ce x lambda) para elegir L0-L4 para este wallet.\n\n"
        f"MATRIZ DE PAGOS U(t,p):\n{json.dumps(pareto, indent=2)}\n\n"
        f"PARAMETROS: V(t)={V_task:.2f} ETH | Ce(t)={Ce_est:.2f} ETH | Pe={Pe_base} | lambda={LAMBDA_COEF}\n"
        "Razona brevemente por que el nivel de mayor U(t,p) es el correcto.\n"
        "Termina con exactamente: NIVEL: L#"
    )
    clean, think = ollama_call(prompt, opts, role="Governor", prefill="Evaluo la Matriz de Pagos en español:", mode="governor")

    nivel = best
    for text in [clean, think]:
        matches = re.findall(r"\bL([0-4])\b", text.upper())
        if matches:
            nivel = "L" + matches[-1]
            break
    return nivel, think

def run_trial(received, sent, balance, sent_contracts, pc, trial_num, opts) -> dict:
    audit, audit_think = agent_auditor(received, sent, balance, sent_contracts, pc["issues"], opts["auditor"])
    total_volume = received + sent
    level, gov_think   = agent_governor(audit, total_volume, balance, opts["governor"])
    return {"trial": trial_num + 1, "vote": level, "audit_json": audit, "audit_think": audit_think, "gov_think": gov_think}

def consensus(votes: list) -> str:
    count = Counter(votes)
    return max([v for v, c in count.items() if c == max(count.values())], key=lambda x: int(x[1]))

def export_csv(all_results: list):
    os.makedirs("results", exist_ok=True)
    csv_rows = []
    for res in all_results:
        for tr in res["trials"]:
            csv_rows.append({
                "wallet_id": res["wallet_id"], "total_received": res["received"], "total_sent": res["sent"], "balance": res["balance"],
                "anomaly": not res["precheck_clean"], "precheck": " | ".join(res["precheck_issues"]) if res["precheck_issues"] else "ok",
                "trial": tr["trial"], "vote": tr["vote"], "consensus": res["decision"], "duration_s": res["duration"],
                "Pe_issues": str(tr["audit_json"].get("riesgo", {}).get("Pe", [])),
                "Ce_issues": str(tr["audit_json"].get("riesgo", {}).get("Ce", [])),
                "notas_generales": str(tr["audit_json"].get("riesgo", {}).get("notas", []))
            })
    pd.DataFrame(csv_rows).to_csv("results/GMD_ETHEREUM_LOG.csv", index=False, encoding="utf-8")

def get_col_val(row, keywords):
    col = next((c for c in row.keys() if any(k in c.lower() for k in keywords)), None)
    try:
        return float(row[col]) if col and pd.notnull(row[col]) else 0.0
    except:
        return 0.0

def run(limit: int, trials: int, mode: str):
    print(f"\n{'='*60}\n  GMD-AAM BATCH: Ethereum Fraud Dataset\n{'='*60}\n")
    try:
        df = pd.read_csv(DATA_PATH).head(limit)
    except FileNotFoundError:
        print(f"ERROR: {DATA_PATH} no encontrado. Descarga el CSV de Kaggle y colócalo en /data")
        return

    all_results = []
    progress = Progress(limit, trials)

    for i, row in df.iterrows():
        wallet_id  = i + 1
        
        # Mapeo dinámico de las métricas agregadas de Ethereum
        received = get_col_val(row, ['total ether received'])
        sent = get_col_val(row, ['total ether sent', 'total ether sent contracts']) # fallback
        sent_contracts = get_col_val(row, ['total ether sent contracts', 'sent contracts'])
        balance = get_col_val(row, ['total ether balance', 'balance'])

        pc = precheck_ethereum(received, sent, balance, sent_contracts)

        t0 = time.time()
        trial_data = [run_trial(received, sent, balance, sent_contracts, pc, t, LLM_OPTS[mode]) for t in range(trials)]

        votes     = [td["vote"] for td in trial_data]
        decision  = consensus(votes)
        duration  = round(time.time() - t0, 1)

        all_results.append({
            "wallet_id": wallet_id, "received": received, "sent": sent, "balance": balance,
            "precheck_clean": pc["clean"], "precheck_issues": pc["issues"], "trials": trial_data,
            "votes": votes, "decision": decision, "duration": duration, "stability": votes.count(decision)
        })
        progress.update(wallet_id, decision, votes, duration)
        if wallet_id % 10 == 0: export_csv(all_results)

    progress.done_line()
    export_csv(all_results)
    print("Archivo generado: results/GMD_ETHEREUM_LOG.csv")

# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMD-AAM Batch — Validación académica con agentes reales (Ethereum)"
    )
    # Default locked to 50 for the benchmark
    parser.add_argument("--limit", type=int, default=50, 
                        help="Número de registros a evaluar (default: 50)")
    
    # Default locked to 3 for the academic consensus requirement
    parser.add_argument("--trials", type=int, default=3, 
                        help="Trials por registro para el consenso (default: 3)")
    
    # Default locked to batch to keep thinking logs concise and readable
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "full"],
                        help="Modo de ejecución del LLM (default: batch)")
    
    args = parser.parse_args()

    if args.trials < 1 or args.trials > 10:
        print("Error: --trials debe estar entre 1 y 10")
        sys.exit(1)

    run(limit=args.limit, trials=args.trials, mode=args.mode)