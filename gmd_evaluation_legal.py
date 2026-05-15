#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmd_evaluation_legal.py
=======================
Módulo de Evaluación Estadística del Mecanismo AAM — Variante Dominio Legal.

Lee el log producido por el motor GMD-AAM Legal (results/LEGAL_XXX_BATCH_LOG.csv)
y genera dos artefactos: un reporte tabular en CSV y un informe ejecutivo en
texto plano con las pruebas inferenciales correspondientes a las cuatro
hipótesis doctorales H1-H4.

Diferencias clave respecto a la versión financiera (gmd_evaluation.py):
  1. λ = 50 (alta severidad jurídica, calibración Sección 3.1.1 de la tesis).
  2. V(t) y Ce(t) NO se reconstruyen desde amount: se leen directamente del
     log (columnas v_task y ce_est ya persistidas por el motor legal). La
     ontología contractual no admite la fórmula amount-based del dominio
     financiero.
  3. Distinción operativa entre voto crudo del LLM (llm_vote) y voto final
     post-guardrail (vote). Ambos se reportan: el primero mide la cualidad
     intrínseca del LLM, el segundo la del AAM completo.
  4. Métrica adicional: tasa de activación del Guardrail Determinista
     Postdecisional (apply_guardrail), reportada como evidencia de
     intervención regulatoria.

Plan estadístico pre-registrado (idéntico al financiero):
  H1: AAM > Laissez-faire (forzar L4 en cada contrato)  →  Wilcoxon pareado
  H2: AAM > HITL universal  (forzar L0 en cada contrato) →  Wilcoxon pareado
  H3: Criticidad ↔ Lx asignado                          →  χ² + V de Cramér
  H4: Estabilidad inter-trial ≥ 70 % unanimidad         →  proporción + Fleiss κ
       (calculada sobre llm_vote y sobre vote final)

Uso:
    python gmd_evaluation_legal.py --log results/LEGAL_XXX_BATCH_LOG.csv
    python gmd_evaluation_legal.py --log results/LEGAL_XXX_BATCH_LOG.csv --lambda 50

Autor: Guillermo Vega Quesada
Tesis Doctoral — Universidad Fidélitas (2023-2026)
"""

import argparse
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------- #
# Configuración por defecto                                                 #
# ------------------------------------------------------------------------- #
DEFAULT_LOG = "results/LEGAL_XXX_BATCH_LOG.csv"
DEFAULT_OUT_CSV = "results/legal_evaluation_report.csv"
DEFAULT_OUT_TXT = "results/legal_evaluation_report.txt"

# λ = 50: alta severidad jurídica (Sección 3.1.1 tesis)
LAMBDA_DEFAULT = 50.0

# Costo de control Cc(p) y multiplicador de exposición por nivel,
# replicando exactamente la pareto_matrix del motor legal_xxx_batch.py.
LEVEL_COSTS = {
    "L0": (100.0, 0.00),
    "L1": ( 50.0, 0.05),
    "L2": ( 15.0, 0.30),
    "L3": (  5.0, 0.80),
    "L4": (  0.0, 1.00),
}

ALPHA = 0.05            # nivel de significación
EFFECT_R_MIN = 0.30     # tamaño de efecto mínimo relevante (Wilcoxon)
KAPPA_MIN = 0.60        # umbral de acuerdo sustancial (Landis y Koch, 1977)
UNANIMITY_MIN = 0.70    # umbral establecido para H4
CRAMER_MIN = 0.20       # umbral de equilibrio de separación (H3)


# ------------------------------------------------------------------------- #
# Reconstrucción de la función de utilidad U(t,p) — dominio legal          #
# ------------------------------------------------------------------------- #
def pareto_legal(v_task: float, ce_est: float, pe_base: float,
                 lam: float = LAMBDA_DEFAULT) -> dict:
    """
    Calcula la matriz de Pareto U(t,p) por nivel para el dominio legal,
    usando V(t) y Ce(t) ya persistidos en el log por el motor.

    Replica fielmente la lógica del motor legal_xxx_batch.py:
        U(p) = V_task − Cc(p) − (Pe_base × mult(p) × Ce_est × λ)
    """
    return {
        lvl: round(v_task - cc - (pe_base * mult * ce_est * lam), 2)
        for lvl, (cc, mult) in LEVEL_COSTS.items()
    }


# ------------------------------------------------------------------------- #
# H1 y H2 — Wilcoxon pareado                                                #
# ------------------------------------------------------------------------- #
def wilcoxon_test(deltas: pd.Series, hyp_name: str) -> dict:
    nonzero = deltas[deltas != 0]
    n_paired = len(nonzero)
    if n_paired < 1:
        return {
            "delta_mean": float(deltas.mean()) if len(deltas) else np.nan,
            "delta_median": float(deltas.median()) if len(deltas) else np.nan,
            "n_paired": n_paired,
            "wilcoxon_W": np.nan,
            "p_value": np.nan,
            "effect_r": np.nan,
            "decision": "datos insuficientes",
        }
    try:
        res = stats.wilcoxon(nonzero, alternative="greater",
                             zero_method="wilcox", correction=False,
                             mode="auto")
        W, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        W, p = np.nan, np.nan

    if not np.isnan(p) and 0 < p < 1:
        z = stats.norm.ppf(1 - p)
        r = float(abs(z) / np.sqrt(n_paired))
    else:
        r = np.nan

    decision = "rechazar H0" if (not np.isnan(p) and p < ALPHA) else "no rechazar H0"
    return {
        "delta_mean": float(deltas.mean()),
        "delta_median": float(deltas.median()),
        "n_paired": n_paired,
        "wilcoxon_W": W,
        "p_value": p,
        "effect_r": r,
        "effect_relevante": (
            "sí" if (not np.isnan(r) and r >= EFFECT_R_MIN) else "no"
        ),
        "decision": decision,
    }


def compute_h1_h2(df_summary: pd.DataFrame, lam: float) -> dict:
    """Compara U_aam contra los baselines U_laissez (L4) y U_hitl (L0)."""
    df_summary["pareto"] = df_summary.apply(
        lambda r: pareto_legal(r["v_task"], r["ce_est"], r["pe_base"], lam), axis=1
    )
    df_summary["U_aam"] = df_summary.apply(
        lambda r: r["pareto"][r["consensus"]], axis=1
    )
    df_summary["U_laissez"] = df_summary["pareto"].apply(lambda d: d["L4"])
    df_summary["U_hitl"] = df_summary["pareto"].apply(lambda d: d["L0"])
    df_summary["delta_h1"] = df_summary["U_aam"] - df_summary["U_laissez"]
    df_summary["delta_h2"] = df_summary["U_aam"] - df_summary["U_hitl"]
    return {
        "H1": wilcoxon_test(df_summary["delta_h1"], "H1"),
        "H2": wilcoxon_test(df_summary["delta_h2"], "H2"),
    }


# ------------------------------------------------------------------------- #
# H3 — Chi-cuadrado de criticidad × Lx                                      #
# ------------------------------------------------------------------------- #
def compute_h3(df_summary: pd.DataFrame) -> dict:
    """
    Criticidad operacionalizada como terciles del Costo de Error Ce(t),
    que en el dominio legal codifica linealmente la severidad jurídica
    (Ce_est = 50 + 100 × high_risk_count).
    """
    df = df_summary.copy()
    
    try:
        # First try to create equal-sized quantiles (terciles)
        df["criticidad"] = pd.qcut(
            df["ce_est"], q=3, labels=["low", "mid", "high"], duplicates="drop"
        )
    except ValueError:
        # If there are too many duplicates to form quantiles, fall back to equal-width bins
        df["criticidad"] = pd.cut(
            df["ce_est"], bins=3, labels=["low", "mid", "high"]
        )

    table = pd.crosstab(df["criticidad"], df["consensus"])
    if table.size == 0 or table.values.sum() == 0:
        return {"H3": {"error": "tabla vacía"}}

    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    min_dim = min(table.shape) - 1
    cramer_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else np.nan

    return {
        "H3": {
            "tabla_contingencia": table.to_dict(),
            "n_total": n,
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "cramer_v": cramer_v,
            "umbral_cramer": CRAMER_MIN,
            "decision": "rechazar H0" if p < ALPHA else "no rechazar H0",
            "evidencia_separation_eq": (
                "sí" if (not np.isnan(cramer_v) and cramer_v >= CRAMER_MIN)
                else "no"
            ),
        }
    }


# ------------------------------------------------------------------------- #
# H4 — Estabilidad inter-trial (Fleiss κ)                                   #
# ------------------------------------------------------------------------- #
def fleiss_kappa(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return float("nan")
    N, k = matrix.shape
    n = float(matrix.sum(axis=1)[0])
    if n <= 1 or N == 0:
        return float("nan")
    p_j = matrix.sum(axis=0) / (N * n)
    P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = float(P_i.mean())
    P_e = float(np.sum(p_j ** 2))
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def kappa_interpretation(kappa: float) -> str:
    if np.isnan(kappa):
        return "no calculable"
    if kappa < 0.0:
        return "acuerdo peor que el azar"
    if kappa < 0.20:
        return "acuerdo escaso"
    if kappa < 0.40:
        return "acuerdo aceptable"
    if kappa < 0.60:
        return "acuerdo moderado"
    if kappa < 0.80:
        return "acuerdo sustancial"
    return "acuerdo casi perfecto"


def stability_block(df_log: pd.DataFrame, vote_col: str) -> dict:
    """Calcula H4 sobre una columna de voto específica (llm_vote o vote)."""
    LEVELS = ["L0", "L1", "L2", "L3", "L4"]
    grouped = df_log.groupby("rec_num")[vote_col].apply(list)
    n_total = len(grouped)
    if n_total == 0:
        return {"error": "sin datos"}
    unanimous = 0
    rows = []
    for rec, votes in grouped.items():
        if len(set(votes)) == 1:
            unanimous += 1
        rows.append([votes.count(lvl) for lvl in LEVELS])
    matrix = np.asarray(rows)
    rate = unanimous / n_total
    kappa = fleiss_kappa(matrix)
    return {
        "n_contratos": int(n_total),
        "n_consenso_unanime": int(unanimous),
        "tasa_unanime": float(rate),
        "umbral_minimo": UNANIMITY_MIN,
        "fleiss_kappa": float(kappa),
        "interpretacion_kappa": kappa_interpretation(kappa),
        "decision_70pct": "se cumple" if rate >= UNANIMITY_MIN else "no se cumple",
        "decision_kappa": (
            "se cumple" if (not np.isnan(kappa) and kappa >= KAPPA_MIN)
            else "no se cumple"
        ),
    }


def compute_h4(df_log: pd.DataFrame) -> dict:
    """
    Calcula estabilidad inter-trial en dos niveles:
      - LLM crudo (llm_vote): cualidad intrínseca del modelo de lenguaje.
      - AAM final (vote): cualidad operativa del mecanismo completo
        (incluye intervención del guardrail postdecisional).
    """
    h4 = {}
    if "llm_vote" in df_log.columns:
        h4["sobre_llm_vote_crudo"] = stability_block(df_log, "llm_vote")
    h4["sobre_vote_final"] = stability_block(df_log, "vote")
    return {"H4": h4}


# ------------------------------------------------------------------------- #
# Métrica adicional: activación del guardrail postdecisional               #
# ------------------------------------------------------------------------- #
def compute_guardrail_metric(df_log: pd.DataFrame) -> dict:
    """
    Reporta la frecuencia y razones de activación del guardrail
    determinista postdecisional (apply_guardrail) en el dominio legal.
    """
    if "guardrail_applied" not in df_log.columns:
        return {"guardrail": {"error": "columna no presente"}}

    df_log = df_log.copy()
    df_log["guardrail_applied"] = df_log["guardrail_applied"].astype(bool)
    n_total = len(df_log)
    n_activado = int(df_log["guardrail_applied"].sum())
    rate = n_activado / n_total if n_total else 0

    # Por contrato: ¿se activó al menos un trial?
    by_rec = df_log.groupby("rec_num")["guardrail_applied"].any()
    n_rec_activado = int(by_rec.sum())
    n_rec_total = len(by_rec)

    # Override: ¿en cuántos casos vote != llm_vote?
    if "llm_vote" in df_log.columns and "vote" in df_log.columns:
        df_log["override"] = df_log["llm_vote"] != df_log["vote"]
        n_override = int(df_log["override"].sum())
    else:
        n_override = None

    return {
        "guardrail": {
            "n_trials_total": n_total,
            "n_trials_activado": n_activado,
            "tasa_activacion_trial": float(rate),
            "n_contratos_total": n_rec_total,
            "n_contratos_con_activacion": n_rec_activado,
            "tasa_contratos_con_activacion": (
                n_rec_activado / n_rec_total if n_rec_total else 0
            ),
            "n_override_llm": n_override,
            "interpretacion": (
                "El guardrail postdecisional opera como circuit-breaker "
                "regulatorio. Una tasa de activación elevada indica que el "
                "Auditor detectó riesgo grave y el LLM no escaló adecuadamente; "
                "una tasa baja indica que el Governor por sí solo ya converge "
                "a niveles conservadores ante señales de Pe."
            ),
        }
    }


# ------------------------------------------------------------------------- #
# Orquestador                                                               #
# ------------------------------------------------------------------------- #
def main(log_path: str, out_csv: str, out_txt: str, lam: float) -> dict:
    df = pd.read_csv(log_path)

    # Validar columnas requeridas
    required = {"rec_num", "trial", "vote", "v_task", "ce_est", "pe_base"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: faltan columnas en el log: {missing}")
        sys.exit(2)

    # Construir summary una fila por contrato
    summary_rows = []
    for rec_num, group in df.groupby("rec_num"):
        votes = list(group["vote"])
        cnt = Counter(votes)
        consensus = max(cnt.items(), key=lambda x: (x[1], int(x[0][1])))[0]
        # Promedio de los parámetros entre trials (deberían ser idénticos)
        v_task = float(group["v_task"].mean())
        ce_est = float(group["ce_est"].mean())
        pe_base = float(group["pe_base"].mean())
        summary_rows.append({
            "rec_num": int(rec_num),
            "consensus": consensus,
            "v_task": v_task,
            "ce_est": ce_est,
            "pe_base": pe_base,
        })
    df_summary = pd.DataFrame(summary_rows)

    h12 = compute_h1_h2(df_summary, lam)
    h3 = compute_h3(df_summary)
    h4 = compute_h4(df)
    grd = compute_guardrail_metric(df)

    report = {
        "dominio": "legal",
        "lambda": lam,
        "alpha": ALPHA,
        "n_contratos": len(df_summary),
        "n_trials_total": len(df),
        **h12,
        **h3,
        **h4,
        **grd,
    }

    # CSV largo (formato tesis)
    rows = []
    for hyp, vals in [
        ("H1", h12["H1"]),
        ("H2", h12["H2"]),
        ("H3", h3["H3"]),
    ]:
        for k, v in vals.items():
            rows.append({"hipotesis": hyp, "metrica": k, "valor": str(v)})
    for sub_label, sub_vals in h4["H4"].items():
        for k, v in sub_vals.items():
            rows.append({"hipotesis": f"H4_{sub_label}", "metrica": k, "valor": str(v)})
    for k, v in grd["guardrail"].items():
        rows.append({"hipotesis": "guardrail", "metrica": k, "valor": str(v)})
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    # Informe TXT
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("GMD-AAM | Reporte de Evaluacion Estadistica — DOMINIO LEGAL\n")
        f.write(f"lambda = {lam} | alpha = {ALPHA}\n")
        f.write(f"n_contratos = {len(df_summary)} | n_trials = {len(df)}\n")
        f.write("=" * 70 + "\n\n")
        f.write(json.dumps(report, indent=2, default=str, ensure_ascii=False))

    print(f"\n[OK] Reportes guardados en:\n  {out_csv}\n  {out_txt}\n")
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    return report


# ------------------------------------------------------------------------- #
# CLI                                                                       #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación estadística del mecanismo AAM — dominio legal."
    )
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help="ruta al CSV producido por legal_xxx_batch.py")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV,
                        help="ruta de salida del reporte tabular")
    parser.add_argument("--out-txt", default=DEFAULT_OUT_TXT,
                        help="ruta de salida del informe ejecutivo")
    parser.add_argument("--lambda", dest="lam", type=float,
                        default=LAMBDA_DEFAULT,
                        help="coeficiente de aversion al riesgo (default 50)")
    args = parser.parse_args()
    sys.exit(0 if main(args.log, args.out_csv, args.out_txt, args.lam) else 1)