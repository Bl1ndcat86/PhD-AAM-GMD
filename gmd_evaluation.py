#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmd_evaluation.py
=================
Módulo de Evaluación Estadística del Mecanismo AAM (Autonomous Allocation Mechanism).

Lee el log producido por el motor GMD-AAM (results/GMD_LOG.csv) y genera dos
artefactos: un reporte tabular en CSV (results/evaluation_report.csv) y un
informe ejecutivo en texto plano (results/evaluation_report.txt) con las
pruebas inferenciales correspondientes a las cuatro hipótesis doctorales.

Plan estadístico pre-registrado:
  H1: AAM > Laissez-faire (forzar L4 en cada tx)   →  Wilcoxon de signos pareados
  H2: AAM > HITL universal  (forzar L0 en cada tx) →  Wilcoxon de signos pareados
  H3: Criticidad ↔ Lx asignado                     →  χ² + V de Cramér
  H4: Estabilidad inter-trial ≥ 70 % unanimidad    →  proporción + Fleiss κ

Uso:
    python gmd_evaluation.py --log results/GMD_LOG.csv --lambda 5
    python gmd_evaluation.py --log results/GMD_LOG.csv --lambda 50

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
DEFAULT_LOG = "results/GMD_LOG.csv"
DEFAULT_OUT_CSV = "results/evaluation_report.csv"
DEFAULT_OUT_TXT = "results/evaluation_report.txt"

# Aversión al riesgo. λ=5 línea base (eficiencia operativa);
# λ=50 escenario de alta severidad (gobernanza estricta).
LAMBDA_DEFAULT = 5.0

# Costo de control Cc(p) y multiplicador de exposición por nivel,
# replicando exactamente la pareto_matrix del motor GMD-AAM.
LEVEL_COSTS = {
    "L0": (100.0, 0.00),
    "L1": ( 50.0, 0.05),
    "L2": ( 15.0, 0.30),
    "L3": (  5.0, 0.80),
    "L4": (  0.0, 1.00),
}

ALPHA = 0.05            # nivel de significación
EFFECT_R_MIN = 0.30     # tamaño de efecto mínimo relevante (Wilcoxon)
KAPPA_MIN = 0.60        # umbral de acuerdo sustancial
UNANIMITY_MIN = 0.70    # umbral establecido para H4


# ------------------------------------------------------------------------- #
# Reconstrucción de la función de utilidad U(t,p)                           #
# ------------------------------------------------------------------------- #
def reconstruct_pareto(amount: float, pe_count: int,
                       lam: float = LAMBDA_DEFAULT) -> dict:
    """
    Reconstruye la matriz de Pareto U(t,p) por nivel.
    Replica fielmente la lógica del motor GMD-AAM:
        V(t) = max(amount * 0.005, 1.0)
        Ce(t) = amount
        Pe(t,p) = 0      si pe_count == 0
                  0.35   si pe_count == 1
                  min(0.35*pe_count, 0.95)  si pe_count >= 2
        U(p) = V - Cc(p) - (Pe * mult(p) * Ce * λ)
    """
    V = max(round(amount * 0.005, 2), 1.0)
    Ce = amount
    if pe_count == 0:
        Pe = 0.0
    elif pe_count == 1:
        Pe = 0.35
    else:
        Pe = min(0.35 * pe_count, 0.95)
    return {
        lvl: round(V - cc - (Pe * mult * Ce * lam), 2)
        for lvl, (cc, mult) in LEVEL_COSTS.items()
    }


def parse_pe_list(s) -> list:
    """Parsea la columna inconsistencias_matematicas (string-list)."""
    if pd.isna(s) or s == "" or s == "[]":
        return []
    if isinstance(s, list):
        return s
    try:
        return list(eval(s))
    except Exception:
        return []


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

    # Tamaño de efecto r = |Z| / sqrt(N), con Z aproximado desde p.
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


def compute_h1_h2(df_summary: pd.DataFrame) -> dict:
    """Compara U_aam contra los baselines U_laissez (L4) y U_hitl (L0)."""
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
    df = df_summary.copy()
    try:
        # First try to create equal-sized quantiles (terciles)
        df["criticidad"] = pd.qcut(
            df["amount"], q=3, labels=["low", "mid", "high"], duplicates="drop"
        )
    except ValueError:
        # If there are too many duplicates to form quantiles, fall back to equal-width bins
        df["criticidad"] = pd.cut(
            df["amount"], bins=3, labels=["low", "mid", "high"]
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
            "decision": "rechazar H0" if p < ALPHA else "no rechazar H0",
        }
    }


# ------------------------------------------------------------------------- #
# H4 — Estabilidad inter-trial (Fleiss κ)                                   #
# ------------------------------------------------------------------------- #
def fleiss_kappa(matrix: np.ndarray) -> float:
    """Cálculo manual del κ de Fleiss para múltiples raters por sujeto."""
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


def compute_h4(df_log: pd.DataFrame) -> dict:
    LEVELS = ["L0", "L1", "L2", "L3", "L4"]
    # Para el dominio financiero, la columna identificadora por defecto es tx_num.
    # NOTA: si usas este script para Ethereum y la columna es wallet_id,
    # el orquestador principal debe renombrarla a tx_num antes de llamar a las funciones.
    id_col = "tx_num"
    grouped = df_log.groupby(id_col)["vote"].apply(list)
    n_total = len(grouped)
    if n_total == 0:
        return {"H4": {"error": "sin datos"}}

    unanimous = 0
    rows = []
    for tx, votes in grouped.items():
        if len(set(votes)) == 1:
            unanimous += 1
        rows.append([votes.count(lvl) for lvl in LEVELS])
    matrix = np.asarray(rows)

    rate = unanimous / n_total
    kappa = fleiss_kappa(matrix)
    return {
        "H4": {
            "n_transacciones": int(n_total),
            "n_consenso_unanime": int(unanimous),
            "tasa_unanime": float(rate),
            "umbral_minimo": UNANIMITY_MIN,
            "fleiss_kappa": float(kappa),
            "interpretacion_kappa": kappa_interpretation(kappa),
            "decision_70pct": (
                "se cumple" if rate >= UNANIMITY_MIN else "no se cumple"
            ),
            "decision_kappa": (
                "se cumple" if (not np.isnan(kappa) and kappa >= KAPPA_MIN)
                else "no se cumple"
            ),
        }
    }


# ------------------------------------------------------------------------- #
# Orquestador                                                               #
# ------------------------------------------------------------------------- #
def main(log_path: str, out_csv: str, out_txt: str, lam: float) -> dict:
    df = pd.read_csv(log_path)
    
    # Manejo de la columna ID para Ethereum (wallet_id) vs PaySim (tx_num)
    if "wallet_id" in df.columns:
        df.rename(columns={"wallet_id": "tx_num"}, inplace=True)
    
    # Validar columnas
    required = {"tx_num", "trial", "vote"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: faltan columnas en el log: {missing}")
        sys.exit(2)

    summary_rows = []
    for tx_num, group in df.groupby("tx_num"):
        # Manejo de 'amount' vs métricas de Ethereum
        if "amount" in group.columns:
            amount = float(group["amount"].iloc[0])
        elif "total_received" in group.columns:
            amount = float(group["total_received"].iloc[0])
        else:
            amount = 0.0
            
        consensus = str(group["consensus"].iloc[0])
        pe_counts = []
        for _, r in group.iterrows():
            # Ethereum no tiene inconsistencias_matematicas, usar Pe_issues si existe
            if "inconsistencias_matematicas" in r:
                pe_counts.append(len(parse_pe_list(r.get("inconsistencias_matematicas"))))
            elif "Pe_issues" in r:
                pe_counts.append(len(parse_pe_list(r.get("Pe_issues"))))
            else:
                pe_counts.append(0)
                
        pe_count_avg = int(round(float(np.mean(pe_counts)))) if pe_counts else 0
        pareto = reconstruct_pareto(amount, pe_count_avg, lam)
        summary_rows.append({
            "tx_num": int(tx_num),
            "amount": amount,
            "consensus": consensus,
            "pe_count": pe_count_avg,
            "pareto": pareto,
        })
    df_summary = pd.DataFrame(summary_rows)

    h12 = compute_h1_h2(df_summary)
    h3 = compute_h3(df_summary)
    h4 = compute_h4(df)

    report = {
        "lambda": lam,
        "alpha": ALPHA,
        "n_transacciones": len(df_summary),
        "n_trials_total": len(df),
        **h12,
        **h3,
        **h4,
    }

    # CSV largo (formato tesis)
    rows = []
    for hyp, vals in [
        ("H1", h12["H1"]),
        ("H2", h12["H2"]),
        ("H3", h3["H3"]),
        ("H4", h4["H4"]),
    ]:
        for k, v in vals.items():
            rows.append({"hipotesis": hyp, "metrica": k, "valor": str(v)})
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

    # Informe TXT
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"GMD-AAM | Reporte de Evaluacion Estadistica\n")
        f.write(f"lambda = {lam} | alpha = {ALPHA}\n")
        f.write(f"n_transacciones = {len(df_summary)} | n_trials = {len(df)}\n")
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
        description="Evaluación estadística del mecanismo AAM (H1-H4)."
    )
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help="ruta al CSV producido por el motor GMD-AAM")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV,
                        help="ruta de salida del reporte tabular")
    parser.add_argument("--out-txt", default=DEFAULT_OUT_TXT,
                        help="ruta de salida del informe ejecutivo")
    parser.add_argument("--lambda", dest="lam", type=float,
                        default=LAMBDA_DEFAULT,
                        help="coeficiente de aversion al riesgo (5 o 50)")
    args = parser.parse_args()
    sys.exit(0 if main(args.log, args.out_csv, args.out_txt, args.lam) else 1)