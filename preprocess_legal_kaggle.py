#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_legal_kaggle.py
==========================
Normaliza los CSV del Legal-Clause-Dataset (Kaggle) al formato JSON tipo CUAD
que consume el motor GMD-AAM legal.

Entrada esperada:
  data/legal-clauses-csvs/*.csv
  (cualquier archivo del dataset Kaggle, p. ej., limitation-of-liability.csv,
   indemnification.csv, governing-law.csv, etc. Cada CSV tiene una columna
   con el texto de la cláusula y, opcionalmente, una columna con el nombre
   del documento.)

Salida:
  data/legal_kaggle_clean.json
  Lista de records con la estructura:
      {
        "doc_name":           str,
        "clause_text":        str,
        "clause_category":    str,    # derivado del nombre del CSV
        "high_risk_clauses":  [str],  # detectadas por heurística semántica
        "uncapped_liability": bool,
        "cap_on_liability":   bool,
        "indemnification":    bool,
        "governing_law":      str,    # texto detectado o "" si no aparece
      }

Uso:
    python preprocess_legal_kaggle.py
    python preprocess_legal_kaggle.py --csv-dir data/legal-clauses-csvs --out data/legal_kaggle_clean.json
    python preprocess_legal_kaggle.py --files data/limitation-of-liability.csv data/indemnification.csv

Autor: Guillermo Vega Quesada
Tesis Doctoral — Universidad Fidélitas (2023-2026)
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

# --------------------------------------------------------------------- #
# Heurísticas semánticas multi-keyword                                  #
# --------------------------------------------------------------------- #
PATTERNS_UNCAPPED = [
    r"\buncapped\b",
    r"without\s+(?:any\s+)?(?:monetary\s+)?(?:limit|cap|maximum)",
    r"no\s+(?:cap|limit|maximum)\s+on\s+(?:the\s+)?liabilit",
    r"unlimited\s+liabilit",
    r"shall\s+(?:not\s+be\s+)?limited\s+to",  # frase abierta
]

PATTERNS_CAPPED = [
    r"\bcap(?:ped)?\s+(?:on|at|of)\s+(?:liabilit|damages|amount)",
    r"maximum\s+(?:aggregate\s+)?liabilit",
    r"shall\s+(?:not\s+exceed|be\s+limited\s+to)",
    r"limited\s+to\s+(?:the\s+)?(?:amount|sum|fees|total)",
    r"liabilit[a-z]+\s+(?:shall\s+(?:not\s+)?exceed|is\s+limited)",
]

PATTERNS_INDEMNIFICATION = [
    r"\bindemnif",
    r"hold\s+harmless",
    r"defend(?:\s+and\s+indemnify)?",
]

PATTERNS_GOVERNING_LAW = [
    r"governed\s+by\s+(?:and\s+construed\s+(?:in\s+accordance\s+)?(?:with\s+)?)?the\s+laws?\s+of\s+([A-Z][A-Za-z .,&-]+)",
    r"laws?\s+of\s+(?:the\s+)?(?:State|Commonwealth|Province|Republic)\s+of\s+([A-Z][A-Za-z .,&-]+)",
    r"jurisdiction\s+of\s+(?:the\s+)?courts?\s+of\s+([A-Z][A-Za-z .,&-]+)",
]

CATEGORY_TO_RISK_MAP = {
    "limitation-of-liability":  "Limitation of Liability",
    "indemnification":          "Indemnification",
    "warranties":               "Warranties",
    "termination":              "Termination",
    "non-compete":              "Non-Compete",
    "exclusivity":              "Exclusivity",
    "uncapped-liability":       "Uncapped Liability",
    "ip-ownership-assignment":  "IP Ownership Assignment",
    "covenant-not-to-sue":      "Covenant Not To Sue",
    "audit-rights":             "Audit Rights",
}


def detect_pattern(text: str, patterns: list) -> bool:
    txt = text.lower()
    for p in patterns:
        if re.search(p, txt, re.IGNORECASE):
            return True
    return False


def extract_governing_law(text: str) -> str:
    for p in PATTERNS_GOVERNING_LAW:
        m = re.search(p, text, re.IGNORECASE)
        if m and m.lastindex:
            return m.group(1).strip().rstrip(".,;").strip()
    return ""


def category_from_filename(filename: str) -> str:
    base = os.path.basename(filename).lower().replace(".csv", "")
    base = re.sub(r"[\s_]+", "-", base)
    return base


def find_text_column(df: pd.DataFrame) -> str:
    """Busca dinámicamente la columna que contiene el texto de la cláusula."""
    candidates = []
    for col in df.columns:
        col_low = col.lower()
        if any(k in col_low for k in ("text", "clause", "provision", "content", "body")):
            candidates.append(col)
    if candidates:
        return candidates[0]

    # Fallback: la columna con strings más largos en promedio
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) == 0:
        raise ValueError("El CSV no tiene columnas de texto.")
    avg_len = {c: df[c].astype(str).str.len().mean() for c in str_cols}
    return max(avg_len, key=avg_len.get)


def find_doc_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        col_low = col.lower()
        if any(k in col_low for k in ("doc", "contract", "document", "filename", "agreement")):
            return col
    return None


# --------------------------------------------------------------------- #
# Procesamiento principal                                               #
# --------------------------------------------------------------------- #
def process_csv(path: str, max_rows: int = None) -> list:
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)

    txt_col = find_text_column(df)
    doc_col = find_doc_column(df)
    category = category_from_filename(path)
    risk_label = CATEGORY_TO_RISK_MAP.get(category, category.replace("-", " ").title())

    records = []
    for idx, row in df.iterrows():
        clause_text = str(row[txt_col]).strip()
        if not clause_text or clause_text.lower() in ("nan", "none", ""):
            continue

        doc_name = (
            str(row[doc_col]).strip() if doc_col and pd.notna(row[doc_col])
            else f"{category}_{idx + 1}"
        )

        uncapped = detect_pattern(clause_text, PATTERNS_UNCAPPED)
        capped   = detect_pattern(clause_text, PATTERNS_CAPPED)
        indemn   = detect_pattern(clause_text, PATTERNS_INDEMNIFICATION)
        gov_law  = extract_governing_law(clause_text)

        # high_risk_clauses: lista no vacía si la cláusula califica como alto
        # riesgo según su categoría o si detectamos uncapped_liability/indemnification.
        high_risk = []
        if uncapped:
            high_risk.append("Uncapped Liability")
        if indemn:
            high_risk.append("Indemnification")


        records.append({
            "doc_name":           doc_name,
            "clause_text":        clause_text,
            "clause_category":    category,
            "high_risk_clauses":  high_risk,
            "uncapped_liability": bool(uncapped),
            "cap_on_liability":   bool(capped),
            "indemnification":    bool(indemn),
            "governing_law":      gov_law,
        })

    return records


def main(csv_dir: str, files: list, out: str, max_rows: int):
    sources = []
    if files:
        sources = files
    elif csv_dir:
        sources = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    else:
        print("ERROR: especifica --csv-dir o --files")
        sys.exit(1)

    if not sources:
        print(f"ERROR: no se encontraron archivos CSV.")
        sys.exit(1)

    print(f"Procesando {len(sources)} archivo(s)...")
    all_records = []
    for src in sources:
        print(f"  - {src}", end="")
        try:
            recs = process_csv(src, max_rows=max_rows)
            all_records.extend(recs)
            print(f" → {len(recs)} cláusulas")
        except Exception as e:
            print(f" → ERROR: {e}")

    # Persiste el JSON unificado
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # Estadísticas de control
    cats     = {}
    n_uncap  = sum(1 for r in all_records if r["uncapped_liability"])
    n_cap    = sum(1 for r in all_records if r["cap_on_liability"])
    n_indem  = sum(1 for r in all_records if r["indemnification"])
    n_govern = sum(1 for r in all_records if r["governing_law"])
    for r in all_records:
        cats[r["clause_category"]] = cats.get(r["clause_category"], 0) + 1

    print(f"\nResumen:")
    print(f"  Total de cláusulas: {len(all_records)}")
    print(f"  Por categoría:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n}")
    print(f"  Uncapped Liability detectado: {n_uncap}")
    print(f"  Cap on Liability detectado:   {n_cap}")
    print(f"  Indemnification detectado:    {n_indem}")
    print(f"  Governing Law detectado:      {n_govern}")
    print(f"\nSalida: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesa CSVs del Legal-Clause-Dataset (Kaggle) a JSON tipo CUAD."
    )
    parser.add_argument("--csv-dir", default="data/legal-clauses-csvs",
                        help="Directorio con los CSV de Kaggle")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Archivos CSV específicos (en lugar de --csv-dir)")
    parser.add_argument("--out", default="data/legal_kaggle_clean.json",
                        help="Ruta de salida del JSON unificado")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limitar a N cláusulas por CSV (útil para pruebas)")
    args = parser.parse_args()
    main(args.csv_dir, args.files, args.out, args.max_rows)
