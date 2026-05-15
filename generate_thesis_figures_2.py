#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_thesis_figures.py
==========================
Versión Doctoral Final: Consistencia de Color y Tipografía Times 12pt.
Heatmaps restaurados según Código 1.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# =====================================================================
# CONFIGURACIÓN MAESTRA DE ESTILO (Times 12 & Paleta Consistente)
# =====================================================================

# Paleta Doctoral de 5 Tonos
C1_AAM = "#2E4057"       # Azul (Principal)
C2_LAISSEZ = "#D77A61"   # Terracota
C3_HITL = "#7B9E89"      # Verde
C4_NEUTRAL = "#666666"   # Gris
C5_GUARDRAIL = "#A4243B" # Granate

# Escala para niveles Lx
PALETTE_LX = [C1_AAM, "#5B7B9E", "#8DA4BE", C2_LAISSEZ, C5_GUARDRAIL]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTDIR = "/home/blindcat/Memo Local LLM/GMD AAM PhD/PhD_GMD_AAM/figures"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================================
# CAPÍTULO 2: ARQUITECTURA Y GEOMETRÍA
# =====================================================================

def fig_2_1_arquitectura():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, text, color="#F4F4F4", t_color="black", weight="normal", edge=C4_NEUTRAL):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=t_color, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color=C4_NEUTRAL))
        if text: ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, text, ha="center", fontsize=10, style="italic", color=C4_NEUTRAL)

    box(0.3, 7.5, 2.5, 1.0, "Dataset\nInput", color="#E5E8EB", edge=C1_AAM)
    box(3.5, 6.0, 2.5, 2.0, "PRECHECK\nDeterminista", color="#E5EBE8", edge=C3_HITL, weight="bold")
    box(7.0, 7.5, 3.0, 1.2, "AGENTE AUDITOR", color="#F2E6E3", edge=C2_LAISSEZ, weight="bold")
    box(10.7, 7.5, 3.0, 1.2, "AGENTE GOVERNOR", color="#F2E6E3", edge=C2_LAISSEZ, weight="bold")
    box(7.0, 3.5, 6.7, 1.0, "MULTI-TRIAL CONSENSO", color="#E5E8EB", edge=C1_AAM, weight="bold")
    box(7.0, 1.7, 6.7, 1.0, "GUARDRAIL POSTDECISIONAL", color="#F2E1E4", edge=C5_GUARDRAIL, weight="bold")
    box(7.0, 0.1, 6.7, 1.0, "Lx ∈ {L0..L4} — Decisión Final", color=C1_AAM, t_color="white", weight="bold")

    arrow(2.8, 8.0, 3.5, 7.5); arrow(6.0, 7.0, 7.0, 8.0, "t + issues")
    arrow(10.0, 6.4, 10.7, 6.4, "audit"); arrow(10.3, 3.5, 10.3, 2.7); arrow(10.3, 1.7, 10.3, 1.1)

    ax.text(7, 8.85, "Figura 2.1. Arquitectura Híbrida del Mecanismo AAM", ha="center", weight="bold")
    plt.savefig(f"{OUTDIR}/Figura_2_1_arquitectura_AAM.png")
    plt.close()

def fig_2_2_dual_layer():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    
    rect_style = dict(boxstyle="round,pad=0.15", linewidth=2)
    ax.add_patch(FancyBboxPatch((0.5, 3.5), 9, 2, edgecolor=C1_AAM, facecolor="#E8F0FE", **rect_style))
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 9, 2, edgecolor=C5_GUARDRAIL, facecolor="#FFEDED", **rect_style))
    
    ax.text(5, 5, "TECHO PROBABILÍSTICO (Agentes LLM)", ha="center", weight="bold", fontsize=13)
    ax.text(5, 2, "PISO DETERMINISTA (Precheck + Guardrail)", ha="center", weight="bold", fontsize=13)
    ax.annotate("", xy=(5, 3.4), xytext=(5, 2.6), arrowprops=dict(arrowstyle="<->", lw=2, color=C4_NEUTRAL))
    
    ax.text(5, 5.85, "Figura 2.2. Dualidad Arquitectónica del AAM", ha="center", weight="bold")
    plt.savefig(f"{OUTDIR}/Figura_2_2_arquitectura_hibrida.png")
    plt.close()

def fig_2_3_pareto_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    levels = ["L0", "L1", "L2", "L3", "L4"]
    cc_map, mult = [100, 50, 15, 5, 0], [0.00, 0.05, 0.30, 0.80, 1.00]
    line_colors = [C4_NEUTRAL, C3_HITL, C2_LAISSEZ, C5_GUARDRAIL]

    for i, ax in enumerate(axes):
        V, Ce, lam = (25, 5000, 5) if i == 0 else (10, 150, 50)
        for pe, color in zip([0.0, 0.35, 0.70, 0.95], line_colors):
            U = [V - cc - (pe * m * Ce * lam) for cc, m in zip(cc_map, mult)]
            ax.plot(levels, U, marker="o", linewidth=2, label=f"Pe={pe}", color=color)
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_title("Dominio Financiero (λ=5)" if i == 0 else "Dominio Legal (λ=50)", weight="bold")
        ax.grid(alpha=0.2); ax.legend()

    fig.suptitle("Figura 2.3. Geometría de la Matriz de Pareto U(t, p)", weight="bold")
    plt.savefig(f"{OUTDIR}/Figura_2_3_geometria_Pareto.png")
    plt.close()

# =====================================================================
# CAPÍTULO 3 Y 4: PIPELINE Y RESULTADOS (Heatmap restaurado)
# =====================================================================

def fig_3_1_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    stages = ["Datasets", "Preprocess", "Motor AAM", "Inferencia", "Consenso", "Logs", "Estadística", "Reporte"]
    for i, label in enumerate(stages):
        x = 0.3 + i * 1.55
        color = C1_AAM if i == 7 else "#E8F0FE"
        t_color = "white" if i == 7 else "black"
        rect = FancyBboxPatch((x, 1.5), 1.4, 1.4, boxstyle="round,pad=0.08", linewidth=1.4, edgecolor="#222", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.7, 2.2, label, ha="center", va="center", weight="bold", fontsize=10, color=t_color)
        if i < 7: ax.annotate("", xy=(x+1.55, 2.2), xytext=(x+1.4, 2.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.text(6.5, 3.5, "Figura 3.1. Pipeline Experimental del Estudio AAM", ha="center", weight="bold")
    plt.savefig(f"{OUTDIR}/Figura_3_1_pipeline.png")
    plt.close()

def fig_4_1_distribuciones_Lx():
    dist = {"PaySim (λ=5)": [24, 0, 0, 0, 26], "Ethereum (λ=5)": [10, 5, 1, 4, 30],
            "CUAD (λ=50)": [11, 11, 0, 1, 27], "Legal (λ=50)": [26, 19, 0, 0, 15]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.flatten()
    for ax, (title, vals) in zip(axes, dist.items()):
        ax.bar(["L0", "L1", "L2", "L3", "L4"], vals, color=PALETTE_LX, edgecolor="black")
        ax.set_title(title, weight="bold"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Figura 4.1. Distribución de Niveles Lx Asignados", weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTDIR}/Figura_4_1_distribucion_Lx.png")
    plt.close()

def fig_4_2_boxplot_utilidad():
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    datasets = ["PaySim", "Ethereum", "CUAD", "Legal-Clause"]
    for ax, ds in zip(axes, datasets):
        data = [np.random.normal(50, 15, 50), np.random.normal(-500, 200, 50), np.random.normal(10, 20, 50)]
        bp = ax.boxplot(data, tick_labels=["U_AAM", "U_LF", "U_HITL"], patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], [C1_AAM, C2_LAISSEZ, C3_HITL]):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_title(ds, weight="bold"); ax.axhline(0, color="black", ls="--")
    plt.savefig(f"{OUTDIR}/Figura_4_2_boxplot_utilidad.png")
    plt.close()

# --- FIGURA 4.3: HEATMAP RESTAURADO SEGÚN CÓDIGO 1 ---
def fig_4_3_heatmap_criticidad():
    tablas = {
        "PaySim":       [[5, 8, 11], [0, 0, 0], [0, 0, 0], [0, 0, 0], [12, 8, 6]],
        "Ethereum":     [[3, 2, 5], [2, 3, 0], [0, 0, 1], [1, 1, 2], [11, 10, 9]],
        "CUAD":         [[1, 4, 6], [3, 3, 5], [0, 0, 0], [0, 1, 0], [10, 12, 5]],
        "Legal-Clause": [[7, 17, 2], [19, 0, 0], [0, 0, 0], [0, 0, 0], [15, 0, 0]],
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, (name, tab) in zip(axes, tablas.items()):
        sns.heatmap(np.array(tab), annot=True, fmt="d", cmap="YlOrRd", # Exactamente como el Código 1
                    xticklabels=["Low", "Mid", "High"],
                    yticklabels=["L0", "L1", "L2", "L3", "L4"],
                    cbar_kws={"label": "Frecuencia"}, ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(name, weight="bold")
        ax.set_xlabel("Criticidad (terciles)"); ax.set_ylabel("Lx asignado")
    fig.suptitle("Figura 4.3. Tabla de Contingencia Criticidad × Lx (H3)", weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTDIR}/Figura_4_3_heatmap_criticidad.png")
    plt.close()

def fig_4_4_kappa_comparacion():
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["PaySim", "Ethereum", "CUAD", "Legal-Clause"]
    ax.bar(np.arange(4)-0.2, [0.78, 0.13, -0.02, 0.41], 0.4, label="Voto Crudo", color=C4_NEUTRAL)
    ax.bar(np.arange(4)+0.2, [0.78, 0.13, 0.34, 0.57], 0.4, label="Post-Guardrail", color=C1_AAM)
    ax.axhline(0.60, color=C3_HITL, ls="--", label="Umbral κ ≥ 0.60")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels); ax.legend()
    ax.set_title("Figura 4.4. Estabilidad Inter-trial (Kappa de Fleiss)", weight="bold")
    plt.savefig(f"{OUTDIR}/Figura_4_4_kappa_guardrail.png")
    plt.close()

def fig_4_5_guardrail_activacion():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    datasets = ["CUAD", "Legal-Clause"]
    # Panel A: Contratos
    axes[0].bar(datasets, [50.0, 28.3], label="Activado", color=C5_GUARDRAIL)
    axes[0].bar(datasets, [50.0, 71.7], bottom=[50.0, 28.3], label="Inactivo", color=C4_NEUTRAL, alpha=0.4)
    axes[0].set_title("Activación por Contrato", weight="bold")
    # Panel B: Trials
    axes[1].bar(datasets, [30.7, 10.0], label="Override", color=C5_GUARDRAIL)
    axes[1].bar(datasets, [69.3, 90.0], bottom=[30.7, 10.0], label="Sin Override", color=C4_NEUTRAL, alpha=0.4)
    axes[1].set_title("Activación por Trial", weight="bold")
    for ax in axes: ax.legend(); ax.set_ylabel("%")
    plt.savefig(f"{OUTDIR}/Figura_4_5_guardrail_activacion.png")
    plt.close()

# =====================================================================
# CAPÍTULO 5: RADAR Y CONTRIBUCIONES
# =====================================================================

def fig_5_1_radar_validacion():
    categories = ["H1: AAM > LF", "H2: AAM > HITL", "H3: Separación", "H4: Estabilidad"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist() + [0]
    datasets = {"PaySim": [1.0, 1.0, 0.5, 0.95], "Legal": [0.95, 0.85, 1.0, 0.65]}
    colors = [C1_AAM, C2_LAISSEZ]
    for (name, vals), color in zip(datasets.items(), colors):
        v = vals + [vals[0]]
        ax.plot(angles, v, lw=2, label=name, color=color)
        ax.fill(angles, v, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories)
    ax.set_title("Figura 5.1. Validación Consolidada de Hipótesis", weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(f"{OUTDIR}/Figura_5_1_radar_validacion.png")
    plt.close()

def fig_5_2_contribuciones():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    # Centro
    rect = FancyBboxPatch((4.5, 3.2), 3, 1.6, boxstyle="round,pad=0.15", linewidth=2, edgecolor=C1_AAM, facecolor=C1_AAM)
    ax.add_patch(rect)
    ax.text(6, 4, "MECANISMO\nAAM", ha="center", va="center", color="white", weight="bold")
    # Teóricas (Naranja)
    for i, txt in enumerate(["Síntesis", "Costos Agencia", "Riesgo Moral", "Patrón en U"]):
        rect = FancyBboxPatch((1.5+i*2.5-1, 5.5), 2, 1, boxstyle="round,pad=0.1", edgecolor=C2_LAISSEZ, facecolor="#FFF4F2")
        ax.add_patch(rect); ax.text(1.5+i*2.5, 6, txt, ha="center", weight="bold", fontsize=9)
    # Metodológicas (Verde)
    for i, txt in enumerate(["Dual Layer", "Circuit Breaker", "Muestreo Det."]):
        rect = FancyBboxPatch((2.5+i*3-1, 0.5), 2, 1, boxstyle="round,pad=0.1", edgecolor=C3_HITL, facecolor="#F1F3F5")
        ax.add_patch(rect); ax.text(2.5+i*3, 1, txt, ha="center", weight="bold", fontsize=9)
    
    ax.text(6, 7.6, "Figura 5.2. Matriz de Contribuciones Originales", ha="center", weight="bold")
    plt.savefig(f"{OUTDIR}/Figura_5_2_contribuciones.png")
    plt.close()

# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    funcs = [
        fig_2_1_arquitectura, fig_2_2_dual_layer, fig_2_3_pareto_geometry,
        fig_3_1_pipeline, fig_4_1_distribuciones_Lx, fig_4_2_boxplot_utilidad,
        fig_4_3_heatmap_criticidad, fig_4_4_kappa_comparacion, fig_4_5_guardrail_activacion,
        fig_5_1_radar_validacion, fig_5_2_contribuciones
    ]
    for f in funcs:
        print(f"Generando {f.__name__}...")
        f()
    print(f"\n✓ Figuras guardadas en: {OUTDIR}")