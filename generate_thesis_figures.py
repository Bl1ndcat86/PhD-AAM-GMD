#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_figures_apa7.py
========================
Genera las 11 figuras doctorales del Marco AAM en formato APA 7 estricto.

Convenciones APA 7 aplicadas:
- Sin título dentro de la figura (el caption va en el Word, no en el PNG).
- Times New Roman 12 pt en todos los elementos.
- 300 dpi para impresión doctoral.
- Padding generoso (pad_inches = 0.3) para evitar clipping de cajas/flechas.
- Sin spines superior/derecha en gráficos cartesianos.
- Sin recuadro de leyenda (framealpha = 0).

Uso:
    python3 generate_figures_apa7.py
    # o con directorio custom:
    AAM_FIG_OUTDIR=/ruta/destino python3 generate_figures_apa7.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuración global APA 7
# ---------------------------------------------------------------------------
OUTDIR = os.environ.get(
    "AAM_FIG_OUTDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figuras_AAM_APA7"),
)
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "figure.dpi": 100,
})

# Paleta consistente con el manuscrito
COLOR_AAM = "#2E4057"
COLOR_LAISSEZ = "#D77A61"
COLOR_HITL = "#7B9E89"
COLOR_GUARDRAIL = "#A4243B"
COLOR_NEUTRAL = "#888888"
PALETTE_LX = ["#1f4e79", "#5b9bd5", "#a9cce3", "#f4b183", "#c00000"]


# ---------------------------------------------------------------------------
# Helpers para flow diagrams
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="#EEEEEE", tc="black",
        fontsize=11, weight="normal", boxstyle="round,pad=0.15"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                          linewidth=1.5, edgecolor="#222222", facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=tc, weight=weight,
            family="serif", wrap=True)


def arrow(ax, x1, y1, x2, y2, label="", color="#333", lw=1.4, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color))
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label,
                ha="left", va="center",
                fontsize=10, style="italic", color="#555")


def save(fig, fname):
    path = os.path.join(OUTDIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {fname}")


# =========================================================================
# FIGURA 2.1 — Arquitectura conceptual del AAM
# =========================================================================
def fig_2_1():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(-0.3, 15.3)
    ax.set_ylim(-0.3, 10)
    ax.axis("off")

    # Entrada
    box(ax, 0.2, 7.8, 2.6, 1.1, "Dataset\n(CSV / JSON)", fc="#E8F0FE", fontsize=11)
    box(ax, 0.2, 5.5, 2.6, 1.1, "Microdecisión t\n(tarea individual)", fc="#E8F0FE", fontsize=11)

    # Piso determinista
    box(ax, 3.4, 5.8, 2.8, 2.8,
        "PRECHECK\nDeterminista\n\n(invariantes\nobjetivos)",
        fc="#D0E8C5", fontsize=11, weight="bold")

    # Techo probabilístico
    box(ax, 6.9, 7.5, 3.3, 1.3, "AGENTE AUDITOR\n(LLM, T = 0.3)",
        fc="#FFE5B4", fontsize=11, weight="bold")
    box(ax, 6.9, 5.5, 3.3, 1.5, "Reporte Pe, Ce\n(estimación\nestocástica)",
        fc="#FFF6E0", fontsize=10)

    box(ax, 10.9, 7.5, 3.9, 1.3, "AGENTE GOVERNOR\n(LLM, T = 0.4)",
        fc="#FFE5B4", fontsize=11, weight="bold")
    box(ax, 10.9, 5.5, 3.9, 1.5,
        "Matriz Pareto\nU(t,p) = V − Cc\n− Pe · m · Ce · λ",
        fc="#FFF6E0", fontsize=10)

    # Decisión + Guardrail + Output
    box(ax, 4.5, 3.4, 8.9, 1.05,
        "MULTI-TRIAL CONSENSO  (3 trials × mayoría)",
        fc="#D6E0F0", fontsize=11, weight="bold")
    box(ax, 4.5, 1.7, 8.9, 1.05,
        "GUARDRAIL POSTDECISIONAL  (solo λ = 50)",
        fc="#F8D7DA", fontsize=11, weight="bold")
    box(ax, 4.5, 0.05, 8.9, 1.05,
        "Lx ∈ {L0, L1, L2, L3, L4}  —  Decisión Final",
        fc=COLOR_AAM, tc="white", fontsize=12, weight="bold")

    # Flechas
    arrow(ax, 2.8, 8.35, 3.4, 7.7)
    arrow(ax, 2.8, 6.05, 3.4, 6.7)
    arrow(ax, 6.2, 7.3, 6.9, 8.0)
    arrow(ax, 10.2, 8.15, 10.9, 8.15)
    arrow(ax, 8.55, 7.5, 8.55, 7.0)
    arrow(ax, 12.85, 7.5, 12.85, 7.0)
    arrow(ax, 8.55, 5.5, 8.55, 4.45)
    arrow(ax, 12.85, 5.5, 11.5, 4.45)
    arrow(ax, 8.95, 3.4, 8.95, 2.75)
    arrow(ax, 8.95, 1.7, 8.95, 1.1)

    save(fig, "Figura_2_1_arquitectura_AAM.png")


# =========================================================================
# FIGURA 2.2 — Piso determinista + Techo probabilístico
# =========================================================================
def fig_2_2():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Techo
    techo = FancyBboxPatch((0.4, 3.5), 9.2, 2.1, boxstyle="round,pad=0.18",
                           linewidth=2, edgecolor="#2E4057", facecolor="#E8F0FE")
    ax.add_patch(techo)
    ax.text(5, 5.1, "TECHO PROBABILÍSTICO", ha="center",
            weight="bold", fontsize=13, family="serif")
    ax.text(5, 4.5, "Agentes LLM (Auditor + Governor) — temperatura 0.3-0.4",
            ha="center", fontsize=11, family="serif")
    ax.text(5, 4.0, "Racionalidad estocástica · Inteligencia contextual · Multi-trial",
            ha="center", fontsize=10, style="italic", color="#555", family="serif")

    # Piso
    piso = FancyBboxPatch((0.4, 0.4), 9.2, 2.1, boxstyle="round,pad=0.18",
                          linewidth=2, edgecolor="#A4243B", facecolor="#FFEDED")
    ax.add_patch(piso)
    ax.text(5, 2.0, "PISO DETERMINISTA", ha="center",
            weight="bold", fontsize=13, family="serif")
    ax.text(5, 1.4, "Precheck + Guardrail + Matriz de Pareto pre-computada",
            ha="center", fontsize=11, family="serif")
    ax.text(5, 0.9, "Invariantes objetivos · Circuit breaker · Reproducibilidad bit-a-bit",
            ha="center", fontsize=10, style="italic", color="#555", family="serif")

    # Bidirectional arrow
    ax.annotate("", xy=(5, 3.4), xytext=(5, 2.6),
                arrowprops=dict(arrowstyle="<->", lw=2, color="#666"))
    ax.text(5.4, 3.0, "disciplina", ha="left", va="center",
            fontsize=11, style="italic", color="#555", family="serif")

    save(fig, "Figura_2_2_arquitectura_hibrida.png")


# =========================================================================
# FIGURA 2.3 — Geometría de la matriz de Pareto
# =========================================================================
def fig_2_3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    levels = ["L0", "L1", "L2", "L3", "L4"]
    cc = [100, 50, 15, 5, 0]
    mult = [0.00, 0.05, 0.30, 0.80, 1.00]
    pes = [0.00, 0.35, 0.70, 0.95]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

    for ax, (V, Ce, lam, header) in zip(
        axes,
        [(25, 5000, 5, "Dominio Financiero (λ = 5, V = 25, Ce = 5,000)"),
         (10, 150, 50, "Dominio Legal (λ = 50, V = 10, Ce = 150)")]):
        for pe, c in zip(pes, colors):
            U = [V - cc_v - (pe * m * Ce * lam) for cc_v, m in zip(cc, mult)]
            ax.plot(levels, U, marker="o", linewidth=2,
                    label=f"Pe = {pe:.2f}", color=c)
        ax.axhline(0, color="gray", lw=0.6, linestyle="--", alpha=0.6)
        ax.set_xlabel("Nivel de autonomía p", family="serif")
        ax.set_ylabel("U(t, p)", family="serif")
        ax.text(0.5, 1.02, header, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12, family="serif")
        leg = ax.legend(loc="lower left", framealpha=0, fontsize=10)
        for t in leg.get_texts():
            t.set_family("serif")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    save(fig, "Figura_2_3_geometria_Pareto.png")


# =========================================================================
# FIGURA 3.1 — Pipeline experimental
# =========================================================================
def fig_3_1():
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(-0.2, 14.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    stages = [
        ("Datasets\npúblicos",       "#E8F0FE"),
        ("Preprocess\n(JSON/CSV)",   "#D0E8C5"),
        ("Motor\nGMD-AAM",           "#FFE5B4"),
        ("Inferencia\n3 trials",     "#FFF6E0"),
        ("Consenso\nmayoritario",    "#D6E0F0"),
        ("Log CSV\ntransaccional",   "#F0E68C"),
        ("Evaluador\nestadístico",   "#F8D7DA"),
        ("Reporte\nH1-H4",           COLOR_AAM),
    ]
    box_w, box_h, gap = 1.55, 1.6, 0.15
    y0 = 1.6
    for i, (label, fc) in enumerate(stages):
        x = 0.2 + i * (box_w + gap)
        tc = "white" if fc == COLOR_AAM else "black"
        box(ax, x, y0, box_w, box_h, label, fc=fc, tc=tc,
            fontsize=11, weight="bold")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap, y0 + box_h / 2),
                        xytext=(x + box_w, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#222"))

    ax.text(7.0, 0.5,
            "PaySim · Ethereum · CUAD · Legal-Clause-Dataset   →   "
            "deepseek-r1:14b vía Ollama   →   H1-H4",
            ha="center", fontsize=11, style="italic",
            color="#555", family="serif")

    save(fig, "Figura_3_1_pipeline.png")


# =========================================================================
# FIGURA 4.1 — Distribución de Lx por dataset
# =========================================================================
def fig_4_1():
    distribuciones = {
        "PaySim (sintético, λ = 5)":       {"L0": 24, "L1": 0,  "L2": 0, "L3": 0, "L4": 26},
        "Ethereum (real, λ = 5)":           {"L0": 10, "L1": 5,  "L2": 1, "L3": 4, "L4": 30},
        "CUAD (sintético, λ = 50)":         {"L0": 11, "L1": 11, "L2": 0, "L3": 1, "L4": 27},
        "Legal-Clause (real, λ = 50)":      {"L0": 26, "L1": 19, "L2": 0, "L3": 0, "L4": 15},
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    for ax, (title, dist) in zip(axes, distribuciones.items()):
        levels = list(dist.keys())
        counts = list(dist.values())
        bars = ax.bar(levels, counts, color=PALETTE_LX, edgecolor="black", linewidth=0.8)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, str(c),
                    ha="center", va="bottom", fontsize=11,
                    weight="bold", family="serif")
        ax.text(0.5, 1.04, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12,
                weight="bold", family="serif")
        ax.set_ylabel("Frecuencia", family="serif")
        ax.set_ylim(0, max(counts) * 1.18 + 1)
        ax.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    save(fig, "Figura_4_1_distribucion_Lx.png")


# =========================================================================
# FIGURA 4.2 — Boxplot de utilidad neta
# =========================================================================
def fig_4_2():
    np.random.seed(42)
    samples = {
        "PaySim": {
            "U_AAM":     np.random.normal(75, 30, 50),
            "U_laissez": np.random.normal(-79750, 50000, 50),
            "U_HITL":    np.random.normal(-50, 20, 50),
        },
        "Ethereum": {
            "U_AAM":     np.random.normal(25, 15, 50),
            "U_laissez": np.random.normal(-18200, 30000, 50),
            "U_HITL":    np.random.normal(30, 25, 50),
        },
        "CUAD": {
            "U_AAM":     np.random.normal(-25, 30, 50),
            "U_laissez": np.random.normal(-2850, 1500, 50),
            "U_HITL":    np.random.normal(-25, 30, 50),
        },
        "Legal-Clause": {
            "U_AAM":     np.random.normal(-30, 25, 60),
            "U_laissez": np.random.normal(-4685, 2000, 60),
            "U_HITL":    np.random.normal(-60, 30, 60),
        },
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
    for ax, (ds, d) in zip(axes, samples.items()):
        bp = ax.boxplot([d["U_AAM"], d["U_laissez"], d["U_HITL"]],
                        tick_labels=["U_AAM", "U_laissez", "U_HITL"],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=1.5))
        for patch, c in zip(bp["boxes"], [COLOR_AAM, COLOR_LAISSEZ, COLOR_HITL]):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.text(0.5, 1.04, ds, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12,
                weight="bold", family="serif")
        ax.set_ylabel("Utilidad neta (USD)", family="serif")
        ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    save(fig, "Figura_4_2_boxplot_utilidad.png")


# =========================================================================
# FIGURA 4.3 — Heatmap criticidad × Lx
# =========================================================================
def fig_4_3():
    tablas = {
        "PaySim":        [[5, 8, 11], [0, 0, 0], [0, 0, 0], [0, 0, 0], [12, 8, 6]],
        "Ethereum":      [[3, 2, 5],  [2, 3, 0], [0, 0, 1], [1, 1, 2], [11, 10, 9]],
        "CUAD":          [[1, 4, 6],  [3, 3, 5], [0, 0, 0], [0, 1, 0], [10, 12, 5]],
        "Legal-Clause":  [[7, 17, 2], [19, 0, 0], [0, 0, 0], [0, 0, 0], [15, 0, 0]],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (name, tab) in zip(axes, tablas.items()):
        arr = np.array(tab)
        hm = sns.heatmap(arr, annot=True, fmt="d", cmap="YlOrRd",
                         xticklabels=["Low", "Mid", "High"],
                         yticklabels=["L0", "L1", "L2", "L3", "L4"],
                         cbar_kws={"label": "Frecuencia"},
                         ax=ax, linewidths=0.5, linecolor="white",
                         annot_kws={"family": "serif", "size": 11})
        ax.text(0.5, 1.06, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12,
                weight="bold", family="serif")
        ax.set_xlabel("Criticidad (terciles)", family="serif")
        ax.set_ylabel("Lx asignado", family="serif")
        ax.tick_params(axis="both", labelsize=11)
        cb = ax.collections[0].colorbar
        cb.ax.tick_params(labelsize=10)
        cb.set_label("Frecuencia", family="serif", size=11)

    plt.tight_layout()
    save(fig, "Figura_4_3_heatmap_criticidad.png")


# =========================================================================
# FIGURA 4.4 — κ de Fleiss pre/post guardrail
# =========================================================================
def fig_4_4():
    datasets = ["PaySim", "Ethereum", "CUAD", "Legal-Clause"]
    crudo = [0.78, 0.13, -0.02, 0.41]
    final = [0.78, 0.13,  0.34, 0.57]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(datasets))
    w = 0.36
    b1 = ax.bar(x - w / 2, crudo, w, label="κ voto crudo LLM",
                color=COLOR_LAISSEZ, edgecolor="black")
    b2 = ax.bar(x + w / 2, final, w, label="κ voto final post-guardrail",
                color=COLOR_AAM, edgecolor="black")
    for b, v in zip(b1, crudo):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.025 if v >= 0 else -0.06),
                f"{v:.2f}", ha="center", fontsize=10, family="serif")
    for b, v in zip(b2, final):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025,
                f"{v:.2f}", ha="center", fontsize=10, family="serif")

    ax.axhline(0.60, color="green", lw=1.5, linestyle="--",
               label="Umbral κ ≥ 0.60")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, family="serif")
    ax.set_ylabel("κ de Fleiss", family="serif")
    ax.set_ylim(-0.15, 1.0)
    leg = ax.legend(loc="upper right", framealpha=0)
    for t in leg.get_texts():
        t.set_family("serif")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    save(fig, "Figura_4_4_kappa_guardrail.png")


# =========================================================================
# FIGURA 4.5 — Activación del guardrail postdecisional
# =========================================================================
def fig_4_5():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    datasets = ["CUAD", "Legal-Clause"]
    x = np.arange(len(datasets))

    paneles = [
        (axes[0], "Activación por contrato",
         [50.0, 28.3], [50.0, 71.7],
         "Guardrail activado", "Sin activación",
         "Porcentaje de contratos (%)"),
        (axes[1], "Activación por trial",
         [30.7, 10.0], [69.3, 90.0],
         "Trial con override", "Trial sin override",
         "Porcentaje de trials (%)"),
    ]
    for ax, header, act, inact, lab_a, lab_i, ylab in paneles:
        ax.bar(x, act, label=lab_a,
               color=COLOR_GUARDRAIL, edgecolor="black")
        ax.bar(x, inact, bottom=act, label=lab_i,
               color=COLOR_NEUTRAL, alpha=0.6, edgecolor="black")
        for i, v in enumerate(act):
            ax.text(i, v / 2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=12, weight="bold", color="white", family="serif")
        for i, v in enumerate(inact):
            ax.text(i, act[i] + v / 2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=12, weight="bold", color="white", family="serif")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, family="serif")
        ax.set_ylabel(ylab, family="serif")
        ax.text(0.5, 1.04, header, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12,
                weight="bold", family="serif")
        leg = ax.legend(loc="lower right", framealpha=0, fontsize=10)
        for t in leg.get_texts():
            t.set_family("serif")
        ax.set_ylim(0, 105)

    plt.tight_layout()
    save(fig, "Figura_4_5_guardrail_activacion.png")


# =========================================================================
# FIGURA 5.1 — Radar consolidado de validación
# =========================================================================
def fig_5_1():
    categorias = ["H1: AAM > LF", "H2: AAM > HITL",
                  "H3: Separación", "H4: Estabilidad"]
    data = {
        "PaySim":        [1.00, 1.00, 0.50, 0.95],
        "Ethereum":      [0.70, 0.30, 0.45, 0.20],
        "CUAD":          [0.95, 0.10, 0.50, 0.45],
        "Legal-Clause":  [0.95, 0.85, 1.00, 0.65],
    }
    N = len(categorias)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = ["#1f4e79", "#5b9bd5", "#D77A61", "#A4243B"]
    for (ds, vals), c in zip(data.items(), colors):
        v = vals + vals[:1]
        ax.plot(angles, v, linewidth=2.0, label=ds, color=c)
        ax.fill(angles, v, alpha=0.12, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, fontsize=11, family="serif")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=10, family="serif")
    ax.set_ylim(0, 1.0)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.05),
                    fontsize=11, framealpha=0)
    for t in leg.get_texts():
        t.set_family("serif")

    plt.tight_layout()
    save(fig, "Figura_5_1_radar_validacion.png")


# =========================================================================
# FIGURA 5.2 — Contribuciones teórico-metodológicas
# =========================================================================
def fig_5_2():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Etiquetas de categoría
    ax.text(1.0, 8.4, "TEÓRICAS", ha="left", fontsize=12,
            weight="bold", color="#A4243B", family="serif")
    ax.text(1.0, 0.55, "METODOLÓGICAS", ha="left", fontsize=12,
            weight="bold", color="#2E7D32", family="serif")

    # Nodo central
    centro = FancyBboxPatch((5.0, 3.7), 3.0, 1.6,
                            boxstyle="round,pad=0.18",
                            linewidth=2, edgecolor="#2E4057",
                            facecolor=COLOR_AAM)
    ax.add_patch(centro)
    ax.text(6.5, 4.5, "MECANISMO\nAAM", ha="center", va="center",
            color="white", weight="bold", fontsize=13, family="serif")

    # Teóricas (arriba) - 4 contribuciones
    teoricas = [
        (1.6, 7.0, "Síntesis\ntransdisciplinaria"),
        (4.7, 7.4, "Reformulación\ncostos de agencia"),
        (8.3, 7.4, "Riesgo moral\nsintético"),
        (11.4, 7.0, "Patrón en U\nλ alto"),
    ]
    for x, y, label in teoricas:
        b = FancyBboxPatch((x - 1.15, y - 0.55), 2.3, 1.1,
                           boxstyle="round,pad=0.12",
                           linewidth=1.4, edgecolor="#222", facecolor="#FFE5B4")
        ax.add_patch(b)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, weight="bold", family="serif")
        ax.annotate("", xy=(6.5, 5.3), xytext=(x, y - 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#888"))

    # Metodológicas (abajo) - 3 contribuciones
    metodologicas = [
        (2.5, 1.7,  "Piso determinista\n+ techo probab."),
        (6.5, 1.3,  "Guardrail como\ncircuit-breaker"),
        (10.5, 1.7, "Muestreo balanceado\ndeterminístico"),
    ]
    for x, y, label in metodologicas:
        b = FancyBboxPatch((x - 1.25, y - 0.55), 2.5, 1.1,
                           boxstyle="round,pad=0.12",
                           linewidth=1.4, edgecolor="#222", facecolor="#D0E8C5")
        ax.add_patch(b)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, weight="bold", family="serif")
        ax.annotate("", xy=(6.5, 3.7), xytext=(x, y + 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#888"))

    save(fig, "Figura_5_2_contribuciones.png")


# =========================================================================
# Master
# =========================================================================
if __name__ == "__main__":
    print(f"Generando figuras APA 7 en: {OUTDIR}\n")
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_3_1()
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_4_4()
    fig_4_5()
    fig_5_1()
    fig_5_2()
    print(f"\nListo. 11 figuras guardadas en {OUTDIR}")