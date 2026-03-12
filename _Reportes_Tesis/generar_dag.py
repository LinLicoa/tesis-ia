"""
Genera una visualización del Grafo Acíclico Dirigido (DAG) de la Red Bayesiana.

El grafo muestra la estructura del modelo:
- 3 nodos de condición clínica (Estrés, Ansiedad, Depresión)
- 78 nodos de features (síntomas neutrosóficos agrupados por instrumento)
- Relaciones inter-condición basadas en evidencia clínica

Autor: Generado automáticamente
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para guardar a archivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_dag():
    fig, ax = plt.subplots(1, 1, figsize=(22, 16))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # ─── Colores ────────────────────────────────────────────────────
    COLOR_ESTRES     = '#f97316'   # Naranja
    COLOR_ANSIEDAD   = '#3b82f6'   # Azul
    COLOR_DEPRESION  = '#a855f7'   # Púrpura
    COLOR_FEAT       = '#1e293b'   # Gris oscuro para nodos feature
    COLOR_EDGE       = '#475569'   # Gris medio para aristas
    COLOR_INTER_EDGE = '#facc15'   # Amarillo para aristas inter-condición
    COLOR_TEXT        = '#f8fafc'   # Blanco

    # ─── Posiciones de los nodos principales ────────────────────────
    # Nivel superior: condiciones clínicas
    cond_y = 12
    pos_estres    = (11, cond_y)
    pos_ansiedad  = (4,  cond_y)
    pos_depresion = (18, cond_y)

    # ─── Dibujar features como arco de nodos debajo de cada condición ──
    def draw_feature_arc(ax, parent_pos, feat_start, feat_end, color, instrument_label, spread=3.0, depth=6.0):
        """Dibuja un arco de nodos feature debajo de un nodo padre."""
        n_feats = feat_end - feat_start
        feat_positions = []

        for idx, i in enumerate(range(feat_start, feat_end)):
            # Distribuir en un arco
            t = (idx / max(n_feats - 1, 1)) - 0.5  # [-0.5, 0.5]
            x = parent_pos[0] + t * spread * 2
            y = parent_pos[1] - depth + (1.0 - abs(t) * 2) * 1.5  # arco

            feat_positions.append((x, y, i))

            # Arista padre -> feature
            ax.annotate(
                '', xy=(x, y + 0.25), xytext=parent_pos,
                arrowprops=dict(
                    arrowstyle='->', color=color, alpha=0.15,
                    lw=0.5, connectionstyle='arc3,rad=0.0'
                )
            )

            # Nodo feature (pequeño)
            circle = plt.Circle((x, y), 0.18, color=COLOR_FEAT, ec=color, lw=0.8, alpha=0.7, zorder=5)
            ax.add_patch(circle)

        # Etiqueta del grupo de features
        center_x = parent_pos[0]
        min_y = min(p[1] for p in feat_positions)
        ax.text(center_x, min_y - 0.8, f'{instrument_label}\n({n_feats} features: Feat_{feat_start}…Feat_{feat_end-1})',
                ha='center', va='top', fontsize=8, color=color, fontweight='bold',
                fontstyle='italic', alpha=0.9)

    # Dibujar features de cada condición
    draw_feature_arc(ax, pos_ansiedad,  0,  21, COLOR_ANSIEDAD,
                     'GAD-7 (Ansiedad)', spread=2.8, depth=5.0)
    draw_feature_arc(ax, pos_estres,    48, 78, COLOR_ESTRES,
                     'PSS-10 (Estrés)', spread=3.2, depth=5.5)
    draw_feature_arc(ax, pos_depresion, 21, 48, COLOR_DEPRESION,
                     'PHQ-9 (Depresión)', spread=3.0, depth=5.5)

    # ─── Aristas inter-condición (las más importantes) ──────────────
    inter_edges = [
        (pos_estres,   pos_ansiedad,  'Estrés → Ansiedad'),
        (pos_estres,   pos_depresion, 'Estrés → Depresión'),
        (pos_ansiedad, pos_depresion, 'Ansiedad → Depresión'),
    ]

    for src, dst, label in inter_edges:
        # Calcular punto medio para la etiqueta
        mid_x = (src[0] + dst[0]) / 2
        mid_y = (src[1] + dst[1]) / 2 + 0.6

        ax.annotate(
            '', xy=dst, xytext=src,
            arrowprops=dict(
                arrowstyle='->', color=COLOR_INTER_EDGE, lw=2.5, alpha=0.85,
                connectionstyle='arc3,rad=0.25'
            ), zorder=8
        )
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
                fontsize=7.5, color=COLOR_INTER_EDGE, fontweight='bold', alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor=COLOR_INTER_EDGE, alpha=0.7, lw=0.8))

    # ─── Nodos de condición clínica (los nodos grandes) ──────────────
    def draw_condition_node(ax, pos, label, subtitle, color):
        """Dibuja un nodo de condición clínica con estilo premium."""
        # Halo
        halo = plt.Circle(pos, 1.1, color=color, alpha=0.08, zorder=2)
        ax.add_patch(halo)
        halo2 = plt.Circle(pos, 0.9, color=color, alpha=0.12, zorder=3)
        ax.add_patch(halo2)
        # Nodo principal
        main_circle = plt.Circle(pos, 0.7, color=color, ec='white', lw=2, alpha=0.95, zorder=10)
        ax.add_patch(main_circle)
        # Texto
        ax.text(pos[0], pos[1] + 0.1, label, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=11)
        ax.text(pos[0], pos[1] - 0.25, subtitle, ha='center', va='center',
                fontsize=7, color='white', alpha=0.8, zorder=11)

    draw_condition_node(ax, pos_ansiedad,  'Ansiedad',  'GAD-7',  COLOR_ANSIEDAD)
    draw_condition_node(ax, pos_estres,    'Estrés',    'PSS-10', COLOR_ESTRES)
    draw_condition_node(ax, pos_depresion, 'Depresión', 'PHQ-9',  COLOR_DEPRESION)

    # ─── Título ─────────────────────────────────────────────────────
    ax.text(11, 15, 'Grafo Acíclico Dirigido (DAG)',
            ha='center', va='center', fontsize=22, color=COLOR_TEXT,
            fontweight='bold', fontfamily='sans-serif')
    ax.text(11, 14.3, 'Estructura de la Red Bayesiana — Modelo Neutrosófico-Bayesiano',
            ha='center', va='center', fontsize=12, color='#94a3b8',
            fontfamily='sans-serif')

    # ─── Leyenda ────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=COLOR_ANSIEDAD, edgecolor='white', label='Ansiedad (GAD-7) → 21 features'),
        mpatches.Patch(facecolor=COLOR_ESTRES, edgecolor='white', label='Estrés (PSS-10) → 30 features'),
        mpatches.Patch(facecolor=COLOR_DEPRESION, edgecolor='white', label='Depresión (PHQ-9) → 27 features'),
        mpatches.Patch(facecolor=COLOR_INTER_EDGE, edgecolor='white', label='Relación inter-condición (3 aristas)'),
        mpatches.Patch(facecolor=COLOR_FEAT, edgecolor='white', label='Feature neutrosófica (T, I, F)'),
    ]
    legend = ax.legend(handles=legend_items, loc='lower center',
                       ncol=3, fontsize=9, framealpha=0.6,
                       facecolor='#1e293b', edgecolor='#475569',
                       labelcolor=COLOR_TEXT,
                       bbox_to_anchor=(0.5, -0.02))
    legend.get_frame().set_linewidth(0.5)

    # ─── Estadísticas del modelo ────────────────────────────────────
    stats_text = (
        "Total de nodos: 81  •  Total de aristas: 81\n"
        "Features por instrumento:  GAD-7 → 7×3=21  |  PHQ-9 → 9×3=27  |  PSS-10 → 10×3=30\n"
        "Cada feature es una tripleta neutrosófica (Verdad, Indeterminación, Falsedad)"
    )
    ax.text(11, 3.8, stats_text, ha='center', va='top', fontsize=8.5,
            color='#94a3b8', fontfamily='monospace', linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor='#334155', alpha=0.8))

    # ─── Config del eje ─────────────────────────────────────────────
    ax.set_xlim(-1, 23)
    ax.set_ylim(3, 16)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=1.0)

    # Guardar
    output_path = os.path.join(OUTPUT_DIR, 'DAG_Red_Bayesiana.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"✅ DAG guardado en: {output_path}")
    return output_path


if __name__ == '__main__':
    path = draw_dag()
    print(f"Archivo generado: {path}")
