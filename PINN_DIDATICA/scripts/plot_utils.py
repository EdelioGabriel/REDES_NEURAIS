"""
Este script foi feito totalmente por IA, por meio do seguinte prompt

Quero esses plots:

 - curvas de aprendizado
 - mapas de calor (solução predita, analítica e erro absoluto)
 - perfis 1d
 - distribuição de pontos

use o plotly

Me retorne o script completo. Quero um layout simples e minimalista
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np

# ── Dimensões e fonte ──────────────────────────────────────────────────────────

FONT_SIZE   = 16          # tamanho base — legível em artigo ABNT
FONT_FAMILY = "Arial"
SQ_SIZE     = 500         # lado do plot quadrado (px)

AXIS_COMMON = dict(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,           # borda nos quatro lados → quadrado
    ticks="inside",
    ticklen=5,
    tickwidth=1.5,
    tickcolor="black",
)

LAYOUT_BASE = dict(
    template="simple_white",
    font=dict(family=FONT_FAMILY, size=FONT_SIZE),
    margin=dict(l=80, r=60, t=70, b=70),
)

COLORSCALE = "RdBu_r"

# Paleta de linhas (3 cortes)
COLORS = ["#2C3E50", "#E74C3C", "#2980B9"]

# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _square_axis(title, log=False):
    """Retorna dict de eixo com borda e fonte ABNT."""
    ax = dict(**AXIS_COMMON, title=dict(text=title, font=dict(size=FONT_SIZE)))
    if log:
        ax["type"] = "log"
    return ax


# ── 1. Curvas de aprendizado ───────────────────────────────────────────────────

def plot_loss(history):
    """
    Plota as curvas de aprendizado: loss total, loss_data e loss_pde.

    Args:
        history: dicionário {'loss': [], 'loss_data': [], 'loss_pde': []}
    """
    epochs = list(range(len(history["loss"])))

    fig = go.Figure()

    styles = [
        ("Total",  "#2C3E50", "solid"),
        ("Dados",  "#E74C3C", "dash"),
        ("PDE",    "#2980B9", "dot"),
    ]
    keys = ["loss", "loss_data", "loss_pde"]

    for (name, color, dash), key in zip(styles, keys):
        fig.add_trace(go.Scatter(
            x=epochs, y=history[key],
            mode="lines", name=name,
            line=dict(color=color, width=2.5, dash=dash),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Curvas de aprendizado", font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis("Época"),
        yaxis=_square_axis("Perda", log=True),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=0.97, y=0.97, xanchor="right", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.update_yaxes(
        type="log",
        dtick=1,              # só potências de 10
        exponentformat="power",
        showexponent="all",
    )

    fig.show()


# ── 2. Mapas de calor ─────────────────────────────────────────────────────────

def plot_heatmaps(model, analytical_fn, device, n_grid=100):
    """
    Plota mapas de calor: predição, solução analítica e erro absoluto.

    Predição e Analítica compartilham escala de cor → barra única (direita).
    Erro absoluto tem sua própria escala, posicionada sem sobrepor imagens.

    Args:
        model:         rede neural treinada
        analytical_fn: função de solução analítica
        device:        dispositivo de execução
        n_grid:        resolução da grade
    """
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)

    X_flat = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        U_pred = model(X_flat).cpu().numpy().reshape(n_grid, n_grid)

    U_anal = analytical_fn(X_flat).cpu().numpy().reshape(n_grid, n_grid)
    E_abs  = np.abs(U_pred - U_anal)

    # Escala compartilhada para predição e analítica
    vmin = min(U_pred.min(), U_anal.min())
    vmax = max(U_pred.max(), U_anal.max())

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Predição", "Analítica", "Erro absoluto"],
        horizontal_spacing=0.15,   # espaço extra para as barras não invadirem
    )

    # -- Predição: sem barra própria (a barra compartilhada será adicionada manualmente)
    fig.add_trace(go.Heatmap(
        z=U_pred, x=x, y=y,
        colorscale=COLORSCALE,
        zmin=vmin, zmax=vmax,
        showscale=False,
    ), row=1, col=1)

    # -- Analítica: exibe a barra compartilhada (col 2, margem direita do subplot)
    fig.add_trace(go.Heatmap(
        z=U_anal, x=x, y=y,
        colorscale=COLORSCALE,
        zmin=vmin, zmax=vmax,
        showscale=True,
        colorbar=dict(
            len=0.85,
            thickness=14,
            x=0.65,          # logo à direita do subplot 2 sem invadir o 3
            xpad=6,
            title=dict(text="u", font=dict(size=FONT_SIZE - 1), side="right"),
            tickfont=dict(size=FONT_SIZE - 2),
        ),
    ), row=1, col=2)

    # -- Erro absoluto: barra independente, encostada na margem direita da figura
    fig.add_trace(go.Heatmap(
        z=E_abs, x=x, y=y,
        colorscale="Reds",
        zmin=0, zmax=E_abs.max(),
        showscale=True,
        colorbar=dict(
            len=0.85,
            thickness=14,
            x=1.01,          # margem direita da figura
            xpad=6,
            title=dict(text="|err|", font=dict(size=FONT_SIZE - 1), side="right"),
            tickfont=dict(size=FONT_SIZE - 2),
        ),
    ), row=1, col=3)

    # Bordas nos eixos de cada subplot
    for col in [1, 2, 3]:
        fig.update_xaxes(
            **AXIS_COMMON,
            title_text="x",
            title_font=dict(size=FONT_SIZE),
            range=[0, 1],             
            constrain="domain",        
            row=1, col=col,
        )

        fig.update_yaxes(
            **AXIS_COMMON,
            title_text="y" if col == 1 else "",
            title_font=dict(size=FONT_SIZE),
            range=[0, 1],
            constrain="domain",        
            scaleanchor=f"x{'' if col==1 else col}",
            scaleratio=1,
            row=1, col=col,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Solução — Equação de Laplace 2D",
            font=dict(size=FONT_SIZE + 2),
        ),
        height=520,
        width=1050
    )

    # Título dos subplots com fonte ABNT
    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


# ── 3. Perfis 1D (subplots) ───────────────────────────────────────────────────

def plot_profiles(model, analytical_fn, device, slices=None):
    """
    Plota perfis 1D em subplots individuais — um subplot por corte em y.

    Layout:
        3 cortes → 1 linha × 3 colunas  (não-quadrado, conforme solicitado)
        4 cortes → 2 linhas × 2 colunas (quadrado)
        Qualquer outro número → 1 linha × N colunas

    Args:
        model:         rede neural treinada
        analytical_fn: função de solução analítica
        device:        dispositivo de execução
        slices:        lista de valores de y para os cortes (default: [0.25, 0.5, 0.75])
    """
    if slices is None:
        slices = [0.25, 0.5, 0.75]

    n = len(slices)

    # Decide layout
    if n == 4:
        rows, cols = 2, 2
    else:
        rows, cols = 1, n

    subplot_titles = [f"y = {y_val}" for y_val in slices]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    x_pts = np.linspace(0, 1, 200)

    for idx, (y_val, color) in enumerate(zip(slices, COLORS[:n] + COLORS[:max(0, n-3)])):
        row = idx // cols + 1
        col = idx %  cols + 1

        X_slice = torch.tensor(
            np.stack([x_pts, np.full_like(x_pts, y_val)], axis=1),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            U_pred = model(X_slice).cpu().numpy().ravel()

        U_anal = analytical_fn(X_slice).cpu().numpy().ravel()

        fig.add_trace(go.Scatter(
            x=x_pts, y=U_pred,
            mode="lines",
            name="Predição",
            legendgroup="pred",
            line=dict(color=color, width=2.5),
            showlegend=(idx == 0),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x_pts, y=U_anal,
            mode="lines",
            name="Analítica",
            legendgroup="anal",
            line=dict(color=color, width=2.5, dash="dash"),
            showlegend=(idx == 0),
        ), row=row, col=col)

    # Bordas em todos os eixos
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                **AXIS_COMMON,
                title_text="x",
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )
            fig.update_yaxes(
                **AXIS_COMMON,
                title_text="u(x, y)" if c == 1 else "",
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )

    # Dimensões: quadrado se 4 perfis, retangular caso contrário
    if n == 4:
        fig_w, fig_h = SQ_SIZE + 80, SQ_SIZE + 80
    else:
        fig_w = cols * (SQ_SIZE // 2) + 80
        fig_h = SQ_SIZE // 2 + 120

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Perfis 1D — cortes horizontais",
            font=dict(size=FONT_SIZE + 2),
        ),
        width=fig_w,
        height=fig_h,
        legend=dict(
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.update_layout(
        legend=dict(
            x=1.02,            # > 1 joga pra fora
            y=1,
            xanchor="left",
            yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        )
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


# ── 4. Distribuição de pontos ─────────────────────────────────────────────────

def plot_points(X_col, X_bc):
    """
    Plota a distribuição dos pontos de colocação e de contorno.

    Args:
        X_col: tensor de shape (N_c, 2) — pontos de colocação
        X_bc:  tensor de shape (4*N_b, 2) — pontos de contorno
    """
    X_col_np = X_col.detach().cpu().numpy()
    X_bc_np  = X_bc.detach().cpu().numpy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_col_np[:, 0], y=X_col_np[:, 1],
        mode="markers", name="Colocação",
        marker=dict(color="#2980B9", size=5, opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=X_bc_np[:, 0], y=X_bc_np[:, 1],
        mode="markers", name="Contorno",
        marker=dict(color="#E74C3C", size=7, symbol="square"),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Distribuição dos pontos de amostragem",
            font=dict(size=FONT_SIZE + 2),
        ),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text="x", font=dict(size=FONT_SIZE)),
            range=[-0.05, 1.05],
        ),
        yaxis=dict(
            **AXIS_COMMON,
            title=dict(text="y", font=dict(size=FONT_SIZE)),
            range=[-0.05, 1.05],
            scaleanchor="x",
            scaleratio=1,
        ),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=0.97, y=0.97, xanchor="right", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.update_layout(
        legend=dict(
            x=1.02,            # > 1 joga pra fora
            y=1,
            xanchor="left",
            yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        )
    )

    fig.show()